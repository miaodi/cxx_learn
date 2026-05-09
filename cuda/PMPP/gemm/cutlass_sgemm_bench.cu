// ===========================================================================
// CUTLASS SIMT SGEMM — systematic peak-finding benchmark.
//
// Sweeps tile configurations × matrix sizes to find the CUDA-core (SIMT)
// FP32 throughput ceiling.  cuBLAS is the vendor baseline.
//
// Build:
//   make -j PMPP_cutlass_sgemm_bench
// Run all:
//   ./PMPP_cutlass_sgemm_bench
// Run only a specific size:
//   ./PMPP_cutlass_sgemm_bench --benchmark_filter=".*4096"
// ===========================================================================

#include <cutlass/cutlass.h>
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/layout/matrix.h>

#include <benchmark/benchmark.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <exception>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr float kAlpha = 1.0f;
constexpr float kBeta = 0.0f;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

void check_cuda(cudaError_t status) {
  if (status != cudaSuccess) {
    throw std::runtime_error(cudaGetErrorString(status));
  }
}

void check_cutlass(cutlass::Status status) {
  if (status != cutlass::Status::kSuccess) {
    throw std::runtime_error(std::string("CUTLASS: ") +
                             cutlass::cutlassGetStatusString(status));
  }
}

void check_cublas(cublasStatus_t status) {
  if (status != CUBLAS_STATUS_SUCCESS) {
    throw std::runtime_error("cuBLAS call failed");
  }
}

// ---------------------------------------------------------------------------
// GPU data holder — allocates A(M×K), B(K×N), C(M×N), D(M×N).
// ---------------------------------------------------------------------------

class GemmData {
public:
  GemmData(int M, int N, int K) : M_(M), N_(N), K_(K) {
    std::mt19937 rng(2026);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    std::vector<float> hA(static_cast<size_t>(M) * K);
    std::vector<float> hB(static_cast<size_t>(K) * N);
    for (float &v : hA) v = dist(rng);
    for (float &v : hB) v = dist(rng);

    check_cuda(cudaMalloc(&d_A, hA.size() * sizeof(float)));
    check_cuda(cudaMalloc(&d_B, hB.size() * sizeof(float)));
    check_cuda(cudaMalloc(&d_C, static_cast<size_t>(M) * N * sizeof(float)));
    check_cuda(cudaMalloc(&d_D, static_cast<size_t>(M) * N * sizeof(float)));
    check_cuda(cudaMemcpy(d_A, hA.data(), hA.size() * sizeof(float),
                          cudaMemcpyHostToDevice));
    check_cuda(cudaMemcpy(d_B, hB.data(), hB.size() * sizeof(float),
                          cudaMemcpyHostToDevice));
    check_cuda(cudaMemset(d_C, 0, static_cast<size_t>(M) * N * sizeof(float)));
  }

  ~GemmData() {
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaFree(d_D);
  }

  GemmData(const GemmData &) = delete;
  GemmData &operator=(const GemmData &) = delete;

  void set_counters(benchmark::State &state) const {
    const double flops = 2.0 * M_ * N_ * K_;
    state.counters["GFLOP/s"] = benchmark::Counter(
        flops, benchmark::Counter::kIsIterationInvariantRate,
        benchmark::Counter::OneK::kIs1000);
  }

  int M_, N_, K_;
  float *d_A = nullptr;
  float *d_B = nullptr;
  float *d_C = nullptr;
  float *d_D = nullptr;
};

// ---------------------------------------------------------------------------
// Generic CUTLASS runner.  Matrix size = state.range(0) (square).
// ---------------------------------------------------------------------------

template <typename CutlassGemm>
void run_cutlass(benchmark::State &state) {
  const int N = static_cast<int>(state.range(0));
  const int M = N, K = N;
  try {
    GemmData data(M, N, K);
    CutlassGemm gemm;

    typename CutlassGemm::Arguments args({M, N, K}, {data.d_A, K},
                                         {data.d_B, N}, {data.d_C, N},
                                         {data.d_D, N}, {kAlpha, kBeta});

    if (gemm.can_implement(args) != cutlass::Status::kSuccess) {
      state.SkipWithError("can_implement failed");
      return;
    }

    const size_t ws_bytes = CutlassGemm::get_workspace_size(args);
    void *workspace = nullptr;
    if (ws_bytes > 0) check_cuda(cudaMalloc(&workspace, ws_bytes));
    check_cutlass(gemm.initialize(args, workspace));

    // Warm-up.
    check_cutlass(gemm());
    check_cuda(cudaDeviceSynchronize());

    for (auto _ : state) {
      check_cutlass(gemm());
      check_cuda(cudaDeviceSynchronize());
      benchmark::DoNotOptimize(data.d_D);
    }

    if (workspace) check_cuda(cudaFree(workspace));
    data.set_counters(state);
  } catch (const std::exception &e) {
    state.SkipWithError(e.what());
  }
}

// ---------------------------------------------------------------------------
// cuBLAS baseline runner.
// ---------------------------------------------------------------------------

void run_cublas(benchmark::State &state) {
  const int N = static_cast<int>(state.range(0));
  const int M = N, K = N;
  try {
    GemmData data(M, N, K);
    cublasHandle_t handle = nullptr;
    check_cublas(cublasCreate(&handle));

    // Warm-up.
    check_cublas(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K,
                             &kAlpha, data.d_B, N, data.d_A, K, &kBeta,
                             data.d_D, N));
    check_cuda(cudaDeviceSynchronize());

    for (auto _ : state) {
      // Row-major D = A*B  <=>  col-major D^T = B^T * A^T.
      check_cublas(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K,
                               &kAlpha, data.d_B, N, data.d_A, K, &kBeta,
                               data.d_D, N));
      check_cuda(cudaDeviceSynchronize());
      benchmark::DoNotOptimize(data.d_D);
    }

    check_cublas(cublasDestroy(handle));
    data.set_counters(state);
  } catch (const std::exception &e) {
    state.SkipWithError(e.what());
  }
}

// ---------------------------------------------------------------------------
// CUTLASS kernel type aliases
// ---------------------------------------------------------------------------

using RowMajor  = cutlass::layout::RowMajor;
using Epilogue  = cutlass::epilogue::thread::LinearCombination<float, 1, float, float>;
using Swizzle   = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;

// Shorthand template for SIMT SGEMM with RowMajor A, B, C.
template <int TbM, int TbN, int TbK, int WpM, int WpN, int WpK,
          int Stages = 2>
using SimtSgemm = cutlass::gemm::device::Gemm<
    float, RowMajor, float, RowMajor, float, RowMajor, float,
    cutlass::arch::OpClassSimt, cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<TbM, TbN, TbK>,
    cutlass::gemm::GemmShape<WpM, WpN, WpK>,
    cutlass::gemm::GemmShape<1, 1, 1>, Epilogue, Swizzle, Stages>;

// ===========================================================================
// Tile configurations to sweep.
//
//  Name         TB shape       WP shape    Warps  Thr  Stg  Grid@1024  AI
//  ----         --------       --------    -----  ---  ---  ---------  --
//  A 128x128    128×128×8      32×64×8       8    256   2      64     32.0
//  B  64x64      64× 64×8      32×32×8       4    128   2     256     16.0
//  C  64x128     64×128×8      32×64×8       4    128   2     128     21.3  ★
//  D 128x64     128× 64×8      64×32×8       4    128   2     128     21.3
//  E  32x128     32×128×8      32×32×8       4    128   2     256     12.8
//  F  64x128/S4  64×128×8      32×64×8       4    128   4     128     21.3
//  G 128x128/S4 128×128×8      32×64×8       8    256   4      64     32.0
//  H  32x64      32× 64×8      32×32×8       2     64   2     512     10.7
//  I  64x256     64×256×8      32×64×8       8    256   2      64     25.6
//  J 128x64b    128× 64×8      32×32×8       8    256   2     128     21.3
//  K  64x128w    64×128×8      64×64×8       2     64   2     128     21.3
// ===========================================================================

//                            TbM  TbN  TbK  WpM  WpN  WpK  Stages
using CfgA = SimtSgemm<       128, 128,  8,   32,  64,  8,    2>;
using CfgB = SimtSgemm<        64,  64,  8,   32,  32,  8,    2>;
using CfgC = SimtSgemm<        64, 128,  8,   32,  64,  8,    2>;
using CfgD = SimtSgemm<       128,  64,  8,   64,  32,  8,    2>;
using CfgE = SimtSgemm<        32, 128,  8,   32,  32,  8,    2>;
using CfgF = SimtSgemm<        64, 128,  8,   32,  64,  8,    4>;
using CfgG = SimtSgemm<       128, 128,  8,   32,  64,  8,    4>;
using CfgH = SimtSgemm<        32,  64,  8,   32,  32,  8,    2>;
using CfgI = SimtSgemm<        64, 256,  8,   32,  64,  8,    2>;
using CfgJ = SimtSgemm<       128,  64,  8,   32,  32,  8,    2>;
using CfgK = SimtSgemm<        64, 128,  8,   64,  64,  8,    2>;

// Default — lets CUTLASS pick its own heuristic.
using CfgDef = cutlass::gemm::device::Gemm<float, RowMajor, float, RowMajor,
                                           float, RowMajor, float>;

// ---------------------------------------------------------------------------
// Matrix sizes: 512 → 4096 (power-of-two, square).
// ---------------------------------------------------------------------------

static void Sizes(benchmark::internal::Benchmark *b) {
  for (int n : { 1024}) b->Arg(n);
}

// ---------------------------------------------------------------------------
// Benchmark registration macro.
// ---------------------------------------------------------------------------

#define REGISTER_CUTLASS_BENCH(Cfg, Label)                                     \
  void BM_##Cfg(benchmark::State &s) { run_cutlass<Cfg>(s); }                 \
  BENCHMARK(BM_##Cfg)                                                         \
      ->Name("CUTLASS/" Label)                                                \
      ->Apply(Sizes)                                                          \
      ->Unit(benchmark::kMillisecond)                                         \
      ->UseRealTime();

REGISTER_CUTLASS_BENCH(CfgDef, "Default")
REGISTER_CUTLASS_BENCH(CfgA,   "TB128x128_WP32x64_S2")
REGISTER_CUTLASS_BENCH(CfgB,   "TB64x64_WP32x32_S2")
REGISTER_CUTLASS_BENCH(CfgC,   "TB64x128_WP32x64_S2")
REGISTER_CUTLASS_BENCH(CfgD,   "TB128x64_WP64x32_S2")
REGISTER_CUTLASS_BENCH(CfgE,   "TB32x128_WP32x32_S2")
REGISTER_CUTLASS_BENCH(CfgF,   "TB64x128_WP32x64_S4")
REGISTER_CUTLASS_BENCH(CfgG,   "TB128x128_WP32x64_S4")
REGISTER_CUTLASS_BENCH(CfgH,   "TB32x64_WP32x32_S2")
REGISTER_CUTLASS_BENCH(CfgI,   "TB64x256_WP32x64_S2")
REGISTER_CUTLASS_BENCH(CfgJ,   "TB128x64_WP32x32_S2")
REGISTER_CUTLASS_BENCH(CfgK,   "TB64x128_WP64x64_S2")

#undef REGISTER_CUTLASS_BENCH

// cuBLAS baseline at each size.
BENCHMARK(run_cublas)
    ->Name("cuBLAS")
    ->Apply(Sizes)
    ->Unit(benchmark::kMillisecond)
    ->UseRealTime();

} // namespace

BENCHMARK_MAIN();