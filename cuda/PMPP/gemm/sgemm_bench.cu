#include "sgemm.h"

#include <benchmark/benchmark.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <exception>
#include <random>
#include <stdexcept>
#include <vector>

namespace {

constexpr int kM = 1024;
constexpr int kN = 1024;
constexpr int kK = 1024;
constexpr float kAlpha = 1.0f;
constexpr float kBeta = 0.0f;

void check_cuda(cudaError_t status) {
  if (status != cudaSuccess) {
    throw std::runtime_error(cudaGetErrorString(status));
  }
}

void check_cublas(cublasStatus_t status) {
  if (status != CUBLAS_STATUS_SUCCESS) {
    throw std::runtime_error("cuBLAS call failed");
  }
}

std::vector<float> random_values(int count) {
  std::vector<float> values(count);
  std::mt19937 rng(2026);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  for (float &value : values) {
    value = dist(rng);
  }
  return values;
}

bool ensure_cuda_runtime_available(benchmark::State &state) {
  const cudaError_t status = cudaFree(nullptr);
  if (status == cudaSuccess) {
    return true;
  }
  state.SkipWithError(cudaGetErrorString(status));
  return false;
}

class SgemmBenchmarkData {
public:
  SgemmBenchmarkData() {
    const std::vector<float> A = random_values(kM * kK);
    const std::vector<float> B = random_values(kK * kN);
    const std::vector<float> C(kM * kN, 0.0f);

    check_cuda(cudaMalloc(&d_A_, A.size() * sizeof(float)));
    check_cuda(cudaMalloc(&d_B_, B.size() * sizeof(float)));
    check_cuda(cudaMalloc(&d_C_, C.size() * sizeof(float)));
    check_cuda(cudaMemcpy(d_A_, A.data(), A.size() * sizeof(float),
                          cudaMemcpyHostToDevice));
    check_cuda(cudaMemcpy(d_B_, B.data(), B.size() * sizeof(float),
                          cudaMemcpyHostToDevice));
    check_cuda(cudaMemcpy(d_C_, C.data(), C.size() * sizeof(float),
                          cudaMemcpyHostToDevice));

    check_cublas(cublasCreate(&handle_));
  }

  ~SgemmBenchmarkData() {
    check_cublas(cublasDestroy(handle_));
    check_cuda(cudaFree(d_A_));
    check_cuda(cudaFree(d_B_));
    check_cuda(cudaFree(d_C_));
  }

  SgemmBenchmarkData(const SgemmBenchmarkData &) = delete;
  SgemmBenchmarkData &operator=(const SgemmBenchmarkData &) = delete;

  void set_metrics(benchmark::State &state) {
    const double flops = 2.0 * kM * kN * kK;
    state.counters["GFLOP/s"] =
        benchmark::Counter(flops, benchmark::Counter::kIsIterationInvariantRate,
                           benchmark::Counter::OneK::kIs1000);
  }

  float *d_A_ = nullptr;
  float *d_B_ = nullptr;
  float *d_C_ = nullptr;
  cublasHandle_t handle_ = nullptr;
};

void BM_NaiveIjk(benchmark::State &state) {
  if (!ensure_cuda_runtime_available(state)) {
    return;
  }
  try {
    SgemmBenchmarkData data;
    for (auto _ : state) {
      check_cuda(pmpp::gemm::sgemm_ijk(kM, kN, kK, kAlpha, data.d_A_, kK,
                                       data.d_B_, kN, kBeta, data.d_C_, kN));
      check_cuda(cudaDeviceSynchronize());
      benchmark::DoNotOptimize(data.d_C_);
    }
    data.set_metrics(state);
  } catch (const std::exception &error) {
    state.SkipWithError(error.what());
  }
}

void BM_CuBLAS(benchmark::State &state) {
  if (!ensure_cuda_runtime_available(state)) {
    return;
  }
  try {
    SgemmBenchmarkData data;
    for (auto _ : state) {
      // Row-major C = A * B is equivalent to column-major C^T = B^T * A^T.
      check_cublas(cublasSgemm(data.handle_, CUBLAS_OP_N, CUBLAS_OP_N, kN, kM,
                               kK, &kAlpha, data.d_B_, kN, data.d_A_, kK,
                               &kBeta, data.d_C_, kN));
      check_cuda(cudaDeviceSynchronize());
      benchmark::DoNotOptimize(data.d_C_);
    }
    data.set_metrics(state);
  } catch (const std::exception &error) {
    state.SkipWithError(error.what());
  }
}

void BM_Tiled16(benchmark::State &state) {
  if (!ensure_cuda_runtime_available(state)) {
    return;
  }
  try {
    SgemmBenchmarkData data;
    for (auto _ : state) {
      check_cuda(pmpp::gemm::sgemm_tiled_16(kM, kN, kK, kAlpha, data.d_A_, kK,
                                            data.d_B_, kN, kBeta, data.d_C_,
                                            kN));
      check_cuda(cudaDeviceSynchronize());
      benchmark::DoNotOptimize(data.d_C_);
    }
    data.set_metrics(state);
  } catch (const std::exception &error) {
    state.SkipWithError(error.what());
  }
}

void BM_Tiled16_2x2(benchmark::State &state) {
  if (!ensure_cuda_runtime_available(state)) {
    return;
  }
  try {
    SgemmBenchmarkData data;
    for (auto _ : state) {
      check_cuda(pmpp::gemm::sgemm_tiled_16_2x2(
          kM, kN, kK, kAlpha, data.d_A_, kK, data.d_B_, kN, kBeta, data.d_C_,
          kN));
      check_cuda(cudaDeviceSynchronize());
      benchmark::DoNotOptimize(data.d_C_);
    }
    data.set_metrics(state);
  } catch (const std::exception &error) {
    state.SkipWithError(error.what());
  }
}

void BM_Tiled16_4x4(benchmark::State &state) {
  if (!ensure_cuda_runtime_available(state)) {
    return;
  }
  try {
    SgemmBenchmarkData data;
    for (auto _ : state) {
      check_cuda(pmpp::gemm::sgemm_tiled_16_4x4(
          kM, kN, kK, kAlpha, data.d_A_, kK, data.d_B_, kN, kBeta, data.d_C_,
          kN));
      check_cuda(cudaDeviceSynchronize());
      benchmark::DoNotOptimize(data.d_C_);
    }
    data.set_metrics(state);
  } catch (const std::exception &error) {
    state.SkipWithError(error.what());
  }
}

void BM_Tiled16_8x8(benchmark::State &state) {
  if (!ensure_cuda_runtime_available(state)) {
    return;
  }
  try {
    SgemmBenchmarkData data;
    for (auto _ : state) {
      check_cuda(pmpp::gemm::sgemm_tiled_16_8x8(
          kM, kN, kK, kAlpha, data.d_A_, kK, data.d_B_, kN, kBeta, data.d_C_,
          kN));
      check_cuda(cudaDeviceSynchronize());
      benchmark::DoNotOptimize(data.d_C_);
    }
    data.set_metrics(state);
  } catch (const std::exception &error) {
    state.SkipWithError(error.what());
  }
}

using SgemmFunction = cudaError_t (*)(int, int, int, float, const float *, int,
                                      const float *, int, float, float *, int,
                                      cudaStream_t);

void BM_SgemmFunction(benchmark::State &state, SgemmFunction function) {
  if (!ensure_cuda_runtime_available(state)) {
    return;
  }
  try {
    SgemmBenchmarkData data;
    for (auto _ : state) {
      check_cuda(function(kM, kN, kK, kAlpha, data.d_A_, kK, data.d_B_, kN,
                          kBeta, data.d_C_, kN, nullptr));
      check_cuda(cudaDeviceSynchronize());
      benchmark::DoNotOptimize(data.d_C_);
    }
    data.set_metrics(state);
  } catch (const std::exception &error) {
    state.SkipWithError(error.what());
  }
}

void BM_Tiled16_2x2K32(benchmark::State &state) {
  BM_SgemmFunction(state, pmpp::gemm::sgemm_tiled_16_2x2_k32);
}

void BM_Tiled16_2x2K64(benchmark::State &state) {
  BM_SgemmFunction(state, pmpp::gemm::sgemm_tiled_16_2x2_k64);
}

void BM_Tiled16_4x4K32(benchmark::State &state) {
  BM_SgemmFunction(state, pmpp::gemm::sgemm_tiled_16_4x4_k32);
}

void BM_Tiled16_4x4K64(benchmark::State &state) {
  BM_SgemmFunction(state, pmpp::gemm::sgemm_tiled_16_4x4_k64);
}

void BM_Tiled16_8x8K32(benchmark::State &state) {
  BM_SgemmFunction(state, pmpp::gemm::sgemm_tiled_16_8x8_k32);
}

void BM_Tiled16_8x8K64(benchmark::State &state) {
  BM_SgemmFunction(state, pmpp::gemm::sgemm_tiled_16_8x8_k64);
}

BENCHMARK(BM_NaiveIjk)
    ->Name("SGEMM/NaiveIjk/1024")
    ->Unit(benchmark::kMillisecond)
    ->UseRealTime();

BENCHMARK(BM_Tiled16)
    ->Name("SGEMM/Tiled16/1024")
    ->Unit(benchmark::kMillisecond)
    ->UseRealTime();

BENCHMARK(BM_Tiled16_2x2)
    ->Name("SGEMM/Tiled16_2x2/1024")
    ->Unit(benchmark::kMillisecond)
    ->UseRealTime();

BENCHMARK(BM_Tiled16_4x4)
    ->Name("SGEMM/Tiled16_4x4/1024")
    ->Unit(benchmark::kMillisecond)
    ->UseRealTime();

BENCHMARK(BM_Tiled16_8x8)
    ->Name("SGEMM/Tiled16_8x8/1024")
    ->Unit(benchmark::kMillisecond)
    ->UseRealTime();

BENCHMARK(BM_Tiled16_2x2K32)
    ->Name("SGEMM/Tiled16_2x2K32/1024")
    ->Unit(benchmark::kMillisecond)
    ->UseRealTime();

BENCHMARK(BM_Tiled16_2x2K64)
    ->Name("SGEMM/Tiled16_2x2K64/1024")
    ->Unit(benchmark::kMillisecond)
    ->UseRealTime();

BENCHMARK(BM_Tiled16_4x4K32)
    ->Name("SGEMM/Tiled16_4x4K32/1024")
    ->Unit(benchmark::kMillisecond)
    ->UseRealTime();

BENCHMARK(BM_Tiled16_4x4K64)
    ->Name("SGEMM/Tiled16_4x4K64/1024")
    ->Unit(benchmark::kMillisecond)
    ->UseRealTime();

BENCHMARK(BM_Tiled16_8x8K32)
    ->Name("SGEMM/Tiled16_8x8K32/1024")
    ->Unit(benchmark::kMillisecond)
    ->UseRealTime();

BENCHMARK(BM_Tiled16_8x8K64)
    ->Name("SGEMM/Tiled16_8x8K64/1024")
    ->Unit(benchmark::kMillisecond)
    ->UseRealTime();

BENCHMARK(BM_CuBLAS)
    ->Name("SGEMM/cuBLAS/1024")
    ->Unit(benchmark::kMillisecond)
    ->UseRealTime();

} // namespace

BENCHMARK_MAIN();
