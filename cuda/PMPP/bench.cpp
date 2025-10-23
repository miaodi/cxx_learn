#include "gemm.h"
#include "vector_add.h"
#include <benchmark/benchmark.h>
#include <cmath>
#include <random>
#include <vector>
#include <cublas_v2.h>
#include <cuda_runtime.h>

// =============================================
// Vector Add Benchmarks
// Measures raw throughput (GB/s) and scaling.
// =============================================

class VectorAddBenchmark : public benchmark::Fixture {
public:
  void SetUp(const ::benchmark::State &state) override {
    size_t N = state.range(0);
    A.resize(N);
    B.resize(N);
    C.resize(N);

    // Initialize with random data
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1000.0f, 1000.0f);

    for (size_t i = 0; i < N; ++i) {
      A[i] = dist(rng);
      B[i] = dist(rng);
    }
  }

  void TearDown(const ::benchmark::State &state) override {
    // Cleanup if needed
  }

protected:
  std::vector<float> A, B, C;
};

BENCHMARK_DEFINE_F(VectorAddBenchmark, CPU)(benchmark::State &state) {
  size_t N = state.range(0);

  for (auto _ : state) {
    cpu_vector_add(A.data(), B.data(), C.data(), N);
    benchmark::DoNotOptimize(C.data());
  }

  // Throughput: read A,B write C (3 * N * sizeof(float)) per iteration.
  const double bytes_per_iter = static_cast<double>(N) * 3.0 * sizeof(float);
  state.SetBytesProcessed(
      static_cast<int64_t>(state.iterations() * bytes_per_iter));
  // Report GiB/s (binary gigabytes per second)
  state.counters["GiB/s"] = benchmark::Counter(
      (bytes_per_iter * state.iterations()) / (1024.0 * 1024.0 * 1024.0),
      benchmark::Counter::kIsRate);
}

BENCHMARK_DEFINE_F(VectorAddBenchmark, GPU)(benchmark::State &state) {
  size_t N = state.range(0);

  for (auto _ : state) {
    gpu_vector_add(A.data(), B.data(), C.data(), N);
    benchmark::DoNotOptimize(C.data());
  }

  const double bytes_per_iter = static_cast<double>(N) * 3.0 * sizeof(float);
  state.SetBytesProcessed(
      static_cast<int64_t>(state.iterations() * bytes_per_iter));
  state.counters["GiB/s"] = benchmark::Counter(
      (bytes_per_iter * state.iterations()) / (1024.0 * 1024.0 * 1024.0),
      benchmark::Counter::kIsRate);
}

// Register benchmarks for different vector sizes
BENCHMARK_REGISTER_F(VectorAddBenchmark, CPU)
    ->RangeMultiplier(4)
    ->Range(1024, 1024 * 1024 * 16) // 1K to 16M elements
    ->Unit(benchmark::kMillisecond)
    ->UseRealTime();

BENCHMARK_REGISTER_F(VectorAddBenchmark, GPU)
    ->RangeMultiplier(4)
    ->Range(1024, 1024 * 1024 * 16)
    ->Unit(benchmark::kMillisecond)
    ->UseRealTime();

// Memory bandwidth focused benchmarks
static void BM_CPU_VectorAdd_Bandwidth(benchmark::State &state) {
  size_t N = state.range(0);
  std::vector<float> A(N), B(N), C(N);

  // Initialize data
  std::fill(A.begin(), A.end(), 1.0f);
  std::fill(B.begin(), B.end(), 2.0f);

  for (auto _ : state) {
    cpu_vector_add(A.data(), B.data(), C.data(), N);
    benchmark::DoNotOptimize(C.data());
  }

  double bytes_per_iteration =
      static_cast<double>(N) * 3.0 * sizeof(float); // Read A, B; Write C
  state.SetBytesProcessed(
      static_cast<int64_t>(state.iterations() * bytes_per_iteration));
  state.counters["GiB/s"] = benchmark::Counter(
      (bytes_per_iteration * state.iterations()) / (1024.0 * 1024.0 * 1024.0),
      benchmark::Counter::kIsRate);
}

static void BM_GPU_VectorAdd_Bandwidth(benchmark::State &state) {
  size_t N = state.range(0);
  std::vector<float> A(N), B(N), C(N);

  // Initialize data
  std::fill(A.begin(), A.end(), 1.0f);
  std::fill(B.begin(), B.end(), 2.0f);

  for (auto _ : state) {
    gpu_vector_add(A.data(), B.data(), C.data(), N);
    benchmark::DoNotOptimize(C.data());
  }

  double bytes_per_iteration =
      static_cast<double>(N) * 3.0 * sizeof(float); // Read A, B; Write C
  state.SetBytesProcessed(
      static_cast<int64_t>(state.iterations() * bytes_per_iteration));
  state.counters["GiB/s"] = benchmark::Counter(
      (bytes_per_iteration * state.iterations()) / (1024.0 * 1024.0 * 1024.0),
      benchmark::Counter::kIsRate);
}

BENCHMARK(BM_CPU_VectorAdd_Bandwidth)
    ->RangeMultiplier(2)
    ->Range(1024 * 1024, 1024 * 1024 * 64)
    ->Unit(benchmark::kMillisecond)
    ->UseRealTime();

BENCHMARK(BM_GPU_VectorAdd_Bandwidth)
    ->RangeMultiplier(2)
    ->Range(1024 * 1024, 1024 * 1024 * 64)
    ->Unit(benchmark::kMillisecond)
    ->UseRealTime();

// Cache-focused benchmarks with smaller sizes
static void BM_CPU_VectorAdd_Cache(benchmark::State &state) {
  size_t N = state.range(0);
  std::vector<float> A(N, 1.0f), B(N, 2.0f), C(N);

  for (auto _ : state) {
    cpu_vector_add(A.data(), B.data(), C.data(), N);
    benchmark::DoNotOptimize(C.data());
    benchmark::ClobberMemory();
  }

  state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) * N);
}

static void BM_GPU_VectorAdd_Cache(benchmark::State &state) {
  size_t N = state.range(0);
  std::vector<float> A(N, 1.0f), B(N, 2.0f), C(N);

  for (auto _ : state) {
    gpu_vector_add(A.data(), B.data(), C.data(), N);
    benchmark::DoNotOptimize(C.data());
    benchmark::ClobberMemory();
  }

  state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) * N);
}

BENCHMARK(BM_CPU_VectorAdd_Cache)
    ->RangeMultiplier(2)
    ->Range(32, 8192)
    ->Unit(benchmark::kMicrosecond);

BENCHMARK(BM_GPU_VectorAdd_Cache)
    ->RangeMultiplier(2)
    ->Range(32, 8192)
    ->Unit(benchmark::kMicrosecond);

// =============================================
// GEMM Benchmarks
// Measures FLOP throughput (GFLOPS) across variants.
// =============================================

// GEMM Benchmarks
class CudaGEMMBenchmark : public benchmark::Fixture {
public:
  void SetUp(const ::benchmark::State &state) override {
    // For square matrices, all dimensions are the same
    if (state.range_size() == 1) {
      M = N = K = state.range(0);
    } else {
      // For non-square matrices, use separate dimensions
      M = state.range(0);
      N = state.range(1);
      K = state.range(2);
    }

    A.resize(M * K);
    B.resize(K * N);
    C.resize(M * N);

    // Initialize with random data
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    for (size_t i = 0; i < A.size(); ++i) {
      A[i] = dist(rng);
    }
    for (size_t i = 0; i < B.size(); ++i) {
      B[i] = dist(rng);
    }
  }

  void TearDown(const ::benchmark::State &state) override {
    // Cleanup if needed
  }

protected:
  std::vector<float> A, B, C;
  int M, N, K;

  // Helper function to calculate metrics
  void SetMetrics(benchmark::State &state) {
    const double flops_per_iter =
        2.0 * static_cast<double>(M) * N * K; // 2*M*N*K
    const double bytes_per_iter =
        static_cast<double>(M * K + K * N + M * N) * sizeof(float);
    // state.SetItemsProcessed(
    //     static_cast<int64_t>(state.iterations() * flops_per_iter));
    state.SetBytesProcessed(
        static_cast<int64_t>(state.iterations() * bytes_per_iter));
    state.counters["GFLOPS"] =
        benchmark::Counter((flops_per_iter * state.iterations()) / 1e9,
                           benchmark::Counter::kIsRate);
  }

  // cuBLAS helper for reference benchmarks
  void cublas_gemm(const float *A, const float *B, float *C, int M, int N, int K) {
    static cublasHandle_t handle = nullptr;
    if (handle == nullptr) {
      cublasCreate(&handle);
    }
    
    float *d_A, *d_B, *d_C;
    size_t sizeA = M * K * sizeof(float);
    size_t sizeB = K * N * sizeof(float);
    size_t sizeC = M * N * sizeof(float);
    
    cudaMalloc(&d_A, sizeA);
    cudaMalloc(&d_B, sizeB);
    cudaMalloc(&d_C, sizeC);
    
    cudaMemcpy(d_A, A, sizeA, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, sizeB, cudaMemcpyHostToDevice);
    
    const float alpha = 1.0f, beta = 0.0f;
    // C = alpha * A * B + beta * C
    // Note: cuBLAS uses column-major, but we're treating as row-major
    // so we compute C^T = B^T * A^T by swapping A,B and dimensions
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, 
                N, M, K, &alpha, d_B, N, d_A, K, &beta, d_C, N);
    
    cudaMemcpy(C, d_C, sizeC, cudaMemcpyDeviceToHost);
    
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
  }
};

BENCHMARK_DEFINE_F(CudaGEMMBenchmark, GPU_Square)(benchmark::State &state) {
  for (auto _ : state) {
    gpu_gemm(A.data(), B.data(), C.data(), M, N, K);
    benchmark::DoNotOptimize(C.data());
  }
  SetMetrics(state);
}

// Benchmark tiled GPU GEMM for square matrices with TILE=16 and TILE=32
BENCHMARK_DEFINE_F(CudaGEMMBenchmark, GPU_Tiled16)(benchmark::State &state) {
  for (auto _ : state) {
    gpu_gemm_tiled(A.data(), B.data(), C.data(), M, N, K, 16);
    benchmark::DoNotOptimize(C.data());
  }
  SetMetrics(state);
}

BENCHMARK_DEFINE_F(CudaGEMMBenchmark, GPU_Tiled32)(benchmark::State &state) {
  for (auto _ : state) {
    gpu_gemm_tiled(A.data(), B.data(), C.data(), M, N, K, 32);
    benchmark::DoNotOptimize(C.data());
  }
  SetMetrics(state);
}

// cuBLAS reference benchmark
BENCHMARK_DEFINE_F(CudaGEMMBenchmark, GPU_cuBLAS)(benchmark::State &state) {
  for (auto _ : state) {
    cublas_gemm(A.data(), B.data(), C.data(), M, N, K);
    benchmark::DoNotOptimize(C.data());
  }
  SetMetrics(state);
}

// Micro-tiled GEMM variants using the same fixture (square matrices only)
// Non-coalesced versions (showing original micro-tiling approach)
BENCHMARK_DEFINE_F(CudaGEMMBenchmark,
                   GPU_MicroTiled_32x2_NonCoalesced)(benchmark::State &state) {
  for (auto _ : state) {
    gpu_gemm_micro_tiled<32, 2, false>(A.data(), B.data(), C.data(), M, N, K);
    benchmark::DoNotOptimize(C.data());
  }
  SetMetrics(state);
}
BENCHMARK_DEFINE_F(CudaGEMMBenchmark,
                   GPU_MicroTiled_32x4_NonCoalesced)(benchmark::State &state) {
  for (auto _ : state) {
    gpu_gemm_micro_tiled<32, 4, false>(A.data(), B.data(), C.data(), M, N, K);
    benchmark::DoNotOptimize(C.data());
  }
  SetMetrics(state);
}
BENCHMARK_DEFINE_F(CudaGEMMBenchmark,
                   GPU_MicroTiled_64x2_NonCoalesced)(benchmark::State &state) {
  for (auto _ : state) {
    gpu_gemm_micro_tiled<64, 2, false>(A.data(), B.data(), C.data(), M, N, K);
    benchmark::DoNotOptimize(C.data());
  }
  SetMetrics(state);
}
BENCHMARK_DEFINE_F(CudaGEMMBenchmark,
                   GPU_MicroTiled_64x4_NonCoalesced)(benchmark::State &state) {
  for (auto _ : state) {
    gpu_gemm_micro_tiled<64, 4, false>(A.data(), B.data(), C.data(), M, N, K);
    benchmark::DoNotOptimize(C.data());
  }
  SetMetrics(state);
}

// Coalesced versions (showing memory access optimization)
BENCHMARK_DEFINE_F(CudaGEMMBenchmark,
                   GPU_MicroTiled_32x2)(benchmark::State &state) {
  for (auto _ : state) {
    gpu_gemm_micro_tiled<32, 2, true>(A.data(), B.data(), C.data(), M, N, K);
    benchmark::DoNotOptimize(C.data());
  }
  SetMetrics(state);
}
BENCHMARK_DEFINE_F(CudaGEMMBenchmark,
                   GPU_MicroTiled_32x4)(benchmark::State &state) {
  for (auto _ : state) {
    gpu_gemm_micro_tiled<32, 4, true>(A.data(), B.data(), C.data(), M, N, K);
    benchmark::DoNotOptimize(C.data());
  }
  SetMetrics(state);
}
BENCHMARK_DEFINE_F(CudaGEMMBenchmark,
                   GPU_MicroTiled_64x2)(benchmark::State &state) {
  for (auto _ : state) {
    gpu_gemm_micro_tiled<64, 2, true>(A.data(), B.data(), C.data(), M, N, K);
    benchmark::DoNotOptimize(C.data());
  }
  SetMetrics(state);
}
BENCHMARK_DEFINE_F(CudaGEMMBenchmark,
                   GPU_MicroTiled_64x4)(benchmark::State &state) {
  for (auto _ : state) {
    gpu_gemm_micro_tiled<64, 4, true>(A.data(), B.data(), C.data(), M, N, K);
    benchmark::DoNotOptimize(C.data());
  }
  SetMetrics(state);
}

// Add rectangular matrix benchmarks
BENCHMARK_DEFINE_F(CudaGEMMBenchmark, GPU_Rectangular)
(benchmark::State &state) {
  for (auto _ : state) {
    gpu_gemm(A.data(), B.data(), C.data(), M, N, K);
    benchmark::DoNotOptimize(C.data());
  }
  SetMetrics(state);
}

BENCHMARK_DEFINE_F(CudaGEMMBenchmark, GPU_Rectangular_Tiled16)
(benchmark::State &state) {
  for (auto _ : state) {
    gpu_gemm_tiled(A.data(), B.data(), C.data(), M, N, K, 16);
    benchmark::DoNotOptimize(C.data());
  }
  SetMetrics(state);
}

BENCHMARK_DEFINE_F(CudaGEMMBenchmark, GPU_Rectangular_Tiled32)
(benchmark::State &state) {
  for (auto _ : state) {
    gpu_gemm_tiled(A.data(), B.data(), C.data(), M, N, K, 32);
    benchmark::DoNotOptimize(C.data());
  }
  SetMetrics(state);
}

BENCHMARK_DEFINE_F(CudaGEMMBenchmark, GPU_Rectangular_cuBLAS)
(benchmark::State &state) {
  for (auto _ : state) {
    cublas_gemm(A.data(), B.data(), C.data(), M, N, K);
    benchmark::DoNotOptimize(C.data());
  }
  SetMetrics(state);
}

// Rectangular matrix micro-tiled benchmarks
// Non-coalesced versions first
BENCHMARK_DEFINE_F(CudaGEMMBenchmark, GPU_Rectangular_MicroTiled_32x2_NonCoalesced)
(benchmark::State &state) {
  for (auto _ : state) {
    gpu_gemm_micro_tiled<32, 2, false>(A.data(), B.data(), C.data(), M, N, K);
    benchmark::DoNotOptimize(C.data());
  }
  SetMetrics(state);
}

BENCHMARK_DEFINE_F(CudaGEMMBenchmark, GPU_Rectangular_MicroTiled_32x4_NonCoalesced)
(benchmark::State &state) {
  for (auto _ : state) {
    gpu_gemm_micro_tiled<32, 4, false>(A.data(), B.data(), C.data(), M, N, K);
    benchmark::DoNotOptimize(C.data());
  }
  SetMetrics(state);
}

BENCHMARK_DEFINE_F(CudaGEMMBenchmark, GPU_Rectangular_MicroTiled_64x2_NonCoalesced)
(benchmark::State &state) {
  for (auto _ : state) {
    gpu_gemm_micro_tiled<64, 2, false>(A.data(), B.data(), C.data(), M, N, K);
    benchmark::DoNotOptimize(C.data());
  }
  SetMetrics(state);
}

BENCHMARK_DEFINE_F(CudaGEMMBenchmark, GPU_Rectangular_MicroTiled_64x4_NonCoalesced)
(benchmark::State &state) {
  for (auto _ : state) {
    gpu_gemm_micro_tiled<64, 4, false>(A.data(), B.data(), C.data(), M, N, K);
    benchmark::DoNotOptimize(C.data());
  }
  SetMetrics(state);
}

// Coalesced versions
BENCHMARK_DEFINE_F(CudaGEMMBenchmark, GPU_Rectangular_MicroTiled_32x2)
(benchmark::State &state) {
  for (auto _ : state) {
    gpu_gemm_micro_tiled<32, 2, true>(A.data(), B.data(), C.data(), M, N, K);
    benchmark::DoNotOptimize(C.data());
  }
  SetMetrics(state);
}

BENCHMARK_DEFINE_F(CudaGEMMBenchmark, GPU_Rectangular_MicroTiled_32x4)
(benchmark::State &state) {
  for (auto _ : state) {
    gpu_gemm_micro_tiled<32, 4, true>(A.data(), B.data(), C.data(), M, N, K);
    benchmark::DoNotOptimize(C.data());
  }
  SetMetrics(state);
}

BENCHMARK_DEFINE_F(CudaGEMMBenchmark, GPU_Rectangular_MicroTiled_64x2)
(benchmark::State &state) {
  for (auto _ : state) {
    gpu_gemm_micro_tiled<64, 2, true>(A.data(), B.data(), C.data(), M, N, K);
    benchmark::DoNotOptimize(C.data());
  }
  SetMetrics(state);
}

BENCHMARK_DEFINE_F(CudaGEMMBenchmark, GPU_Rectangular_MicroTiled_64x4)
(benchmark::State &state) {
  for (auto _ : state) {
    gpu_gemm_micro_tiled<64, 4, true>(A.data(), B.data(), C.data(), M, N, K);
    benchmark::DoNotOptimize(C.data());
  }
  SetMetrics(state);
}

// Register square matrix benchmarks in order of optimization complexity
// 1. Reference implementation (cuBLAS - optimized baseline)
BENCHMARK_REGISTER_F(CudaGEMMBenchmark, GPU_cuBLAS)
    ->Name("GEMM_Square_cuBLAS")
    ->RangeMultiplier(2)
    ->Range(16, 16384)
    ->Unit(benchmark::kMillisecond)
    ->UseRealTime();

// 2. Naive implementation (basic GPU parallelization)
BENCHMARK_REGISTER_F(CudaGEMMBenchmark, GPU_Square)
    ->Name("GEMM_Square_Naive")
    ->RangeMultiplier(2)
    ->Range(16, 16384)
    ->Unit(benchmark::kMillisecond)
    ->UseRealTime();

// 3. Tiled implementations (shared memory optimization)
BENCHMARK_REGISTER_F(CudaGEMMBenchmark, GPU_Tiled16)
    ->Name("GEMM_Square_Tiled16")
    ->RangeMultiplier(2)
    ->Range(16, 16384)
    ->Unit(benchmark::kMillisecond)
    ->UseRealTime();

BENCHMARK_REGISTER_F(CudaGEMMBenchmark, GPU_Tiled32)
    ->Name("GEMM_Square_Tiled32")
    ->RangeMultiplier(2)
    ->Range(16, 16384)
    ->Unit(benchmark::kMillisecond)
    ->UseRealTime();

// 4. Micro-tiled implementations (advanced thread-level optimization)
// 4a. Non-coalesced micro-tiled (original micro-tiling approach)
BENCHMARK_REGISTER_F(CudaGEMMBenchmark, GPU_MicroTiled_32x2_NonCoalesced)
    ->Name("GEMM_Square_MicroTiled_32x2_NonCoalesced")
    ->RangeMultiplier(2)
    ->Range(32, 16384)
    ->Unit(benchmark::kMillisecond)
    ->UseRealTime();

BENCHMARK_REGISTER_F(CudaGEMMBenchmark, GPU_MicroTiled_32x4_NonCoalesced)
    ->Name("GEMM_Square_MicroTiled_32x4_NonCoalesced")
    ->RangeMultiplier(2)
    ->Range(64, 16384)
    ->Unit(benchmark::kMillisecond)
    ->UseRealTime();

BENCHMARK_REGISTER_F(CudaGEMMBenchmark, GPU_MicroTiled_64x2_NonCoalesced)
    ->Name("GEMM_Square_MicroTiled_64x2_NonCoalesced")
    ->RangeMultiplier(2)
    ->Range(64, 16384)
    ->Unit(benchmark::kMillisecond)
    ->UseRealTime();

BENCHMARK_REGISTER_F(CudaGEMMBenchmark, GPU_MicroTiled_64x4_NonCoalesced)
    ->Name("GEMM_Square_MicroTiled_64x4_NonCoalesced")
    ->RangeMultiplier(2)
    ->Range(64, 16384)
    ->Unit(benchmark::kMillisecond)
    ->UseRealTime();

// 4b. Coalesced micro-tiled (memory access optimization)
BENCHMARK_REGISTER_F(CudaGEMMBenchmark, GPU_MicroTiled_32x2)
    ->Name("GEMM_Square_MicroTiled_32x2_Coalesced")
    ->RangeMultiplier(2)
    ->Range(32, 16384)
    ->Unit(benchmark::kMillisecond)
    ->UseRealTime();

BENCHMARK_REGISTER_F(CudaGEMMBenchmark, GPU_MicroTiled_32x4)
    ->Name("GEMM_Square_MicroTiled_32x4_Coalesced")
    ->RangeMultiplier(2)
    ->Range(64, 16384)
    ->Unit(benchmark::kMillisecond)
    ->UseRealTime();

BENCHMARK_REGISTER_F(CudaGEMMBenchmark, GPU_MicroTiled_64x2)
    ->Name("GEMM_Square_MicroTiled_64x2_Coalesced")
    ->RangeMultiplier(2)
    ->Range(64, 16384)
    ->Unit(benchmark::kMillisecond)
    ->UseRealTime();

BENCHMARK_REGISTER_F(CudaGEMMBenchmark, GPU_MicroTiled_64x4)
    ->Name("GEMM_Square_MicroTiled_64x4_Coalesced")
    ->RangeMultiplier(2)
    ->Range(64, 16384)
    ->Unit(benchmark::kMillisecond)
    ->UseRealTime();

// Representative rectangular cases (tall, wide, deep)
// cuBLAS reference for rectangular matrices
BENCHMARK_REGISTER_F(CudaGEMMBenchmark, GPU_Rectangular_cuBLAS)
    ->Name("GEMM_Rectangular_cuBLAS")
    ->Args({512, 64, 128})
    ->Args({64, 512, 128})
    ->Args({128, 128, 512})
    ->Unit(benchmark::kMillisecond)
    ->UseRealTime();

// Basic rectangular implementation
BENCHMARK_REGISTER_F(CudaGEMMBenchmark, GPU_Rectangular)
    ->Name("GEMM_Rectangular_Naive")
    ->Args({512, 64, 128})
    ->Args({64, 512, 128})
    ->Args({128, 128, 512})
    ->Unit(benchmark::kMillisecond)
    ->UseRealTime();

// Tiled versions for rectangular matrices
BENCHMARK_REGISTER_F(CudaGEMMBenchmark, GPU_Rectangular_Tiled16)
    ->Name("GEMM_Rectangular_Tiled16")
    ->Args({512, 64, 128})
    ->Args({64, 512, 128})
    ->Args({128, 128, 512})
    ->Unit(benchmark::kMillisecond)
    ->UseRealTime();

BENCHMARK_REGISTER_F(CudaGEMMBenchmark, GPU_Rectangular_Tiled32)
    ->Name("GEMM_Rectangular_Tiled32")
    ->Args({512, 64, 128})
    ->Args({64, 512, 128})
    ->Args({128, 128, 512})
    ->Unit(benchmark::kMillisecond)
    ->UseRealTime();

// Micro-tiled versions for rectangular matrices
// Non-coalesced variants first
BENCHMARK_REGISTER_F(CudaGEMMBenchmark, GPU_Rectangular_MicroTiled_32x2_NonCoalesced)
    ->Name("GEMM_Rectangular_MicroTiled_32x2_NonCoalesced")
    ->Args({512, 64, 128})
    ->Args({64, 512, 128})
    ->Args({128, 128, 512})
    ->Unit(benchmark::kMillisecond)
    ->UseRealTime();

BENCHMARK_REGISTER_F(CudaGEMMBenchmark, GPU_Rectangular_MicroTiled_32x4_NonCoalesced)
    ->Name("GEMM_Rectangular_MicroTiled_32x4_NonCoalesced")
    ->Args({512, 64, 128})
    ->Args({64, 512, 128})
    ->Args({128, 128, 512})
    ->Unit(benchmark::kMillisecond)
    ->UseRealTime();

BENCHMARK_REGISTER_F(CudaGEMMBenchmark, GPU_Rectangular_MicroTiled_64x2_NonCoalesced)
    ->Name("GEMM_Rectangular_MicroTiled_64x2_NonCoalesced")
    ->Args({512, 64, 128})
    ->Args({64, 512, 128})
    ->Args({128, 128, 512})
    ->Unit(benchmark::kMillisecond)
    ->UseRealTime();

BENCHMARK_REGISTER_F(CudaGEMMBenchmark, GPU_Rectangular_MicroTiled_64x4_NonCoalesced)
    ->Name("GEMM_Rectangular_MicroTiled_64x4_NonCoalesced")
    ->Args({512, 64, 128})
    ->Args({64, 512, 128})
    ->Args({128, 128, 512})
    ->Unit(benchmark::kMillisecond)
    ->UseRealTime();

// Coalesced variants
BENCHMARK_REGISTER_F(CudaGEMMBenchmark, GPU_Rectangular_MicroTiled_32x2)
    ->Name("GEMM_Rectangular_MicroTiled_32x2_Coalesced")
    ->Args({512, 64, 128})
    ->Args({64, 512, 128})
    ->Args({128, 128, 512})
    ->Unit(benchmark::kMillisecond)
    ->UseRealTime();

BENCHMARK_REGISTER_F(CudaGEMMBenchmark, GPU_Rectangular_MicroTiled_32x4)
    ->Name("GEMM_Rectangular_MicroTiled_32x4_Coalesced")
    ->Args({512, 64, 128})
    ->Args({64, 512, 128})
    ->Args({128, 128, 512})
    ->Unit(benchmark::kMillisecond)
    ->UseRealTime();

BENCHMARK_REGISTER_F(CudaGEMMBenchmark, GPU_Rectangular_MicroTiled_64x2)
    ->Name("GEMM_Rectangular_MicroTiled_64x2_Coalesced")
    ->Args({512, 64, 128})
    ->Args({64, 512, 128})
    ->Args({128, 128, 512})
    ->Unit(benchmark::kMillisecond)
    ->UseRealTime();

BENCHMARK_REGISTER_F(CudaGEMMBenchmark, GPU_Rectangular_MicroTiled_64x4)
    ->Name("GEMM_Rectangular_MicroTiled_64x4_Coalesced")
    ->Args({512, 64, 128})
    ->Args({64, 512, 128})
    ->Args({128, 128, 512})
    ->Unit(benchmark::kMillisecond)
    ->UseRealTime();

BENCHMARK_MAIN();