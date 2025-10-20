#include "gemm.h"
#include "vector_add.h"
#include <benchmark/benchmark.h>
#include <random>
#include <vector>

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

  // Calculate throughput
  state.SetBytesProcessed(static_cast<int64_t>(state.iterations()) * N * 3 *
                          sizeof(float));
  state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) * N);
}

BENCHMARK_DEFINE_F(VectorAddBenchmark, GPU)(benchmark::State &state) {
  size_t N = state.range(0);

  for (auto _ : state) {
    gpu_vector_add(A.data(), B.data(), C.data(), N);
    benchmark::DoNotOptimize(C.data());
  }

  // Calculate throughput
  state.SetBytesProcessed(static_cast<int64_t>(state.iterations()) * N * 3 *
                          sizeof(float));
  state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) * N);
}

// Register benchmarks for different vector sizes
BENCHMARK_REGISTER_F(VectorAddBenchmark, CPU)
    ->Range(1024, 1024 * 1024 * 16) // 1K to 16M elements
    ->Unit(benchmark::kMicrosecond)
    ->UseRealTime();

BENCHMARK_REGISTER_F(VectorAddBenchmark, GPU)
    ->Range(1024, 1024 * 1024 * 16) // 1K to 16M elements
    ->Unit(benchmark::kMicrosecond)
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

  double bytes_per_iteration = N * 3 * sizeof(float); // Read A, B; Write C
  state.SetBytesProcessed(
      static_cast<int64_t>(state.iterations() * bytes_per_iteration));
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

  double bytes_per_iteration = N * 3 * sizeof(float); // Read A, B; Write C
  state.SetBytesProcessed(
      static_cast<int64_t>(state.iterations() * bytes_per_iteration));
}

BENCHMARK(BM_CPU_VectorAdd_Bandwidth)
    ->Range(1024 * 1024, 1024 * 1024 * 64) // 1M to 64M elements
    ->Unit(benchmark::kMillisecond)
    ->UseRealTime();

BENCHMARK(BM_GPU_VectorAdd_Bandwidth)
    ->Range(1024 * 1024, 1024 * 1024 * 64) // 1M to 64M elements
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
    ->Range(8, 8192) // 8 to 8K elements (fits in various cache levels)
    ->Unit(benchmark::kNanosecond);

BENCHMARK(BM_GPU_VectorAdd_Cache)
    ->Range(8, 8192) // 8 to 8K elements
    ->Unit(benchmark::kNanosecond);

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
    int64_t flops = static_cast<int64_t>(2) * M * N * K;
    state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) * flops);

    int64_t bytes = static_cast<int64_t>(M * K + K * N + M * N) * sizeof(float);
    state.SetBytesProcessed(static_cast<int64_t>(state.iterations()) * bytes);
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

// Register square matrix benchmarks
BENCHMARK_REGISTER_F(CudaGEMMBenchmark, GPU_Square)
    ->Range(16, 512) // 16x16 to 512x512 matrices
    ->Unit(benchmark::kMillisecond)
    ->UseRealTime();

BENCHMARK_REGISTER_F(CudaGEMMBenchmark, GPU_Tiled16)
    ->Range(16, 512)
    ->Unit(benchmark::kMillisecond)
    ->UseRealTime();

BENCHMARK_REGISTER_F(CudaGEMMBenchmark, GPU_Tiled32)
    ->Range(16, 512)
    ->Unit(benchmark::kMillisecond)
    ->UseRealTime();

// Register rectangular matrix benchmarks with various aspect ratios
// Tall-skinny matrices (M >> N)
BENCHMARK_REGISTER_F(CudaGEMMBenchmark, GPU_Rectangular)
    ->Args({512, 64, 128})   // Tall matrix: 512x64 * 128x64
    ->Args({1024, 64, 256})  // Very tall: 1024x64 * 256x64
    ->Args({2048, 128, 512}) // Extra tall: 2048x128 * 512x128
    ->Unit(benchmark::kMillisecond)
    ->UseRealTime();

// Short-wide matrices (N >> M)
BENCHMARK_REGISTER_F(CudaGEMMBenchmark, GPU_Rectangular)
    ->Args({64, 512, 128})   // Wide matrix: 64x128 * 128x512
    ->Args({64, 1024, 256})  // Very wide: 64x256 * 256x1024
    ->Args({128, 2048, 512}) // Extra wide: 128x512 * 512x2048
    ->Unit(benchmark::kMillisecond)
    ->UseRealTime();

// Deep multiplication (K >> M, N)
BENCHMARK_REGISTER_F(CudaGEMMBenchmark, GPU_Rectangular)
    ->Args({64, 64, 512})    // Deep: 64x512 * 512x64
    ->Args({128, 128, 1024}) // Very deep: 128x1024 * 1024x128
    ->Args({256, 256, 2048}) // Extra deep: 256x2048 * 2048x256
    ->Unit(benchmark::kMillisecond)
    ->UseRealTime();

// Mixed aspect ratios
BENCHMARK_REGISTER_F(CudaGEMMBenchmark, GPU_Rectangular)
    ->Args({256, 512, 128})  // Mixed: 256x128 * 128x512
    ->Args({512, 256, 1024}) // Mixed: 512x1024 * 1024x256
    ->Args({128, 1024, 256}) // Mixed: 128x256 * 256x1024
    ->Unit(benchmark::kMillisecond)
    ->UseRealTime();

// Tiled versions for rectangular matrices
BENCHMARK_REGISTER_F(CudaGEMMBenchmark, GPU_Rectangular_Tiled16)
    ->Args({512, 64, 128})   // Tall with 16x16 tiles
    ->Args({64, 512, 128})   // Wide with 16x16 tiles
    ->Args({256, 256, 1024}) // Deep with 16x16 tiles
    ->Unit(benchmark::kMillisecond)
    ->UseRealTime();

BENCHMARK_REGISTER_F(CudaGEMMBenchmark, GPU_Rectangular_Tiled32)
    ->Args({512, 64, 128})   // Tall with 32x32 tiles
    ->Args({64, 512, 128})   // Wide with 32x32 tiles
    ->Args({256, 256, 1024}) // Deep with 32x32 tiles
    ->Unit(benchmark::kMillisecond)
    ->UseRealTime();

// Thread boundary tests for rectangular matrices
BENCHMARK_REGISTER_F(CudaGEMMBenchmark, GPU_Rectangular)
    ->Args({32, 64, 32}) // Small rectangular around warp boundaries
    ->Args({64, 32, 64})
    ->Args({128, 64, 128}) // Medium rectangular
    ->Args({64, 128, 64})
    ->Args({256, 128, 256}) // Large rectangular
    ->Args({128, 256, 128})
    ->Unit(benchmark::kMicrosecond);

// Batch-like operations (common in ML workloads)
BENCHMARK_REGISTER_F(CudaGEMMBenchmark, GPU_Rectangular)
    ->Args({784, 128, 10})    // MNIST-like: image features to hidden layer
    ->Args({128, 64, 784})    // Hidden layer processing
    ->Args({1000, 512, 2048}) // ImageNet-like classification
    ->Args({512, 1000, 2048}) // Reverse classification
    ->Unit(benchmark::kMillisecond)
    ->UseRealTime();

// Large matrix tests (assuming 10GB memory limit)
// Memory usage: A(M*K) + B(K*N) + C(M*N) floats * 4 bytes = total bytes
// Target ~8GB usage to leave room for GPU overhead

// Large square matrices (~2.7GB each for A, B, C = ~8GB total)
BENCHMARK_REGISTER_F(CudaGEMMBenchmark, GPU_Square)
    ->Name("CudaGEMMBenchmark/GPU_Square_Large")
    ->Arg(13000) // 13k x 13k = ~2.7GB per matrix
    ->Arg(15000) // 15k x 15k = ~3.6GB per matrix
    ->Arg(16384) // 16k x 16k = ~4.3GB per matrix (power of 2)
    ->Unit(benchmark::kMillisecond)
    ->UseRealTime()
    ->Iterations(3); // Fewer iterations for large matrices

// Large rectangular - memory bound scenarios
BENCHMARK_REGISTER_F(CudaGEMMBenchmark, GPU_Rectangular)
    ->Name("CudaGEMMBenchmark/GPU_Rectangular_Large")
    ->Args({32768, 8192, 4096})  // Tall: 32k x 8k x 4k (~2.1GB total)
    ->Args({8192, 32768, 4096})  // Wide: 8k x 32k x 4k (~2.1GB total)
    ->Args({16384, 16384, 8192}) // Deep: 16k x 16k x 8k (~4.2GB total)
    ->Args({20000, 12000, 8000}) // Mixed large: 20k x 12k x 8k (~3.7GB total)
    ->Unit(benchmark::kMillisecond)
    ->UseRealTime()
    ->Iterations(3);

// Large tiled performance comparison
BENCHMARK_REGISTER_F(CudaGEMMBenchmark, GPU_Tiled16)
    ->Name("CudaGEMMBenchmark/GPU_Tiled16_Large")
    ->Arg(12000) // 12k x 12k with 16x16 tiles
    ->Arg(14000) // 14k x 14k with 16x16 tiles
    ->Unit(benchmark::kMillisecond)
    ->UseRealTime()
    ->Iterations(3);

BENCHMARK_REGISTER_F(CudaGEMMBenchmark, GPU_Tiled32)
    ->Name("CudaGEMMBenchmark/GPU_Tiled32_Large")
    ->Arg(12000) // 12k x 12k with 32x32 tiles
    ->Arg(14000) // 14k x 14k with 32x32 tiles
    ->Unit(benchmark::kMillisecond)
    ->UseRealTime()
    ->Iterations(3);

// Large rectangular tiled tests
BENCHMARK_REGISTER_F(CudaGEMMBenchmark, GPU_Rectangular_Tiled16)
    ->Name("CudaGEMMBenchmark/GPU_Rectangular_Tiled16_Large")
    ->Args({24576, 6144, 4096}) // Large tall with 16x16 tiles (~2.4GB)
    ->Args({6144, 24576, 4096}) // Large wide with 16x16 tiles (~2.4GB)
    ->Unit(benchmark::kMillisecond)
    ->UseRealTime()
    ->Iterations(3);

BENCHMARK_REGISTER_F(CudaGEMMBenchmark, GPU_Rectangular_Tiled32)
    ->Name("CudaGEMMBenchmark/GPU_Rectangular_Tiled32_Large")
    ->Args({24576, 6144, 4096}) // Large tall with 32x32 tiles (~2.4GB)
    ->Args({6144, 24576, 4096}) // Large wide with 32x32 tiles (~2.4GB)
    ->Unit(benchmark::kMillisecond)
    ->UseRealTime()
    ->Iterations(3);

// Memory-intensive scenarios for stress testing
BENCHMARK_REGISTER_F(CudaGEMMBenchmark, GPU_Rectangular)
    ->Name("CudaGEMMBenchmark/GPU_Rectangular_Stress")
    ->Args({65536, 2048, 1024}) // Very tall: 65k x 2k x 1k (~0.5GB)
    ->Args({2048, 65536, 1024}) // Very wide: 2k x 65k x 1k (~0.5GB)
    ->Args({8192, 8192, 16384}) // Very deep: 8k x 8k x 16k (~4.2GB)
    ->Args({32768, 4096, 2048}) // Extreme aspect: 32k x 4k x 2k (~1.1GB)
    ->Unit(benchmark::kMillisecond)
    ->UseRealTime()
    ->Iterations(2); // Very few iterations for stress tests

BENCHMARK_MAIN();