#include "convolution.h"
#include "vector_add.h"
#include <benchmark/benchmark.h>
#include <numeric>
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

  void TearDown([[maybe_unused]] const ::benchmark::State &state) override {
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

// ============================================================================
// Convolution Benchmarks - CPU vs GPU (Global Memory) vs GPU (Constant Memory)
// ============================================================================

class ConvolutionBenchmark : public benchmark::Fixture {
public:
  void SetUp(const ::benchmark::State &state) override {
    width = state.range(0);
    height = state.range(1);
    kernel_size = state.range(2);

    input.resize(width * height);
    output.resize(width * height);
    kernel.resize(kernel_size * kernel_size);

    // Initialize with random data
    std::mt19937 rng(12345);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    for (size_t i = 0; i < input.size(); ++i) {
      input[i] = dist(rng);
    }

    // Create a simple Gaussian-like kernel
    float sum = 0.0f;
    for (int i = 0; i < kernel_size; ++i) {
      for (int j = 0; j < kernel_size; ++j) {
        float val = 1.0f;
        kernel[i * kernel_size + j] = val;
        sum += val;
      }
    }
    // Normalize kernel
    for (auto &k : kernel) {
      k /= sum;
    }
  }

  void TearDown([[maybe_unused]] const ::benchmark::State &state) override {}

protected:
  int width, height, kernel_size;
  std::vector<float> input, output, kernel;
};

BENCHMARK_DEFINE_F(ConvolutionBenchmark, CPU)(benchmark::State &state) {
  for (auto _ : state) {
    convolution_2d(input.data(), output.data(), kernel.data(), width, height, kernel_size);
    benchmark::DoNotOptimize(output.data());
  }

  // Calculate throughput
  int64_t ops_per_iteration = static_cast<int64_t>(width) * height * kernel_size * kernel_size * 2; // multiply + add
  state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) * ops_per_iteration);
  state.counters["pixels"] = width * height;
  state.counters["kernel_size"] = kernel_size;
}

BENCHMARK_DEFINE_F(ConvolutionBenchmark, GPU_GlobalMem)(benchmark::State &state) {
  for (auto _ : state) {
    convolution_2d_gpu(input.data(), output.data(), kernel.data(), width, height, kernel_size);
    benchmark::DoNotOptimize(output.data());
  }
  int64_t ops_per_iteration = static_cast<int64_t>(width) * height * kernel_size * kernel_size * 2;
  state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) * ops_per_iteration);
  state.counters["pixels"] = width * height;
  state.counters["kernel_size"] = kernel_size;
}

BENCHMARK_DEFINE_F(ConvolutionBenchmark, GPU_ConstMem)(benchmark::State &state) {
  for (auto _ : state) {
    convolution_2d_gpu_constmem(input.data(), output.data(), kernel.data(), width, height, kernel_size);
    benchmark::DoNotOptimize(output.data());
  }
  int64_t ops_per_iteration = static_cast<int64_t>(width) * height * kernel_size * kernel_size * 2;
  state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) * ops_per_iteration);
  state.counters["pixels"] = width * height;
  state.counters["kernel_size"] = kernel_size;
}

BENCHMARK_DEFINE_F(ConvolutionBenchmark, GPU_ConstSharedMem)(benchmark::State &state) {
  for (auto _ : state) {
    convolution_2d_gpu_const_shared(input.data(), output.data(), kernel.data(), width, height, kernel_size);
    benchmark::DoNotOptimize(output.data());
  }
  int64_t ops_per_iteration = static_cast<int64_t>(width) * height * kernel_size * kernel_size * 2;
  state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) * ops_per_iteration);
  state.counters["pixels"] = width * height;
  state.counters["kernel_size"] = kernel_size;
}

// Small images with different kernel sizes
BENCHMARK_REGISTER_F(ConvolutionBenchmark, CPU)
    ->Name("Convolution/CPU/Small")
    ->Args({512, 512, 3})   // 512x512, 3x3 kernel
    ->Args({512, 512, 5})   // 512x512, 5x5 kernel
    ->Args({512, 512, 7})   // 512x512, 7x7 kernel
    ->Args({512, 512, 9})   // 512x512, 9x9 kernel
    ->Unit(benchmark::kMillisecond)
    ->UseRealTime();

BENCHMARK_REGISTER_F(ConvolutionBenchmark, GPU_GlobalMem)
    ->Name("Convolution/GPU_GlobalMem/Small")
    ->Args({512, 512, 3})
    ->Args({512, 512, 5})
    ->Args({512, 512, 7})
    ->Args({512, 512, 9})
    ->Unit(benchmark::kMillisecond)
    ->UseRealTime();

BENCHMARK_REGISTER_F(ConvolutionBenchmark, GPU_ConstMem)
    ->Name("Convolution/GPU_ConstMem/Small")
    ->Args({512, 512, 3})
    ->Args({512, 512, 5})
    ->Args({512, 512, 7})
    ->Args({512, 512, 9})
    ->Unit(benchmark::kMillisecond)
    ->UseRealTime();
BENCHMARK_REGISTER_F(ConvolutionBenchmark, GPU_ConstSharedMem)
  ->Name("Convolution/GPU_ConstSharedMem/Small")
  ->Args({512, 512, 3})
  ->Args({512, 512, 5})
  ->Args({512, 512, 7})
  ->Args({512, 512, 9})
  ->Unit(benchmark::kMillisecond)
  ->UseRealTime();

// Medium images
BENCHMARK_REGISTER_F(ConvolutionBenchmark, CPU)
    ->Name("Convolution/CPU/Medium")
    ->Args({1024, 1024, 3})
    ->Args({1024, 1024, 5})
    ->Args({1024, 1024, 7})
    ->Unit(benchmark::kMillisecond)
    ->UseRealTime();

BENCHMARK_REGISTER_F(ConvolutionBenchmark, GPU_GlobalMem)
    ->Name("Convolution/GPU_GlobalMem/Medium")
    ->Args({1024, 1024, 3})
    ->Args({1024, 1024, 5})
    ->Args({1024, 1024, 7})
    ->Unit(benchmark::kMillisecond)
    ->UseRealTime();

BENCHMARK_REGISTER_F(ConvolutionBenchmark, GPU_ConstMem)
    ->Name("Convolution/GPU_ConstMem/Medium")
    ->Args({1024, 1024, 3})
    ->Args({1024, 1024, 5})
    ->Args({1024, 1024, 7})
    ->Unit(benchmark::kMillisecond)
    ->UseRealTime();
BENCHMARK_REGISTER_F(ConvolutionBenchmark, GPU_ConstSharedMem)
  ->Name("Convolution/GPU_ConstSharedMem/Medium")
  ->Args({1024, 1024, 3})
  ->Args({1024, 1024, 5})
  ->Args({1024, 1024, 7})
  ->Unit(benchmark::kMillisecond)
  ->UseRealTime();

// Large images
BENCHMARK_REGISTER_F(ConvolutionBenchmark, CPU)
    ->Name("Convolution/CPU/Large")
    ->Args({2048, 2048, 5})
    ->Args({4096, 4096, 5})
    ->Unit(benchmark::kMillisecond)
    ->UseRealTime()
    ->Iterations(5);

BENCHMARK_REGISTER_F(ConvolutionBenchmark, GPU_GlobalMem)
    ->Name("Convolution/GPU_GlobalMem/Large")
    ->Args({2048, 2048, 5})
    ->Args({4096, 4096, 5})
    ->Unit(benchmark::kMillisecond)
    ->UseRealTime()
    ->Iterations(5);

BENCHMARK_REGISTER_F(ConvolutionBenchmark, GPU_ConstMem)
    ->Name("Convolution/GPU_ConstMem/Large")
    ->Args({2048, 2048, 5})
    ->Args({4096, 4096, 5})
    ->Unit(benchmark::kMillisecond)
    ->UseRealTime()
    ->Iterations(5);
BENCHMARK_REGISTER_F(ConvolutionBenchmark, GPU_ConstSharedMem)
  ->Name("Convolution/GPU_ConstSharedMem/Large")
  ->Args({2048, 2048, 5})
  ->Args({4096, 4096, 5})
  ->Unit(benchmark::kMillisecond)
  ->UseRealTime()
  ->Iterations(5);

BENCHMARK_MAIN();
