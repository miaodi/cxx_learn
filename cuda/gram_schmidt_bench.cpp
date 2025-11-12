#include "gram_schmidt.h"
#include <benchmark/benchmark.h>
#include <random>
#include <vector>

// Templated Gram-Schmidt Orthogonalization Benchmarks
template <typename T>
class GramSchmidtTemplatedBenchmark : public benchmark::Fixture {
public:
  void SetUp(const ::benchmark::State &state) override {
    // Check if we have two ranges (m and n) or one (square matrix)
    // Use range(1) to check if a second range exists
    m = state.range(0);
    n = (state.range(1) > 0) ? state.range(1) : state.range(0);

    input_matrix.resize(m * n);
    gram_schmidt::generateRandomMatrix(input_matrix.data(), m, n, 42);
  }

  void TearDown([[maybe_unused]] const ::benchmark::State &state) override {
    // Cleanup if needed
  }

protected:
  std::vector<T> input_matrix;
  int m, n;
};

// Float benchmarks
class GramSchmidtFloatBenchmark : public GramSchmidtTemplatedBenchmark<float> {
};
class GramSchmidtDoubleBenchmark
    : public GramSchmidtTemplatedBenchmark<double> {};

BENCHMARK_DEFINE_F(GramSchmidtFloatBenchmark,
                   DeviceMemory)(benchmark::State &state) {
  for (auto _ : state) {
    try {
      gram_schmidt::GramSchmidtOrthogonalizer<float> orthogonalizer(
          m, n, gram_schmidt::MemoryScheme::DEVICE_MEMORY);
      float *result_matrix;
      orthogonalizer.orthogonalize(input_matrix.data(), &result_matrix);
      benchmark::DoNotOptimize(result_matrix);
    } catch (const std::exception &e) {
      state.SkipWithError(e.what());
    }
  }

  // Calculate FLOPS (approximate for Gram-Schmidt)
  int64_t flops = 0;
  for (int j = 0; j < n; ++j) {
    flops += 3 * m * j + m; // dot products + norm + scaling
  }
  state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) * flops);
}

BENCHMARK_DEFINE_F(GramSchmidtFloatBenchmark,
                   HostMemory)(benchmark::State &state) {
  for (auto _ : state) {
    try {
      gram_schmidt::GramSchmidtOrthogonalizer<float> orthogonalizer(
          m, n, gram_schmidt::MemoryScheme::HOST_MEMORY);
      float *result_matrix;
      orthogonalizer.orthogonalize(input_matrix.data(), &result_matrix);
      benchmark::DoNotOptimize(result_matrix);
    } catch (const std::exception &e) {
      state.SkipWithError(e.what());
    }
  }

  int64_t flops = 0;
  for (int j = 0; j < n; ++j) {
    flops += 3 * m * j + m;
  }
  state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) * flops);
}

BENCHMARK_DEFINE_F(GramSchmidtFloatBenchmark,
                   UnifiedMemory)(benchmark::State &state) {
  for (auto _ : state) {
    try {
      gram_schmidt::GramSchmidtOrthogonalizer<float> orthogonalizer(
          m, n, gram_schmidt::MemoryScheme::UNIFIED_MEMORY);
      float *result_matrix;
      orthogonalizer.orthogonalize(input_matrix.data(), &result_matrix);
      benchmark::DoNotOptimize(result_matrix);
    } catch (const std::exception &e) {
      state.SkipWithError(e.what());
    }
  }

  int64_t flops = 0;
  for (int j = 0; j < n; ++j) {
    flops += 3 * m * j + m;
  }
  state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) * flops);
}

// Double benchmarks
BENCHMARK_DEFINE_F(GramSchmidtDoubleBenchmark,
                   DeviceMemory)(benchmark::State &state) {
  for (auto _ : state) {
    try {
      gram_schmidt::GramSchmidtOrthogonalizer<double> orthogonalizer(
          m, n, gram_schmidt::MemoryScheme::DEVICE_MEMORY);
      double *result_matrix;
      orthogonalizer.orthogonalize(input_matrix.data(), &result_matrix);
      benchmark::DoNotOptimize(result_matrix);
    } catch (const std::exception &e) {
      state.SkipWithError(e.what());
    }
  }

  int64_t flops = 0;
  for (int j = 0; j < n; ++j) {
    flops += 3 * m * j + m;
  }
  state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) * flops);
}

BENCHMARK_DEFINE_F(GramSchmidtDoubleBenchmark,
                   HostMemory)(benchmark::State &state) {
  for (auto _ : state) {
    try {
      gram_schmidt::GramSchmidtOrthogonalizer<double> orthogonalizer(
          m, n, gram_schmidt::MemoryScheme::HOST_MEMORY);
      double *result_matrix;
      orthogonalizer.orthogonalize(input_matrix.data(), &result_matrix);
      benchmark::DoNotOptimize(result_matrix);
    } catch (const std::exception &e) {
      state.SkipWithError(e.what());
    }
  }

  int64_t flops = 0;
  for (int j = 0; j < n; ++j) {
    flops += 3 * m * j + m;
  }
  state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) * flops);
}

BENCHMARK_DEFINE_F(GramSchmidtDoubleBenchmark,
                   UnifiedMemory)(benchmark::State &state) {
  for (auto _ : state) {
    try {
      gram_schmidt::GramSchmidtOrthogonalizer<double> orthogonalizer(
          m, n, gram_schmidt::MemoryScheme::UNIFIED_MEMORY);
      double *result_matrix;
      orthogonalizer.orthogonalize(input_matrix.data(), &result_matrix);
      benchmark::DoNotOptimize(result_matrix);
    } catch (const std::exception &e) {
      state.SkipWithError(e.what());
    }
  }

  int64_t flops = 0;
  for (int j = 0; j < n; ++j) {
    flops += 3 * m * j + m;
  }
  state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) * flops);
}

// Register Float benchmarks
BENCHMARK_REGISTER_F(GramSchmidtFloatBenchmark, DeviceMemory)
    ->Range(32, 128) // Square matrices from 32x32 to 128x128
    ->Unit(benchmark::kMillisecond)
    ->UseRealTime();

BENCHMARK_REGISTER_F(GramSchmidtFloatBenchmark, HostMemory)
    ->Range(32, 128)
    ->Unit(benchmark::kMillisecond)
    ->UseRealTime();

BENCHMARK_REGISTER_F(GramSchmidtFloatBenchmark, UnifiedMemory)
    ->Range(32, 128)
    ->Unit(benchmark::kMillisecond)
    ->UseRealTime();

// Register Double benchmarks
BENCHMARK_REGISTER_F(GramSchmidtDoubleBenchmark, DeviceMemory)
    ->Range(32, 128) // Square matrices from 32x32 to 128x128
    ->Unit(benchmark::kMillisecond)
    ->UseRealTime();

BENCHMARK_REGISTER_F(GramSchmidtDoubleBenchmark, HostMemory)
    ->Range(32, 128)
    ->Unit(benchmark::kMillisecond)
    ->UseRealTime();

BENCHMARK_REGISTER_F(GramSchmidtDoubleBenchmark, UnifiedMemory)
    ->Range(32, 128)
    ->Unit(benchmark::kMillisecond)
    ->UseRealTime();

// Very tall matrix benchmarks for all three memory types

static void BM_GramSchmidt_Rectangular_Float_Device(benchmark::State &state) {
  int m = state.range(0);
  int n = state.range(1);
  std::vector<float> input_matrix(m * n);
  gram_schmidt::generateRandomMatrix(input_matrix.data(), m, n, 42);

  for (auto _ : state) {
    try {
      gram_schmidt::GramSchmidtOrthogonalizer<float> orthogonalizer(
          m, n, gram_schmidt::MemoryScheme::DEVICE_MEMORY);
      float *result_matrix;
      orthogonalizer.orthogonalize(input_matrix.data(), &result_matrix);
      benchmark::DoNotOptimize(result_matrix);
    } catch (const std::exception &e) {
      state.SkipWithError(e.what());
    }
  }
}

static void BM_GramSchmidt_Rectangular_Float_Host(benchmark::State &state) {
  int m = state.range(0);
  int n = state.range(1);
  std::vector<float> input_matrix(m * n);
  gram_schmidt::generateRandomMatrix(input_matrix.data(), m, n, 42);

  for (auto _ : state) {
    try {
      gram_schmidt::GramSchmidtOrthogonalizer<float> orthogonalizer(
          m, n, gram_schmidt::MemoryScheme::HOST_MEMORY);
      float *result_matrix;
      orthogonalizer.orthogonalize(input_matrix.data(), &result_matrix);
      benchmark::DoNotOptimize(result_matrix);
    } catch (const std::exception &e) {
      state.SkipWithError(e.what());
    }
  }
}

static void BM_GramSchmidt_Rectangular_Float_Unified(benchmark::State &state) {
  int m = state.range(0);
  int n = state.range(1);
  std::vector<float> input_matrix(m * n);
  gram_schmidt::generateRandomMatrix(input_matrix.data(), m, n, 42);

  for (auto _ : state) {
    try {
      gram_schmidt::GramSchmidtOrthogonalizer<float> orthogonalizer(
          m, n, gram_schmidt::MemoryScheme::UNIFIED_MEMORY);
      float *result_matrix;
      orthogonalizer.orthogonalize(input_matrix.data(), &result_matrix);
      benchmark::DoNotOptimize(result_matrix);
    } catch (const std::exception &e) {
      state.SkipWithError(e.what());
    }
  }
}

static void BM_GramSchmidt_Rectangular_Double_Device(benchmark::State &state) {
  int m = state.range(0);
  int n = state.range(1);
  std::vector<double> input_matrix(m * n);
  gram_schmidt::generateRandomMatrix(input_matrix.data(), m, n, 42);

  for (auto _ : state) {
    try {
      gram_schmidt::GramSchmidtOrthogonalizer<double> orthogonalizer(
          m, n, gram_schmidt::MemoryScheme::DEVICE_MEMORY);
      double *result_matrix;
      orthogonalizer.orthogonalize(input_matrix.data(), &result_matrix);
      benchmark::DoNotOptimize(result_matrix);
    } catch (const std::exception &e) {
      state.SkipWithError(e.what());
    }
  }
}

static void BM_GramSchmidt_Rectangular_Double_Host(benchmark::State &state) {
  int m = state.range(0);
  int n = state.range(1);
  std::vector<double> input_matrix(m * n);
  gram_schmidt::generateRandomMatrix(input_matrix.data(), m, n, 42);

  for (auto _ : state) {
    try {
      gram_schmidt::GramSchmidtOrthogonalizer<double> orthogonalizer(
          m, n, gram_schmidt::MemoryScheme::HOST_MEMORY);
      double *result_matrix;
      orthogonalizer.orthogonalize(input_matrix.data(), &result_matrix);
      benchmark::DoNotOptimize(result_matrix);
    } catch (const std::exception &e) {
      state.SkipWithError(e.what());
    }
  }
}

static void BM_GramSchmidt_Rectangular_Double_Unified(benchmark::State &state) {
  int m = state.range(0);
  int n = state.range(1);
  std::vector<double> input_matrix(m * n);
  gram_schmidt::generateRandomMatrix(input_matrix.data(), m, n, 42);

  for (auto _ : state) {
    try {
      gram_schmidt::GramSchmidtOrthogonalizer<double> orthogonalizer(
          m, n, gram_schmidt::MemoryScheme::UNIFIED_MEMORY);
      double *result_matrix;
      orthogonalizer.orthogonalize(input_matrix.data(), &result_matrix);
      benchmark::DoNotOptimize(result_matrix);
    } catch (const std::exception &e) {
      state.SkipWithError(e.what());
    }
  }
}

// Register very tall matrix benchmarks for all memory types
BENCHMARK(BM_GramSchmidt_Rectangular_Float_Device)
    ->Args({1000000, 1000})
    ->Unit(benchmark::kMillisecond);

BENCHMARK(BM_GramSchmidt_Rectangular_Float_Host)
    ->Args({1000000, 1000})
    ->Unit(benchmark::kMillisecond);

BENCHMARK(BM_GramSchmidt_Rectangular_Float_Unified)
    ->Args({1000000, 1000})
    ->Unit(benchmark::kMillisecond);

BENCHMARK(BM_GramSchmidt_Rectangular_Double_Device)
    ->Args({1000000, 1000})
    ->Unit(benchmark::kMillisecond);

BENCHMARK(BM_GramSchmidt_Rectangular_Double_Host)
    ->Args({1000000, 1000})
    ->Unit(benchmark::kMillisecond);

BENCHMARK(BM_GramSchmidt_Rectangular_Double_Unified)
    ->Args({1000000, 1000})
    ->Unit(benchmark::kMillisecond);

// Simple benchmarks for direct comparison
static void BM_GramSchmidt_Float_Device_64(benchmark::State &state) {
  const int n = 64;
  std::vector<float> input_matrix(n * n);
  gram_schmidt::generateRandomMatrix(input_matrix.data(), n, n, 42);

  for (auto _ : state) {
    try {
      gram_schmidt::GramSchmidtOrthogonalizer<float> orthogonalizer(
          n, n, gram_schmidt::MemoryScheme::DEVICE_MEMORY);
      float *result_matrix;
      orthogonalizer.orthogonalize(input_matrix.data(), &result_matrix);
      benchmark::DoNotOptimize(result_matrix);
    } catch (const std::exception &e) {
      state.SkipWithError(e.what());
    }
  }
}

static void BM_GramSchmidt_Double_Device_64(benchmark::State &state) {
  const int n = 64;
  std::vector<double> input_matrix(n * n);
  gram_schmidt::generateRandomMatrix(input_matrix.data(), n, n, 42);

  for (auto _ : state) {
    try {
      gram_schmidt::GramSchmidtOrthogonalizer<double> orthogonalizer(
          n, n, gram_schmidt::MemoryScheme::DEVICE_MEMORY);
      double *result_matrix;
      orthogonalizer.orthogonalize(input_matrix.data(), &result_matrix);
      benchmark::DoNotOptimize(result_matrix);
    } catch (const std::exception &e) {
      state.SkipWithError(e.what());
    }
  }
}
// Register simple benchmarks
BENCHMARK(BM_GramSchmidt_Float_Device_64)->Unit(benchmark::kMillisecond);

BENCHMARK(BM_GramSchmidt_Double_Device_64)->Unit(benchmark::kMillisecond);

BENCHMARK_MAIN();