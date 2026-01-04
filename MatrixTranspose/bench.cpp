#include "transpose.hpp"
#include <benchmark/benchmark.h>
#include <random>
#include <vector>

using namespace matrix_transpose;

// ============================================================================
// Benchmark Fixture
// ============================================================================

class TransposeBenchmark : public benchmark::Fixture {
public:
  void SetUp(const ::benchmark::State &state) override {
    M = state.range(0);
    N = state.range(1);
    
    input.resize(M * N);
    output.resize(M * N);
    
    // Initialize with random data
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (auto &val : input) {
      val = dist(rng);
    }
  }
  
  void TearDown([[maybe_unused]] const ::benchmark::State &state) override {
    // Optional: verify correctness
    // std::vector<float> reference(M * N);
    // NaiveTranspose(input.data(), reference.data(), M, N);
    // if (!VerifyTranspose(input.data(), output.data(), M, N)) {
    //   std::cerr << "Transpose verification failed!\n";
    // }
  }

protected:
  int M, N;
  std::vector<float> input;
  std::vector<float> output;
};

// ============================================================================
// Out-of-Place Transpose Benchmarks
// ============================================================================

BENCHMARK_DEFINE_F(TransposeBenchmark, Naive)(benchmark::State &state) {
  for (auto _ : state) {
    NaiveTranspose(input.data(), output.data(), M, N);
    benchmark::DoNotOptimize(output.data());
    benchmark::ClobberMemory();
  }
  
  // Report throughput
  int64_t bytes = static_cast<int64_t>(M) * N * sizeof(float) * 2; // read + write
  state.SetBytesProcessed(state.iterations() * bytes);
  state.counters["M"] = M;
  state.counters["N"] = N;
}

BENCHMARK_DEFINE_F(TransposeBenchmark, Tiled16)(benchmark::State &state) {
  for (auto _ : state) {
    TiledTranspose<float, 16>(input.data(), output.data(), M, N);
    benchmark::DoNotOptimize(output.data());
    benchmark::ClobberMemory();
  }
  
  int64_t bytes = static_cast<int64_t>(M) * N * sizeof(float) * 2;
  state.SetBytesProcessed(state.iterations() * bytes);
  state.counters["M"] = M;
  state.counters["N"] = N;
  state.counters["TileSize"] = 16;
}

BENCHMARK_DEFINE_F(TransposeBenchmark, Tiled32)(benchmark::State &state) {
  for (auto _ : state) {
    TiledTranspose<float, 32>(input.data(), output.data(), M, N);
    benchmark::DoNotOptimize(output.data());
    benchmark::ClobberMemory();
  }
  
  int64_t bytes = static_cast<int64_t>(M) * N * sizeof(float) * 2;
  state.SetBytesProcessed(state.iterations() * bytes);
  state.counters["M"] = M;
  state.counters["N"] = N;
  state.counters["TileSize"] = 32;
}

BENCHMARK_DEFINE_F(TransposeBenchmark, Tiled64)(benchmark::State &state) {
  for (auto _ : state) {
    TiledTranspose<float, 64>(input.data(), output.data(), M, N);
    benchmark::DoNotOptimize(output.data());
    benchmark::ClobberMemory();
  }
  
  int64_t bytes = static_cast<int64_t>(M) * N * sizeof(float) * 2;
  state.SetBytesProcessed(state.iterations() * bytes);
  state.counters["M"] = M;
  state.counters["N"] = N;
  state.counters["TileSize"] = 64;
}

BENCHMARK_DEFINE_F(TransposeBenchmark, CacheOblivious)(benchmark::State &state) {
  for (auto _ : state) {
    CacheObliviousTranspose(input.data(), output.data(), M, N);
    benchmark::DoNotOptimize(output.data());
    benchmark::ClobberMemory();
  }
  
  int64_t bytes = static_cast<int64_t>(M) * N * sizeof(float) * 2;
  state.SetBytesProcessed(state.iterations() * bytes);
  state.counters["M"] = M;
  state.counters["N"] = N;
}

#if defined(__AVX2__)
BENCHMARK_DEFINE_F(TransposeBenchmark, AVX2_8x8)(benchmark::State &state) {
  for (auto _ : state) {
    TiledTranspose<float, 8, AVX2Kernel>(input.data(), output.data(), M, N);
    benchmark::DoNotOptimize(output.data());
    benchmark::ClobberMemory();
  }
  
  int64_t bytes = static_cast<int64_t>(M) * N * sizeof(float) * 2;
  state.SetBytesProcessed(state.iterations() * bytes);
  state.counters["M"] = M;
  state.counters["N"] = N;
  state.counters["TileSize"] = 8;
}
#endif

// ============================================================================
// Square Matrix Benchmarks
// ============================================================================

BENCHMARK_REGISTER_F(TransposeBenchmark, Naive)
    ->Args({256, 256})
    ->Args({512, 512})
    ->Args({1024, 1024})
    ->Args({2048, 2048})
    ->Args({4096, 4096})
    ->Unit(benchmark::kMillisecond)
    ->UseRealTime();

BENCHMARK_REGISTER_F(TransposeBenchmark, Tiled16)
    ->Args({256, 256})
    ->Args({512, 512})
    ->Args({1024, 1024})
    ->Args({2048, 2048})
    ->Args({4096, 4096})
    ->Unit(benchmark::kMillisecond)
    ->UseRealTime();

BENCHMARK_REGISTER_F(TransposeBenchmark, Tiled32)
    ->Args({256, 256})
    ->Args({512, 512})
    ->Args({1024, 1024})
    ->Args({2048, 2048})
    ->Args({4096, 4096})
    ->Unit(benchmark::kMillisecond)
    ->UseRealTime();

BENCHMARK_REGISTER_F(TransposeBenchmark, Tiled64)
    ->Args({256, 256})
    ->Args({512, 512})
    ->Args({1024, 1024})
    ->Args({2048, 2048})
    ->Args({4096, 4096})
    ->Unit(benchmark::kMillisecond)
    ->UseRealTime();

BENCHMARK_REGISTER_F(TransposeBenchmark, CacheOblivious)
    ->Args({256, 256})
    ->Args({512, 512})
    ->Args({1024, 1024})
    ->Args({2048, 2048})
    ->Args({4096, 4096})
    ->Unit(benchmark::kMillisecond)
    ->UseRealTime();

#if defined(__AVX2__)
BENCHMARK_REGISTER_F(TransposeBenchmark, AVX2_8x8)
    ->Args({256, 256})
    ->Args({512, 512})
    ->Args({1024, 1024})
    ->Args({2048, 2048})
    ->Args({4096, 4096})
    ->Unit(benchmark::kMillisecond)
    ->UseRealTime();
#endif

// ============================================================================
// Rectangular Matrix Benchmarks
// ============================================================================

BENCHMARK_REGISTER_F(TransposeBenchmark, Naive)
    ->Name("TransposeBenchmark/Naive/Rectangular")
    ->Args({1024, 256})   // Tall
    ->Args({256, 1024})   // Wide
    ->Args({2048, 512})   // Very tall
    ->Args({512, 2048})   // Very wide
    ->Args({4096, 1024})  // Extremely tall
    ->Args({1024, 4096})  // Extremely wide
    ->Unit(benchmark::kMillisecond)
    ->UseRealTime();

BENCHMARK_REGISTER_F(TransposeBenchmark, Tiled32)
    ->Name("TransposeBenchmark/Tiled32/Rectangular")
    ->Args({1024, 256})
    ->Args({256, 1024})
    ->Args({2048, 512})
    ->Args({512, 2048})
    ->Args({4096, 1024})
    ->Args({1024, 4096})
    ->Unit(benchmark::kMillisecond)
    ->UseRealTime();

BENCHMARK_MAIN();
