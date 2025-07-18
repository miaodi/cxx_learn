#include "func.h"
#include <benchmark/benchmark.h>
#include <vector>

static void BM_TwiddleOrigin(benchmark::State &state) {
  std::uint64_t u = 123456789, v = 987654321;
  std::uint64_t result = 0;
  for (auto _ : state) {
    for (size_t i = 0; i < state.range(0); ++i) {
      result ^= MCRand::twiddle_origin(u, v);
      ++u;
      ++v;
      benchmark::DoNotOptimize(result);
    }
  }
}

static void BM_TwiddleNew(benchmark::State &state) {
  std::uint64_t u = 123456789, v = 987654321;
  std::uint64_t result = 0;
  for (auto _ : state) {
    for (size_t i = 0; i < state.range(0); ++i) {
      result ^= MCRand::twiddle_new(u, v);
      ++u;
      ++v;
      benchmark::DoNotOptimize(result);
    }
  }
}

BENCHMARK(BM_TwiddleOrigin)->RangeMultiplier(16)->Range(1 << 0, 1 << 20);
BENCHMARK(BM_TwiddleNew)->RangeMultiplier(16)->Range(1 << 0, 1 << 20);

BENCHMARK_MAIN();