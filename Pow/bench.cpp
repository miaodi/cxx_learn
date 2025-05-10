#include <cmath>
#include <benchmark/benchmark.h>

static void Multi(benchmark::State &state) {

  for (auto _ : state) {
    double pow = 1.;
    for (int i = 0; i < state.range(0); i++)
      benchmark::DoNotOptimize(pow *= 2.);
  }
}

BENCHMARK(Multi)->RangeMultiplier(2)->Range(1 << 0, 1 << 10);

static void STLPow(benchmark::State &state) {

  for (auto _ : state) {
    double pow = 1.;
    benchmark::DoNotOptimize(pow = std::pow(2., state.range(0)));
  }
}

BENCHMARK(STLPow)->RangeMultiplier(2)->Range(1 << 0, 1 << 10);

BENCHMARK_MAIN();