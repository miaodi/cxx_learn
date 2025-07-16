#include <benchmark/benchmark.h>
#include <vector>
#include <random>
#include <cmath>

static std::vector<double> a, b, c, out;

static void SetupVectors(size_t N) {
    a.resize(N);
    b.resize(N);
    c.resize(N);
    out.resize(N);
    std::mt19937 rng(42);
    std::uniform_real_distribution<double> dist(1.0, 10.0);
    for (size_t i = 0; i < N; ++i) {
        a[i] = dist(rng);
        b[i] = dist(rng);
        c[i] = dist(rng);
    }
}

// Benchmark manual multiply-subtract: a*b - c
static void BM_ManualMultiplySubtract(benchmark::State& state) {
    size_t N = static_cast<size_t>(state.range(0));
    SetupVectors(N);
    for (auto _ : state) {
        for (size_t i = 0; i < N; ++i) {
            out[i] = a[i] * b[i] - c[i];
        }
        benchmark::DoNotOptimize(out.data());
    }
}
BENCHMARK(BM_ManualMultiplySubtract)->Arg(1 << 24); // ~16M elements

// Benchmark std::fma
static void BM_StdFMA(benchmark::State& state) {
    size_t N = static_cast<size_t>(state.range(0));
    SetupVectors(N);
    for (auto _ : state) {
        for (size_t i = 0; i < N; ++i) {
            out[i] = std::fma(a[i], b[i], -c[i]);
        }
        benchmark::DoNotOptimize(out.data());
    }
}
BENCHMARK(BM_StdFMA)->Arg(1 << 24);

BENCHMARK_MAIN();