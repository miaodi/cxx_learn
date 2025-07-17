#include "VectorOps.h"
#include <algorithm>
#include <benchmark/benchmark.h>
#include <immintrin.h> // AVX2/FMA
#include <iostream>
#include <vector>

static void CrossProdBM(benchmark::State &state) {
  const auto size = state.range(0);
  const size_t vec_size = 1000;
  std::vector<double[3]> a(vec_size);
  std::vector<double[3]> b(vec_size);
  std::vector<double[3]> cross_prod(vec_size);

  // Initialize vectors a and b with random values
  FillRandom(a.begin(), a.end());
  FillRandom(b.begin(), b.end());
  for (auto _ : state) {
    for (size_t i = 0; i < size; i++) {
      CrossProd(a[i % vec_size], b[i % vec_size], cross_prod[i % vec_size]);
      benchmark::DoNotOptimize(cross_prod[i % vec_size]);
    }
  }
}

BENCHMARK(CrossProdBM)->RangeMultiplier(2)->Range(1 << 1, 1 << 16);
static void CrossProdAlignBM(benchmark::State &state) {
  const auto size = state.range(0);
  const size_t vec_size = 1000;
  std::vector<AlignedDouble3> a(vec_size);
  std::vector<AlignedDouble3> b(vec_size);
  std::vector<AlignedDouble3> cross_prod(vec_size);
  FillRandom(a.begin(), a.end());
  FillRandom(b.begin(), b.end());
  for (auto _ : state) {
    for (size_t i = 0; i < size; i++) {
      CrossProd(a[i % vec_size].data, b[i % vec_size].data,
                cross_prod[i % vec_size].data);
      benchmark::DoNotOptimize(cross_prod[i % vec_size].data);
    }
  }
}
BENCHMARK(CrossProdAlignBM)->RangeMultiplier(2)->Range(1 << 1, 1 << 16);

static void CrossProdAVX2BM(benchmark::State &state) {
  const auto size = state.range(0);
  const size_t vec_size = 1000;
  std::vector<AlignedDouble3> a(vec_size);
  std::vector<AlignedDouble3> b(vec_size);
  std::vector<AlignedDouble3> cross_prod(vec_size);
  FillRandom(a.begin(), a.end());
  FillRandom(b.begin(), b.end());
  for (auto _ : state) {
    for (size_t i = 0; i < size; i++) {
      CrossProdAVX2(a[i % vec_size].data, b[i % vec_size].data,
                    cross_prod[i % vec_size].data);
      benchmark::DoNotOptimize(cross_prod[i % vec_size].data);
    }
  }
}
BENCHMARK(CrossProdAVX2BM)->RangeMultiplier(2)->Range(1 << 1, 1 << 16);

static void DotProdBM(benchmark::State &state) {
  const auto size = state.range(0);
  const size_t vec_size = 1000;
  std::vector<double[3]> a(vec_size);
  std::vector<double[3]> b(vec_size);
  FillRandom(a.begin(), a.end());
  FillRandom(b.begin(), b.end());
  for (auto _ : state) {
    for (size_t i = 0; i < size; i++) {
      double result = DotProd(a[i % vec_size], b[i % vec_size]);
      benchmark::DoNotOptimize(result);
    }
  }
}
BENCHMARK(DotProdBM)->RangeMultiplier(2)->Range(1 << 1, 1 << 16);

static void DotProdAlignBM(benchmark::State &state) {
  const auto size = state.range(0);
  const size_t vec_size = 1000;
  std::vector<AlignedDouble3> a(vec_size);
  std::vector<AlignedDouble3> b(vec_size);
  FillRandom(a.begin(), a.end());
  FillRandom(b.begin(), b.end());
  for (auto _ : state) {
    for (size_t i = 0; i < size; i++) {
      double result = DotProd(a[i % vec_size].data, b[i % vec_size].data);
      benchmark::DoNotOptimize(result);
    }
  }
}
BENCHMARK(DotProdAlignBM)->RangeMultiplier(2)->Range(1 << 1, 1 << 16);

static void DotProdAVX2BM(benchmark::State &state) {
  const auto size = state.range(0);
  const size_t vec_size = 1000;
  std::vector<AlignedDouble3> a(vec_size);
  std::vector<AlignedDouble3> b(vec_size);
  FillRandom(a.begin(), a.end());
  FillRandom(b.begin(), b.end());
  for (auto _ : state) {
    for (size_t i = 0; i < size; i++) {
      double result = DotProdAVX2(a[i % vec_size].data, b[i % vec_size].data);
      benchmark::DoNotOptimize(result);
    }
  }
}
BENCHMARK(DotProdAVX2BM)->RangeMultiplier(2)->Range(1 << 1, 1 << 16);

static void DotProdFMABM(benchmark::State &state) {
  const auto size = state.range(0);
  const size_t vec_size = 1000;
  std::vector<AlignedDouble3> a(vec_size);
  std::vector<AlignedDouble3> b(vec_size);
  FillRandom(a.begin(), a.end());
  FillRandom(b.begin(), b.end());
  for (auto _ : state) {
    for (size_t i = 0; i < size; i++) {
      double result = DotProdFMA(a[i % vec_size].data, b[i % vec_size].data);
      benchmark::DoNotOptimize(result);
    }
  }
}
BENCHMARK(DotProdFMABM)->RangeMultiplier(2)->Range(1 << 1, 1 << 16);

BENCHMARK_MAIN();