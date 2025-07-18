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

static void CrossProAVX2InlineBM(benchmark::State &state) {
  const auto size = state.range(0);
  const size_t vec_size = 1000;
  std::vector<M256D> a(vec_size);
  std::vector<M256D> b(vec_size);
  std::vector<M256D> cross_prod(vec_size);
  FillRandom(a.begin(), a.end());
  FillRandom(b.begin(), b.end());
  for (auto _ : state) {
    for (size_t i = 0; i < size; i++) {
      CrossProdAVX2Inline(a[i % vec_size].m256d, b[i % vec_size].m256d,
                          cross_prod[i % vec_size].m256d);
      benchmark::DoNotOptimize(cross_prod[i % vec_size].data[0]);
    }
  }
}

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

static void DotProdAVX2InlineBM(benchmark::State &state) {
  const auto size = state.range(0);
  const size_t vec_size = 1000;
  std::vector<M256D> a(vec_size);
  std::vector<M256D> b(vec_size);
  FillRandom(a.begin(), a.end());
  FillRandom(b.begin(), b.end());
  for (auto _ : state) {
    for (size_t i = 0; i < size; i++) {
      double result = DotProdAVX2Inline(a[i % vec_size].m256d,
                                        b[i % vec_size].m256d);
      benchmark::DoNotOptimize(result);
    }
  }
}

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

#define APPLY_BENCHMARK_SETTINGS(fn)                                           \
  BENCHMARK(fn)->RangeMultiplier(8)->Range(1 << 1, 1 << 20);

APPLY_BENCHMARK_SETTINGS(CrossProdBM)
APPLY_BENCHMARK_SETTINGS(CrossProdAlignBM)
APPLY_BENCHMARK_SETTINGS(CrossProdAVX2BM)
APPLY_BENCHMARK_SETTINGS(CrossProAVX2InlineBM)
APPLY_BENCHMARK_SETTINGS(DotProdBM)
APPLY_BENCHMARK_SETTINGS(DotProdAlignBM)
APPLY_BENCHMARK_SETTINGS(DotProdAVX2BM)
APPLY_BENCHMARK_SETTINGS(DotProdAVX2InlineBM)
APPLY_BENCHMARK_SETTINGS(DotProdFMABM)
BENCHMARK_MAIN();