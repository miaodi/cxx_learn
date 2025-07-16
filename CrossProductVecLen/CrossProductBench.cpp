#include <algorithm>
#include <benchmark/benchmark.h>
#include <immintrin.h> // AVX2/FMA
#include <iostream>
#include <random>
#include <vector>

void CrossProd(double const *a, double const *b, double *cross_prod) {
  cross_prod[0] = a[1] * b[2] - a[2] * b[1];
  cross_prod[1] = a[2] * b[0] - a[0] * b[2];
  cross_prod[2] = a[0] * b[1] - a[1] * b[0];
}

static void CrossProdBM(benchmark::State &state) {
  // First create an instance of an engine.
  std::random_device rnd_device;
  // Specify the engine and distribution.
  std::mt19937 mersenne_engine{rnd_device()}; // Generates random integers
  std::uniform_real_distribution<double> dist{-1.0,
                                              1.0}; // Range for double values
  auto gen = [&]() { return dist(mersenne_engine); };
  const auto size = state.range(0);
  const size_t vec_size = 1000;
  std::vector<double[3]> a(vec_size);
  std::vector<double[3]> b(vec_size);
  std::vector<double[3]> cross_prod(vec_size);
  for (size_t i = 0; i < vec_size; ++i) {
    a[i][0] = gen();
    a[i][1] = gen();
    a[i][2] = gen();
    b[i][0] = gen();
    b[i][1] = gen();
    b[i][2] = gen();
  }
  for (auto _ : state) {
    for (size_t i = 0; i < size; i++) {
      CrossProd(a[i % vec_size], b[i % vec_size], cross_prod[i % vec_size]);
      benchmark::DoNotOptimize(cross_prod[i % vec_size]);
    }
  }
}

BENCHMARK(CrossProdBM)->RangeMultiplier(2)->Range(1 << 1, 1 << 16);

struct alignas(32) AlignedDouble3 {
  double data[4];
  AlignedDouble3() : data{0.0, 0.0, 0.0, 0.0} {}
  double &operator[](size_t index) { return data[index]; }
};

static void CrossProdAlignBM(benchmark::State &state) {
  // First create an instance of an engine.
  std::random_device rnd_device;
  // Specify the engine and distribution.
  std::mt19937 mersenne_engine{rnd_device()}; // Generates random integers
  std::uniform_real_distribution<double> dist{-1.0,
                                              1.0}; // Range for double values
  auto gen = [&]() { return dist(mersenne_engine); };
  const auto size = state.range(0);
  const size_t vec_size = 1000;
  std::vector<AlignedDouble3> a(vec_size);
  std::vector<AlignedDouble3> b(vec_size);
  std::vector<AlignedDouble3> cross_prod(vec_size);
  for (size_t i = 0; i < vec_size; ++i) {
    a[i][0] = gen();
    a[i][1] = gen();
    a[i][2] = gen();
    b[i][0] = gen();
    b[i][1] = gen();
    b[i][2] = gen();
  }
  for (auto _ : state) {
    for (size_t i = 0; i < size; i++) {
      CrossProd(a[i % vec_size].data, b[i % vec_size].data,
                cross_prod[i % vec_size].data);
      benchmark::DoNotOptimize(cross_prod[i % vec_size].data);
    }
  }
}
BENCHMARK(CrossProdAlignBM)->RangeMultiplier(2)->Range(1 << 1, 1 << 16);

void CrossProdAVX2(const double *a, const double *b, double *cross_prod) {
  __m256d va = _mm256_load_pd(a); // [a0, a1, a2, a3]
  __m256d vb = _mm256_load_pd(b); // [b0, b1, b2, b3]

  // Permute: [a1, a2, a0, a3] and [b1, b2, b0, b3]
  __m256d va_yzx = _mm256_permute4x64_pd(va, _MM_SHUFFLE(3, 0, 2, 1));
  __m256d vb_yzx = _mm256_permute4x64_pd(vb, _MM_SHUFFLE(3, 0, 2, 1));

  // Compute cross product using only 3 permutations
  __m256d mul1 = _mm256_mul_pd(va, vb_yzx);
  __m256d mul2 = _mm256_mul_pd(vb, va_yzx);
  __m256d result = _mm256_sub_pd(mul1, mul2);

  // Final permutation to get the correct order: [res2, res0, res1, res3]
  __m256d result_yzx = _mm256_permute4x64_pd(result, _MM_SHUFFLE(3, 1, 0, 2));

  _mm256_store_pd(cross_prod, result_yzx);
}

static void CrossProdAVX2BM(benchmark::State &state) {
  // First create an instance of an engine.
  std::random_device rnd_device;
  // Specify the engine and distribution.
  std::mt19937 mersenne_engine{rnd_device()}; // Generates random integers
  std::uniform_real_distribution<double> dist{-1.0,
                                              1.0}; // Range for double values
  auto gen = [&]() { return dist(mersenne_engine); };
  const auto size = state.range(0);
  const size_t vec_size = 1000;
  std::vector<AlignedDouble3> a(vec_size);
  std::vector<AlignedDouble3> b(vec_size);
  std::vector<AlignedDouble3> cross_prod(vec_size);
  for (size_t i = 0; i < vec_size; ++i) {
    a[i][0] = gen();
    a[i][1] = gen();
    a[i][2] = gen();
    b[i][0] = gen();
    b[i][1] = gen();
    b[i][2] = gen();
  }
  for (auto _ : state) {
    for (size_t i = 0; i < size; i++) {
      CrossProdAVX2(a[i % vec_size].data, b[i % vec_size].data,
                    cross_prod[i % vec_size].data);
      benchmark::DoNotOptimize(cross_prod[i % vec_size].data);
    }
  }
}
BENCHMARK(CrossProdAVX2BM)->RangeMultiplier(2)->Range(1 << 1, 1 << 16);
BENCHMARK_MAIN();