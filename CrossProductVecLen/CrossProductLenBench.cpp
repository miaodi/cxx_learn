#include <algorithm>
#include <benchmark/benchmark.h>
#include <immintrin.h> // AVX2/FMA
#include <iostream>
#include <random>
#include <vector>

double CrossProdVecLen1(double const *a, double const *b) {
  double cross_prod[3];
  cross_prod[0] = a[1] * b[2] - a[2] * b[1];
  cross_prod[1] = a[2] * b[0] - a[0] * b[2];
  cross_prod[2] = a[0] * b[1] - a[1] * b[0];

  return cross_prod[0] * cross_prod[0] + cross_prod[1] * cross_prod[1] +
         cross_prod[2] * cross_prod[2];
}

static void CrossProdVecLen1BM(benchmark::State &state) {
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
      benchmark::DoNotOptimize(
          CrossProdVecLen1(a[i % vec_size], b[i % vec_size]));
    }
  }
}

double CrossProdVecLen2(double const *a, double const *b) {
  double dot_ab = a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
  double norm2_a = a[0] * a[0] + a[1] * a[1] + a[2] * a[2];
  double norm2_b = b[0] * b[0] + b[1] * b[1] + b[2] * b[2];
  return norm2_a * norm2_b - dot_ab * dot_ab;
}

static void CrossProdVecLen2BM(benchmark::State &state) {

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
      benchmark::DoNotOptimize(
          CrossProdVecLen2(a[i % vec_size], b[i % vec_size]));
    }
  }
}


struct alignas(32) AlignedDouble3 {
  double data[4];
  AlignedDouble3() : data{0.0, 0.0, 0.0, 0.0} {}
  double &operator[](size_t index) { return data[index]; }
};

static void CrossProdVecLen2BMAlign(benchmark::State &state) {

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
      benchmark::DoNotOptimize(
          CrossProdVecLen2(&a[i % vec_size].data[0], &b[i % vec_size].data[0]));
    }
  }
}

double CrossProdVecLen2_avx2_aligned(const double *a, const double *b) {
  // Load 4 aligned doubles from a and b
  __m256d va = _mm256_load_pd(a); // a[0], a[1], a[2], a[3] == 0
  __m256d vb = _mm256_load_pd(b); // b[0], b[1], b[2], b[3] == 0

  // dot_ab = dot(a, b)
  __m256d ab = _mm256_mul_pd(va, vb);
  __m128d ab_low = _mm256_castpd256_pd128(ab);
  __m128d ab_high = _mm256_extractf128_pd(ab, 1);
  __m128d ab_sum2 = _mm_add_pd(ab_low, ab_high);
  __m128d ab_sum = _mm_hadd_pd(ab_sum2, ab_sum2);
  double dot_ab = _mm_cvtsd_f64(ab_sum);

  // norm2_a = dot(a, a)
  __m256d aa = _mm256_mul_pd(va, va);
  __m128d aa_low = _mm256_castpd256_pd128(aa);
  __m128d aa_high = _mm256_extractf128_pd(aa, 1);
  __m128d aa_sum2 = _mm_add_pd(aa_low, aa_high);
  __m128d aa_sum = _mm_hadd_pd(aa_sum2, aa_sum2);
  double norm2_a = _mm_cvtsd_f64(aa_sum);

  // norm2_b = dot(b, b)
  __m256d bb = _mm256_mul_pd(vb, vb);
  __m128d bb_low = _mm256_castpd256_pd128(bb);
  __m128d bb_high = _mm256_extractf128_pd(bb, 1);
  __m128d bb_sum2 = _mm_add_pd(bb_low, bb_high);
  __m128d bb_sum = _mm_hadd_pd(bb_sum2, bb_sum2);
  double norm2_b = _mm_cvtsd_f64(bb_sum);
  return std::fma(norm2_a, norm2_b, -dot_ab * dot_ab);
}

static void CrossProdVecLen2BMAlignAVX2(benchmark::State &state) {

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
      benchmark::DoNotOptimize(CrossProdVecLen2_avx2_aligned(
          &a[i % vec_size].data[0], &b[i % vec_size].data[0]));
    }
  }
}
// You can use a macro or a helper function to apply the same benchmark settings to all tests.
// Example using a macro:


struct alignas(64) AlignedDouble64Aligned {
  double data[8];
  AlignedDouble64Aligned() : data{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0} {}
  double &operator[](size_t index) { return data[index]; }
};

#ifdef AVX512_SUPPORTED
double CrossProdVecLen2_avx512_aligned64(const double *a, const double *b) {
  // Full 512-bit load: assumes a[4..7] = 0.0, and memory is 64-byte aligned
  __m512d va = _mm512_load_pd(a); // load a[0..7]
  __m512d vb = _mm512_load_pd(b); // load b[0..7]

  // dot_ab = dot(a, b)
  __m512d ab = _mm512_mul_pd(va, vb);
  double dot_ab = _mm512_reduce_add_pd(ab);

  // norm2_a = dot(a, a)
  __m512d aa = _mm512_mul_pd(va, va);
  double norm2_a = _mm512_reduce_add_pd(aa);

  // norm2_b = dot(b, b)
  __m512d bb = _mm512_mul_pd(vb, vb);
  double norm2_b = _mm512_reduce_add_pd(bb);

  // return ||a||² * ||b||² - (a · b)²
  return std::fma(norm2_a, norm2_b, -dot_ab * dot_ab); // FMA for precision/perf
}

static void CrossProdVecLen2BMAlignAVX512(benchmark::State &state) {

  // First create an instance of an engine.
  std::random_device rnd_device;
  // Specify the engine and distribution.
  std::mt19937 mersenne_engine{rnd_device()}; // Generates random integers
  std::uniform_real_distribution<double> dist{-1.0,
                                              1.0}; // Range for double values
  auto gen = [&]() { return dist(mersenne_engine); };
  const auto size = state.range(0);
  const size_t vec_size = 1000;
  std::vector<AlignedDouble64Aligned> a(vec_size);
  std::vector<AlignedDouble64Aligned> b(vec_size);
  for (size_t i = 0; i < vec_size; ++i) {
    a[i][0] = gen();
    a[i][1] = gen();
    a[i][2] = gen();
    a[i][3] = gen();
    b[i][0] = gen();
    b[i][1] = gen();
    b[i][2] = gen();
    b[i][3] = gen();
  }
  for (auto _ : state) {
    for (size_t i = 0; i < size; i++) {
      benchmark::DoNotOptimize(CrossProdVecLen2_avx512_aligned64(
          &a[i % vec_size].data[0], &b[i % vec_size].data[0]));
    }
  }
}
#endif

#define APPLY_BENCHMARK_SETTINGS(fn) \
  BENCHMARK(fn)->RangeMultiplier(2)->Range(1 << 1, 1 << 16);

APPLY_BENCHMARK_SETTINGS(CrossProdVecLen1BM)
APPLY_BENCHMARK_SETTINGS(CrossProdVecLen2BM)
APPLY_BENCHMARK_SETTINGS(CrossProdVecLen2BMAlign)
APPLY_BENCHMARK_SETTINGS(CrossProdVecLen2BMAlignAVX2)
#ifdef AVX512_SUPPORTED
APPLY_BENCHMARK_SETTINGS(CrossProdVecLen2BMAlignAVX512)
#endif
BENCHMARK_MAIN();