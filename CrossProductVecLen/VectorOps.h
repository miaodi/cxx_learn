#pragma once
#include <immintrin.h> // AVX2/FMA
#include <random>

void CrossProd(double const *a, double const *b, double *cross_prod);

void CrossProdAVX2(const double *a, const double *b, double *cross_prod);

inline void CrossProdAVX2Inline(const __m256d &va, const __m256d &vb,
                                __m256d &cross_prod) {

  // Permute: [a1, a2, a0, a3] and [b1, b2, b0, b3]
  __m256d va_yzx = _mm256_permute4x64_pd(va, _MM_SHUFFLE(3, 0, 2, 1));
  __m256d vb_yzx = _mm256_permute4x64_pd(vb, _MM_SHUFFLE(3, 0, 2, 1));

  // Compute cross product using only 3 permutations
  __m256d mul1 = _mm256_mul_pd(va, vb_yzx);
  __m256d mul2 = _mm256_mul_pd(vb, va_yzx);
  __m256d result = _mm256_sub_pd(mul1, mul2);

  // Final permutation to get the correct order: [res1, res2, res0, res3]
  cross_prod = _mm256_permute4x64_pd(result, _MM_SHUFFLE(3, 0, 2, 1));
}

double DotProd(double const *a, double const *b);

double DotProdAVX2(const double *a, const double *b);

inline double DotProdAVX2Inline(const __m256d &va, const __m256d &vb) {
  __m256d mul = _mm256_mul_pd(va, vb);
  alignas(32) double temp[4];
  _mm256_store_pd(temp, mul); // Store intermediate results
  return temp[0] + temp[1] + temp[2];
}

double DotProdFMA(const double *a, const double *b);

template <typename Iterator>
void FillRandom(Iterator begin, Iterator end, double min = -1.0,
                double max = 1.0) {
  std::random_device rnd_device;
  std::mt19937 mersenne_engine{rnd_device()};
  std::uniform_real_distribution<double> dist{min, max};
  auto gen = [&]() { return dist(mersenne_engine); };
  for (auto it = begin; it != end; ++it) {
    (*it)[0] = gen();
    (*it)[1] = gen();
    (*it)[2] = gen();
  }
}

struct alignas(32) AlignedDouble3 {
  double data[3];
  AlignedDouble3() : data{0.0, 0.0, 0.0} {}
  inline double &operator[](size_t index) { return data[index]; }
};

union M256D {
  __m256d m256d;
  double data[4];
  M256D() : m256d(_mm256_setzero_pd()) {}
  M256D(const __m256d &v) : m256d(v) {}
  M256D(const double val) : m256d(_mm256_set1_pd(val)) {}
  M256D(double v0, double v1, double v2)
      : m256d(_mm256_setr_pd(v0, v1, v2, 0.0)) {}
  inline double &operator[](size_t index) { return data[index]; }
  inline const double &operator[](size_t index) const { return data[index]; }
};