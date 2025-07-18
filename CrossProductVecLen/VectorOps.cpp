#include "VectorOps.h"

void CrossProd(double const *a, double const *b, double *cross_prod) {
  cross_prod[0] = a[1] * b[2] - a[2] * b[1];
  cross_prod[1] = a[2] * b[0] - a[0] * b[2];
  cross_prod[2] = a[0] * b[1] - a[1] * b[0];
}

double DotProd(double const *a, double const *b) {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

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

  // Final permutation to get the correct order: [res1, res2, res0, res3]
  __m256d result_yzx = _mm256_permute4x64_pd(result, _MM_SHUFFLE(3, 0, 2, 1));

  _mm256_store_pd(cross_prod, result_yzx);
}

double DotProdAVX2(const double *a, const double *b) {
  __m256d va = _mm256_load_pd(a); // [a0, a1, a2, a3]
  __m256d vb = _mm256_load_pd(b); // [b0, b1, b2, b3]

  // Compute dot product
  __m256d mul = _mm256_mul_pd(va, vb);
  alignas(32) double temp[4];
  _mm256_store_pd(temp, mul); // Store intermediate results

  return temp[0] + temp[1] + temp[2];
}

double DotProdFMA(const double *a, const double *b) {
  return std::fma(a[0], b[0], std::fma(a[1], b[1], a[2] * b[2]));
}