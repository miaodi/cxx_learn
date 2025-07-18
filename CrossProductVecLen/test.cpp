#include "VectorOps.h"
#include <algorithm>
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <immintrin.h> // AVX2/FMA

bool is_aligned_modulo(void *ptr, size_t alignment) {
  return (reinterpret_cast<uintptr_t>(ptr) % alignment == 0);
}

TEST(AlignedArray, Size) {
  AlignedDouble3 a;
  AlignedDouble3 b;
  EXPECT_EQ(sizeof(a), sizeof(b));
  EXPECT_EQ(sizeof(a.data), sizeof(b.data));
  EXPECT_EQ(3 * sizeof(double), sizeof(a.data));
  EXPECT_EQ(32, sizeof(AlignedDouble3));
  AlignedDouble3 c[3];
  EXPECT_EQ(sizeof(c), 3 * sizeof(AlignedDouble3));
  EXPECT_TRUE(is_aligned_modulo(c, 32));
}

TEST(Intrinsic, AVX2) {
  alignas(32) double a[4] = {0.0, 1.0, 2.0, 3.0};
  __m256d va = _mm256_load_pd(a); // a[0], a[1], a[2], a[3]
  __m256d perm_va = _mm256_permute4x64_pd(va, _MM_SHUFFLE(0, 3, 2, 1));
  double result[4];
  _mm256_store_pd(result, perm_va);
  EXPECT_EQ(result[0], 1.0);
  EXPECT_EQ(result[1], 2.0);
  EXPECT_EQ(result[2], 3.0);
  EXPECT_EQ(result[3], 0.0);
}

TEST(CrossProduct, AVX2) {
  alignas(32) double a[4] = {1.0, 2.0, 3.0, 0.0};
  alignas(32) double b[4] = {4.0, 5.0, 6.0, 0.0};
  alignas(32) double cross_prod1[4];
  alignas(32) double cross_prod2[4];

  CrossProd(a, b, cross_prod1);
  CrossProdAVX2(a, b, cross_prod2);
  EXPECT_EQ(cross_prod1[0], cross_prod2[0]);
  EXPECT_EQ(cross_prod1[1], cross_prod2[1]);
  EXPECT_EQ(cross_prod1[2], cross_prod2[2]);
}

TEST(DotProduct, AVX2_and_FMA) {
  alignas(32) double a[4] = {1.0, 2.0, 3.0, 0.0};
  alignas(32) double b[4] = {4.0, 5.0, 6.0, 0.0};

  double dot1 = DotProd(a, b);
  double dot2 = DotProdAVX2(a, b);
  double dot3 = DotProdFMA(a, b);

  EXPECT_DOUBLE_EQ(dot1, dot2);
  EXPECT_DOUBLE_EQ(dot1, dot3);
}

TEST(M256D, BasicOperations) {
  M256D v1(1.0);
  M256D v2(2.0);
  EXPECT_EQ(v1[0], 1.0);
  EXPECT_EQ(v1[1], 1.0);
  EXPECT_EQ(v1[2], 1.0);
  EXPECT_EQ(v2[0], 2.0);
  EXPECT_EQ(v2[1], 2.0);
  EXPECT_EQ(v2[2], 2.0);

  M256D v3(1.0, 2.0, 3.0);
  EXPECT_EQ(v3[0], 1.0);
  EXPECT_EQ(v3[1], 2.0);
  EXPECT_EQ(v3[2], 3.0);
  EXPECT_EQ(v3[3], 0.0); // Last element should be zero
}

TEST(M256D, DotProduct) {
  M256D v1(1.0, 2.0, 3.0);
  M256D v2(4.0, 5.0, 6.0);
  double result = DotProdAVX2Inline(v1.m256d, v2.m256d);
  EXPECT_DOUBLE_EQ(result, 1.0 * 4.0 + 2.0 * 5.0 + 3.0 * 6.0); // 32.0
}