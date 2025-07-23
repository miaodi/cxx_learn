#include "func.h"
#include <algorithm>
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <random>
#include "Vector.h"
TEST(MCRand, TwiddleFunctions) {
  std::random_device rd;
  std::mt19937_64 gen(rd());
  std::uniform_int_distribution<std::uint64_t> dist;

  std::uint64_t u = dist(gen);
  std::uint64_t v = dist(gen);

  std::uint64_t result_origin = MCRand::twiddle_origin(u, v);
  std::uint64_t result_new = MCRand::twiddle_new(u, v);

  EXPECT_EQ(result_origin, result_new);
}

#ifdef __AVX2__
TEST(MCRand, TmcRand_vs_TmcRandAVX2) {
  MCRand::tmcRand rand1(12345);
  MCRand::tmcRandAVX2 rand2(12345);

  for (int i = 0; i < 10000; ++i) {
    EXPECT_EQ(rand1.drand(), rand2.drand());
  }
}
#endif // __AVX2__


#ifdef __AVX2__
TEST(MCVector, MCVectorAVX2OpOverload) {
  using namespace misc;
  MCVectorAVX2 vec1(1.0, 2.0, 3.0);

  EXPECT_EQ(vec1[0], 1.0);
  EXPECT_EQ(vec1[1], 2.0);
  EXPECT_EQ(vec1[2], 3.0);

  vec1[0] = 4.0;
  vec1[1] = 5.0;
  vec1[2] = 6.0;

  EXPECT_EQ(vec1[0], 4.0);
  EXPECT_EQ(vec1[1], 5.0);
  EXPECT_EQ(vec1[2], 6.0);
}

TEST(MCVector, MCVectorAVX2Sum) {
  using namespace misc;
  MCVectorAVX2 vec1(1.0, 2.0, 3.0);
  MCVectorAVX2 vec2(4.0, 5.0, 6.0);

  MCVectorAVX2 vec3 = vec1 + vec2;
  EXPECT_EQ(vec3[0], 5.0);
  EXPECT_EQ(vec3[1], 7.0);
  EXPECT_EQ(vec3[2], 9.0);
}

TEST(MCVector, MCVectorAVX2Constructor) {
  using namespace misc;
  MCVectorAVX2 vec1(1.0, 2.0, 3.0);
  MCVectorAVX2 vec2(vec1);
  EXPECT_EQ(vec2[0], 1.0);
  EXPECT_EQ(vec2[1], 2.0);
  EXPECT_EQ(vec2[2], 3.0);

  MCVectorAVX2 vec3;
  EXPECT_EQ(vec3[0], 0.0);
  EXPECT_EQ(vec3[1], 0.0);
  EXPECT_EQ(vec3[2], 0.0);

  vec3 = vec1;
  EXPECT_EQ(vec3[0], 1.0);
  EXPECT_EQ(vec3[1], 2.0);
  EXPECT_EQ(vec3[2], 3.0);
}



bool is_aligned_modulo(void *ptr, size_t alignment) {
  return (reinterpret_cast<uintptr_t>(ptr) % alignment == 0);
}

TEST(MCVector, MCVectorAVX2Alignment) {
  using namespace misc;
  MCVectorAVX2 vec1(1.0, 2.0, 3.0);
  EXPECT_TRUE(is_aligned_modulo(&vec1, 32));
}
#endif // __AVX2__
