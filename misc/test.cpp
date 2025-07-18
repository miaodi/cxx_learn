#include "func.h"
#include <algorithm>
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <random>
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