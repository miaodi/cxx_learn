#include "reduction_sum.h"

#include <gtest/gtest.h>

#include <cmath>
#include <numeric>
#include <vector>

namespace {

float cpu_sum(const std::vector<float> &values) {
  return std::accumulate(values.begin(), values.end(), 0.0f);
}

void expect_sums_near(const std::vector<float> &values, float tolerance) {
  const float expected = cpu_sum(values);
  const float stride_actual =
      pmpp::reduction::simple_stride_sum(values.data(), values.size());
  const float sequential_actual =
      pmpp::reduction::sequential_addressing_sum(values.data(), values.size());
  const float coarsened_shared_actual =
      pmpp::reduction::coarsened_shared_memory_sum(values.data(),
                                                   values.size());
  const float coarsened_shared_optimized_actual =
      pmpp::reduction::coarsened_shared_memory_sum_optimized(values.data(),
                                                             values.size());
  const float thrust_actual =
      pmpp::reduction::thrust_reference_sum(values.data(), values.size());

  EXPECT_NEAR(expected, stride_actual, tolerance)
      << "size = " << values.size() << ", expected = " << expected
      << ", actual = " << stride_actual;
  EXPECT_NEAR(expected, sequential_actual, tolerance)
      << "size = " << values.size() << ", expected = " << expected
      << ", actual = " << sequential_actual;
  EXPECT_NEAR(expected, coarsened_shared_actual, tolerance)
      << "size = " << values.size() << ", expected = " << expected
      << ", actual = " << coarsened_shared_actual;
  EXPECT_NEAR(expected, coarsened_shared_optimized_actual, tolerance)
      << "size = " << values.size() << ", expected = " << expected
      << ", actual = " << coarsened_shared_optimized_actual;
  EXPECT_NEAR(expected, thrust_actual, tolerance)
      << "size = " << values.size() << ", expected = " << expected
      << ", actual = " << thrust_actual;
}

} // namespace

TEST(SimpleStrideSumTest, EmptyInputReturnsZero) {
  EXPECT_FLOAT_EQ(0.0f, pmpp::reduction::simple_stride_sum(nullptr, 0));
  EXPECT_FLOAT_EQ(0.0f,
                  pmpp::reduction::sequential_addressing_sum(nullptr, 0));
  EXPECT_FLOAT_EQ(0.0f,
                  pmpp::reduction::coarsened_shared_memory_sum(nullptr, 0));
  EXPECT_FLOAT_EQ(
      0.0f, pmpp::reduction::coarsened_shared_memory_sum_optimized(nullptr, 0));
  EXPECT_FLOAT_EQ(0.0f, pmpp::reduction::thrust_reference_sum(nullptr, 0));
}

TEST(SimpleStrideSumTest, HandlesSmallNonPowerOfTwoInput) {
  expect_sums_near({1.0f, 2.0f, 3.0f, 4.0f, 5.0f}, 1e-6f);
}

TEST(SimpleStrideSumTest, HandlesOneBlockInput) {
  std::vector<float> values(512);
  for (std::size_t i = 0; i < values.size(); ++i) {
    values[i] = static_cast<float>(i % 7) - 3.0f;
  }
  expect_sums_near(values, 1e-5f);
}

TEST(SimpleStrideSumTest, HandlesMultipleBlocks) {
  std::vector<float> values(12345);
  for (std::size_t i = 0; i < values.size(); ++i) {
    values[i] = static_cast<float>((i * 17) % 23) * 0.25f;
  }
  expect_sums_near(values, 1e-2f);
}

TEST(SimpleStrideSumTest, HandlesAlternatingCancellation) {
  std::vector<float> values(10000);
  for (std::size_t i = 0; i < values.size(); ++i) {
    values[i] = (i % 2 == 0) ? 1.0f : -1.0f;
  }
  expect_sums_near(values, 1e-6f);
}
