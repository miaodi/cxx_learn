#include "scan.h"

#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <numeric>
#include <vector>

namespace {

#define CHECK_CUDA(status)                                                     \
  ASSERT_EQ((status), cudaSuccess) << cudaGetErrorString((status))

#define SKIP_IF_CUDA_RUNTIME_UNAVAILABLE()                                     \
  do {                                                                         \
    const cudaError_t status = cudaFree(nullptr);                              \
    if (status == cudaErrorInsufficientDriver || status == cudaErrorNoDevice) {\
      GTEST_SKIP() << cudaGetErrorString(status);                              \
    }                                                                          \
    ASSERT_EQ(status, cudaSuccess) << cudaGetErrorString(status);              \
  } while (false)

using ScanFunction = cudaError_t (*)(const float *, float *, int, cudaStream_t);

void expect_placeholder_contract(ScanFunction function) {
  auto *dummy = reinterpret_cast<float *>(0x1);

  EXPECT_EQ(function(nullptr, nullptr, 0, nullptr), cudaSuccess);
  EXPECT_EQ(function(dummy, dummy, -1, nullptr), cudaErrorInvalidValue);
  EXPECT_EQ(function(nullptr, dummy, 4, nullptr), cudaErrorInvalidValue);
  EXPECT_EQ(function(dummy, nullptr, 4, nullptr), cudaErrorInvalidValue);
  EXPECT_EQ(function(dummy, dummy, 4, nullptr), cudaErrorNotSupported);
}

std::vector<float> inclusive_scan_reference(const std::vector<float> &input) {
  std::vector<float> output(input.size());
  std::partial_sum(input.begin(), input.end(), output.begin());
  return output;
}

void expect_brent_kung_matches_reference(const std::vector<float> &input) {
  SKIP_IF_CUDA_RUNTIME_UNAVAILABLE();

  const std::vector<float> expected = inclusive_scan_reference(input);
  std::vector<float> actual(input.size(), 0.0f);

  float *d_input = nullptr;
  float *d_output = nullptr;
  const auto bytes = input.size() * sizeof(float);

  CHECK_CUDA(cudaMalloc(&d_input, bytes));
  CHECK_CUDA(cudaMalloc(&d_output, bytes));
  CHECK_CUDA(cudaMemcpy(d_input, input.data(), bytes, cudaMemcpyHostToDevice));

  CHECK_CUDA(pmpp::scan::brent_kung_inclusive_scan(
      d_input, d_output, static_cast<int>(input.size())));
  CHECK_CUDA(cudaDeviceSynchronize());
  CHECK_CUDA(
      cudaMemcpy(actual.data(), d_output, bytes, cudaMemcpyDeviceToHost));

  for (std::size_t i = 0; i < input.size(); ++i) {
    EXPECT_FLOAT_EQ(expected[i], actual[i]) << "index = " << i;
  }

  CHECK_CUDA(cudaFree(d_input));
  CHECK_CUDA(cudaFree(d_output));
}

} // namespace

TEST(KoggeStoneInclusiveScanTest, HasPlaceholderContract) {
  expect_placeholder_contract(pmpp::scan::kogge_stone_inclusive_scan);
}

TEST(BrentKungInclusiveScanTest, ValidatesArgumentsAndSupportedSize) {
  auto *dummy = reinterpret_cast<float *>(0x1);

  EXPECT_EQ(pmpp::scan::brent_kung_inclusive_scan(nullptr, nullptr, 0),
            cudaSuccess);
  EXPECT_EQ(pmpp::scan::brent_kung_inclusive_scan(dummy, dummy, -1),
            cudaErrorInvalidValue);
  EXPECT_EQ(pmpp::scan::brent_kung_inclusive_scan(nullptr, dummy, 4),
            cudaErrorInvalidValue);
  EXPECT_EQ(pmpp::scan::brent_kung_inclusive_scan(dummy, nullptr, 4),
            cudaErrorInvalidValue);
  EXPECT_EQ(pmpp::scan::brent_kung_inclusive_scan(
                dummy, dummy, pmpp::scan::kBrentKungItemsPerBlock + 1),
            cudaErrorNotSupported);
}

TEST(BrentKungInclusiveScanTest, ScansSmallNonPowerOfTwoInput) {
  expect_brent_kung_matches_reference({3.0f, 1.0f, 7.0f, 0.0f, 4.0f, 1.0f});
}

TEST(BrentKungInclusiveScanTest, ScansFullSingleBlockTile) {
  std::vector<float> input(pmpp::scan::kBrentKungItemsPerBlock);
  for (std::size_t i = 0; i < input.size(); ++i) {
    input[i] = static_cast<float>(static_cast<int>(i % 5) - 2);
  }

  expect_brent_kung_matches_reference(input);
}

TEST(CoarsenedBrentKungInclusiveScanTest, HasPlaceholderContract) {
  expect_placeholder_contract(pmpp::scan::coarsened_brent_kung_inclusive_scan);
}

TEST(HierarchicalInclusiveScanTest, HasPlaceholderContract) {
  expect_placeholder_contract(pmpp::scan::hierarchical_inclusive_scan);
}

TEST(CubInclusiveScanReferenceTest, HasPlaceholderContract) {
  expect_placeholder_contract(pmpp::scan::cub_inclusive_scan_reference);
}
