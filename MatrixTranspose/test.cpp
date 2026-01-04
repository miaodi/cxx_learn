#include "transpose.hpp"
#include <gtest/gtest.h>
#include <vector>
#include <random>

using namespace matrix_transpose;

class TransposeTest : public ::testing::Test {
protected:
  void SetUp() override {
    // Initialize random data
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-100.0f, 100.0f);
    
    for (auto &val : small_input) val = dist(rng);
    for (auto &val : medium_input) val = dist(rng);
    for (auto &val : rect_tall_input) val = dist(rng);
    for (auto &val : rect_wide_input) val = dist(rng);
  }
  
  // Test data
  static constexpr int SMALL = 4;
  static constexpr int MEDIUM = 64;
  static constexpr int RECT_M = 128;
  static constexpr int RECT_N = 32;
  
  std::vector<float> small_input = std::vector<float>(SMALL * SMALL);
  std::vector<float> medium_input = std::vector<float>(MEDIUM * MEDIUM);
  std::vector<float> rect_tall_input = std::vector<float>(RECT_M * RECT_N);
  std::vector<float> rect_wide_input = std::vector<float>(RECT_N * RECT_M);
};

// ============================================================================
// Correctness Tests
// ============================================================================

TEST_F(TransposeTest, NaiveTranspose_Small) {
  std::vector<float> output(SMALL * SMALL);
  NaiveTranspose(small_input.data(), output.data(), SMALL, SMALL);
  EXPECT_TRUE(VerifyTranspose(small_input.data(), output.data(), SMALL, SMALL));
}

TEST_F(TransposeTest, NaiveTranspose_Medium) {
  std::vector<float> output(MEDIUM * MEDIUM);
  NaiveTranspose(medium_input.data(), output.data(), MEDIUM, MEDIUM);
  EXPECT_TRUE(VerifyTranspose(medium_input.data(), output.data(), MEDIUM, MEDIUM));
}

TEST_F(TransposeTest, TiledTranspose16_Medium) {
  std::vector<float> output(MEDIUM * MEDIUM);
  TiledTranspose<float, 16>(medium_input.data(), output.data(), MEDIUM, MEDIUM);
  EXPECT_TRUE(VerifyTranspose(medium_input.data(), output.data(), MEDIUM, MEDIUM));
}

TEST_F(TransposeTest, TiledTranspose32_Medium) {
  std::vector<float> output(MEDIUM * MEDIUM);
  TiledTranspose<float, 32>(medium_input.data(), output.data(), MEDIUM, MEDIUM);
  EXPECT_TRUE(VerifyTranspose(medium_input.data(), output.data(), MEDIUM, MEDIUM));
}

TEST_F(TransposeTest, TiledTranspose64_Medium) {
  std::vector<float> output(MEDIUM * MEDIUM);
  TiledTranspose<float, 64>(medium_input.data(), output.data(), MEDIUM, MEDIUM);
  EXPECT_TRUE(VerifyTranspose(medium_input.data(), output.data(), MEDIUM, MEDIUM));
}

TEST_F(TransposeTest, CacheOblivious_Medium) {
  std::vector<float> output(MEDIUM * MEDIUM);
  CacheObliviousTranspose(medium_input.data(), output.data(), MEDIUM, MEDIUM);
  EXPECT_TRUE(VerifyTranspose(medium_input.data(), output.data(), MEDIUM, MEDIUM));
}

TEST_F(TransposeTest, RectangularTall_Naive) {
  std::vector<float> output(RECT_M * RECT_N);
  NaiveTranspose(rect_tall_input.data(), output.data(), RECT_M, RECT_N);
  EXPECT_TRUE(VerifyTranspose(rect_tall_input.data(), output.data(), RECT_M, RECT_N));
}

TEST_F(TransposeTest, RectangularTall_Tiled32) {
  std::vector<float> output(RECT_M * RECT_N);
  TiledTranspose<float, 32>(rect_tall_input.data(), output.data(), RECT_M, RECT_N);
  EXPECT_TRUE(VerifyTranspose(rect_tall_input.data(), output.data(), RECT_M, RECT_N));
}

TEST_F(TransposeTest, RectangularWide_Naive) {
  std::vector<float> output(RECT_N * RECT_M);
  NaiveTranspose(rect_wide_input.data(), output.data(), RECT_N, RECT_M);
  EXPECT_TRUE(VerifyTranspose(rect_wide_input.data(), output.data(), RECT_N, RECT_M));
}

TEST_F(TransposeTest, RectangularWide_Tiled32) {
  std::vector<float> output(RECT_N * RECT_M);
  TiledTranspose<float, 32>(rect_wide_input.data(), output.data(), RECT_N, RECT_M);
  EXPECT_TRUE(VerifyTranspose(rect_wide_input.data(), output.data(), RECT_N, RECT_M));
}

// ============================================================================
// In-Place Transpose Tests
// ============================================================================

// ============================================================================
// Edge Cases
// ============================================================================

TEST_F(TransposeTest, SingleElement) {
  float input[1] = {42.0f};
  float output[1] = {0.0f};
  
  NaiveTranspose(input, output, 1, 1);
  EXPECT_FLOAT_EQ(output[0], 42.0f);
}

TEST_F(TransposeTest, SingleRow) {
  std::vector<float> input = {1, 2, 3, 4, 5};
  std::vector<float> output(5);
  
  NaiveTranspose(input.data(), output.data(), 1, 5);
  
  for (int i = 0; i < 5; ++i) {
    EXPECT_FLOAT_EQ(output[i], input[i]);
  }
}

TEST_F(TransposeTest, SingleColumn) {
  std::vector<float> input = {1, 2, 3, 4, 5};
  std::vector<float> output(5);
  
  NaiveTranspose(input.data(), output.data(), 5, 1);
  
  for (int i = 0; i < 5; ++i) {
    EXPECT_FLOAT_EQ(output[i], input[i]);
  }
}

TEST_F(TransposeTest, NonMultipleOfTileSize) {
  // Test with dimensions not divisible by tile size
  constexpr int M = 65; // Not divisible by 16, 32, or 64
  constexpr int N = 97;
  
  std::vector<float> input(M * N);
  std::mt19937 rng(42);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  for (auto &val : input) val = dist(rng);
  
  std::vector<float> output(M * N);
  TiledTranspose<float, 32>(input.data(), output.data(), M, N);
  
  EXPECT_TRUE(VerifyTranspose(input.data(), output.data(), M, N));
}
