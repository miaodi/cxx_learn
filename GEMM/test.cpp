
#include "gemm.hpp"
#include <cmath>
#include <gtest/gtest.h>
#include <random>
#include <vector>

using namespace gemm;

template <typename T>
void fill_random(std::vector<T> &v, T min = T(-1), T max = T(1)) {
  std::mt19937 gen(42);
  std::uniform_real_distribution<T> dist(min, max);
  for (auto &x : v)
    x = dist(gen);
}

template <typename T>
bool all_close(const std::vector<T> &a, const std::vector<T> &b,
               T tol = T(1e-4)) {
  if (a.size() != b.size())
    return false;
  for (size_t i = 0; i < a.size(); ++i) {
    if (std::abs(a[i] - b[i]) > tol)
      return false;
  }
  return true;
}

struct MatSize {
  int M, N, K;
};

class GemmTest : public ::testing::TestWithParam<MatSize> {
protected:
  using T = float;
  void run_all_algorithms(int M, int N, int K) {
    std::vector<T> A(M * K), B(K * N), C_ref(M * N);
    fill_random(A);
    fill_random(B);

    MatMatMul<T>(A.data(), B.data(), C_ref.data(), M, N, K);

    // // MatMatTransMul
    // std::vector<T> C_trans(M * N, 0);
    // MatMatTransMul<T>(A.data(), B.data(), C_trans.data(), M, N, K);
    // EXPECT_TRUE(all_close(C_ref, C_trans));

    // TiledMatMatMul
    std::vector<T> C_tiled(M * N, 0);
    TiledMatMatMul<T>(A.data(), B.data(), C_tiled.data(), M, N, K);
    EXPECT_TRUE(all_close(C_ref, C_tiled));

    // // TiledMatMatTransMul
    // std::vector<T> C_tiled_trans(M * N, 0);
    // TiledMatMatTransMul<T>(A.data(), B.data(), C_tiled_trans.data(), M, N,
    // K); EXPECT_TRUE(all_close(C_ref, C_tiled_trans));

    // TiledMatMatMulInternalTrans
    std::vector<T> C_internal_trans(M * N, 0);
    TiledMatMatMulInternalTrans<T>(A.data(), B.data(), C_internal_trans.data(),
                                   M, N, K);
    EXPECT_TRUE(all_close(C_ref, C_internal_trans));

    // TiledMatMatMulInternalTiledPadded
    std::vector<T> C_internal_tiled(M * N, 0);
    TiledMatMatMulInternalTiledPadded<T>(A.data(), B.data(),
                                         C_internal_tiled.data(), M, N, K);
    EXPECT_TRUE(all_close(C_ref, C_internal_tiled));
  }
};

TEST_P(GemmTest, AllAlgorithmsRectangular) {
  MatSize sz = GetParam();
  run_all_algorithms(sz.M, sz.N, sz.K);
}

INSTANTIATE_TEST_SUITE_P(
    RectangularSizes, GemmTest,
    ::testing::Values(MatSize{16, 16, 16}, MatSize{24, 17, 19},
                      MatSize{15, 15, 15}, // not divisible by 4
                      MatSize{17, 19, 23}, // not divisible by 4
                      MatSize{9, 7, 5},    // not divisible by 4
                      MatSize{5, 9, 7},    // not divisible by 4
                      MatSize{6, 10, 14},  // not divisible by 4
                      MatSize{16, 8, 8}, MatSize{8, 16, 8}, MatSize{8, 8, 16},
                      MatSize{32, 32, 32}, MatSize{64, 32, 16},
                      MatSize{16, 64, 32}, MatSize{32, 16, 64},
                      MatSize{7, 13, 11}, MatSize{1, 8, 8}, MatSize{8, 1, 8},
                      MatSize{8, 8, 1}));

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
