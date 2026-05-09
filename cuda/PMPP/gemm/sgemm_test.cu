#include "sgemm.h"

#include <algorithm>
#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <random>
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

std::vector<float> random_matrix(int rows, int cols, int ld) {
  std::vector<float> values(rows * ld, -99.0f);
  std::mt19937 rng(2026);
  std::uniform_real_distribution<float> dist(-2.0f, 2.0f);

  for (int row = 0; row < rows; ++row) {
    for (int col = 0; col < cols; ++col) {
      values[row * ld + col] = dist(rng);
    }
  }
  return values;
}

void reference_sgemm(int m, int n, int k, float alpha,
                     const std::vector<float> &A, int lda,
                     const std::vector<float> &B, int ldb, float beta,
                     std::vector<float> &C, int ldc) {
  for (int row = 0; row < m; ++row) {
    for (int col = 0; col < n; ++col) {
      float sum = 0.0f;
      for (int kk = 0; kk < k; ++kk) {
        sum += A[row * lda + kk] * B[kk * ldb + col];
      }
      C[row * ldc + col] = alpha * sum + beta * C[row * ldc + col];
    }
  }
}

void expect_active_matrix_near(const std::vector<float> &expected,
                               const std::vector<float> &actual, int rows,
                               int cols, int ld, float tolerance) {
  for (int row = 0; row < rows; ++row) {
    for (int col = 0; col < cols; ++col) {
      const int index = row * ld + col;
      EXPECT_NEAR(expected[index], actual[index], tolerance)
          << "row=" << row << " col=" << col;
    }
  }
}

TEST(SgemmIjkTest, ComputesAlphaABPlusBetaCWithLeadingDimensions) {
  SKIP_IF_CUDA_RUNTIME_UNAVAILABLE();

  constexpr int m = 7;
  constexpr int n = 5;
  constexpr int k = 3;
  constexpr int lda = 6;
  constexpr int ldb = 8;
  constexpr int ldc = 9;

  const float alpha = 1.25f;
  const float beta = -0.5f;

  const std::vector<float> A = random_matrix(m, k, lda);
  const std::vector<float> B = random_matrix(k, n, ldb);
  std::vector<float> C = random_matrix(m, n, ldc);
  std::vector<float> expected = C;

  reference_sgemm(m, n, k, alpha, A, lda, B, ldb, beta, expected, ldc);

  float *d_A = nullptr;
  float *d_B = nullptr;
  float *d_C = nullptr;
  CHECK_CUDA(cudaMalloc(&d_A, A.size() * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&d_B, B.size() * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&d_C, C.size() * sizeof(float)));

  CHECK_CUDA(cudaMemcpy(d_A, A.data(), A.size() * sizeof(float),
                        cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_B, B.data(), B.size() * sizeof(float),
                        cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_C, C.data(), C.size() * sizeof(float),
                        cudaMemcpyHostToDevice));

  CHECK_CUDA(pmpp::gemm::sgemm_ijk(m, n, k, alpha, d_A, lda, d_B, ldb, beta,
                                   d_C, ldc));
  CHECK_CUDA(cudaDeviceSynchronize());

  CHECK_CUDA(cudaMemcpy(C.data(), d_C, C.size() * sizeof(float),
                        cudaMemcpyDeviceToHost));
  expect_active_matrix_near(expected, C, m, n, ldc, 1e-4f);

  CHECK_CUDA(cudaFree(d_A));
  CHECK_CUDA(cudaFree(d_B));
  CHECK_CUDA(cudaFree(d_C));
}

TEST(SgemmTiled16Test, ComputesAlphaABPlusBetaCWithLeadingDimensions) {
  SKIP_IF_CUDA_RUNTIME_UNAVAILABLE();

  constexpr int m = 19;
  constexpr int n = 21;
  constexpr int k = 17;
  constexpr int lda = 24;
  constexpr int ldb = 25;
  constexpr int ldc = 27;

  const float alpha = -0.75f;
  const float beta = 0.5f;

  const std::vector<float> A = random_matrix(m, k, lda);
  const std::vector<float> B = random_matrix(k, n, ldb);
  std::vector<float> C = random_matrix(m, n, ldc);
  std::vector<float> expected = C;

  reference_sgemm(m, n, k, alpha, A, lda, B, ldb, beta, expected, ldc);

  float *d_A = nullptr;
  float *d_B = nullptr;
  float *d_C = nullptr;
  CHECK_CUDA(cudaMalloc(&d_A, A.size() * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&d_B, B.size() * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&d_C, C.size() * sizeof(float)));

  CHECK_CUDA(cudaMemcpy(d_A, A.data(), A.size() * sizeof(float),
                        cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_B, B.data(), B.size() * sizeof(float),
                        cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_C, C.data(), C.size() * sizeof(float),
                        cudaMemcpyHostToDevice));

  CHECK_CUDA(pmpp::gemm::sgemm_tiled_16(m, n, k, alpha, d_A, lda, d_B, ldb,
                                        beta, d_C, ldc));
  CHECK_CUDA(cudaDeviceSynchronize());

  CHECK_CUDA(cudaMemcpy(C.data(), d_C, C.size() * sizeof(float),
                        cudaMemcpyDeviceToHost));
  expect_active_matrix_near(expected, C, m, n, ldc, 1e-4f);

  CHECK_CUDA(cudaFree(d_A));
  CHECK_CUDA(cudaFree(d_B));
  CHECK_CUDA(cudaFree(d_C));
}

TEST(SgemmTiled16_2x2Test, ComputesAlphaABPlusBetaCWithLeadingDimensions) {
  SKIP_IF_CUDA_RUNTIME_UNAVAILABLE();

  constexpr int m = 33;
  constexpr int n = 35;
  constexpr int k = 19;
  constexpr int lda = 23;
  constexpr int ldb = 41;
  constexpr int ldc = 39;

  const float alpha = 0.875f;
  const float beta = -0.25f;

  const std::vector<float> A = random_matrix(m, k, lda);
  const std::vector<float> B = random_matrix(k, n, ldb);
  std::vector<float> C = random_matrix(m, n, ldc);
  std::vector<float> expected = C;

  reference_sgemm(m, n, k, alpha, A, lda, B, ldb, beta, expected, ldc);

  float *d_A = nullptr;
  float *d_B = nullptr;
  float *d_C = nullptr;
  CHECK_CUDA(cudaMalloc(&d_A, A.size() * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&d_B, B.size() * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&d_C, C.size() * sizeof(float)));

  CHECK_CUDA(cudaMemcpy(d_A, A.data(), A.size() * sizeof(float),
                        cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_B, B.data(), B.size() * sizeof(float),
                        cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_C, C.data(), C.size() * sizeof(float),
                        cudaMemcpyHostToDevice));

  CHECK_CUDA(pmpp::gemm::sgemm_tiled_16_2x2(m, n, k, alpha, d_A, lda, d_B,
                                            ldb, beta, d_C, ldc));
  CHECK_CUDA(cudaDeviceSynchronize());

  CHECK_CUDA(cudaMemcpy(C.data(), d_C, C.size() * sizeof(float),
                        cudaMemcpyDeviceToHost));
  expect_active_matrix_near(expected, C, m, n, ldc, 1e-4f);

  CHECK_CUDA(cudaFree(d_A));
  CHECK_CUDA(cudaFree(d_B));
  CHECK_CUDA(cudaFree(d_C));
}

TEST(SgemmTiled16_2x2CoalescedTest,
     ComputesAlphaABPlusBetaCWithLeadingDimensions) {
  SKIP_IF_CUDA_RUNTIME_UNAVAILABLE();

  constexpr int m = 33;
  constexpr int n = 35;
  constexpr int k = 19;
  constexpr int lda = 23;
  constexpr int ldb = 41;
  constexpr int ldc = 39;

  const float alpha = 0.875f;
  const float beta = -0.25f;

  const std::vector<float> A = random_matrix(m, k, lda);
  const std::vector<float> B = random_matrix(k, n, ldb);
  std::vector<float> C = random_matrix(m, n, ldc);
  std::vector<float> expected = C;

  reference_sgemm(m, n, k, alpha, A, lda, B, ldb, beta, expected, ldc);

  float *d_A = nullptr;
  float *d_B = nullptr;
  float *d_C = nullptr;
  CHECK_CUDA(cudaMalloc(&d_A, A.size() * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&d_B, B.size() * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&d_C, C.size() * sizeof(float)));

  CHECK_CUDA(cudaMemcpy(d_A, A.data(), A.size() * sizeof(float),
                        cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_B, B.data(), B.size() * sizeof(float),
                        cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_C, C.data(), C.size() * sizeof(float),
                        cudaMemcpyHostToDevice));

  CHECK_CUDA(pmpp::gemm::sgemm_tiled_16_2x2_coalesced(
      m, n, k, alpha, d_A, lda, d_B, ldb, beta, d_C, ldc));
  CHECK_CUDA(cudaDeviceSynchronize());

  CHECK_CUDA(cudaMemcpy(C.data(), d_C, C.size() * sizeof(float),
                        cudaMemcpyDeviceToHost));
  expect_active_matrix_near(expected, C, m, n, ldc, 1e-4f);

  CHECK_CUDA(cudaFree(d_A));
  CHECK_CUDA(cudaFree(d_B));
  CHECK_CUDA(cudaFree(d_C));
}

TEST(SgemmTiled16_2x2K32CoalescedTest,
     ComputesAlphaABPlusBetaCWithLeadingDimensions) {
  SKIP_IF_CUDA_RUNTIME_UNAVAILABLE();

  constexpr int m = 33;
  constexpr int n = 35;
  constexpr int k = 19;
  constexpr int lda = 23;
  constexpr int ldb = 41;
  constexpr int ldc = 39;

  const float alpha = 0.875f;
  const float beta = -0.25f;

  const std::vector<float> A = random_matrix(m, k, lda);
  const std::vector<float> B = random_matrix(k, n, ldb);
  std::vector<float> C = random_matrix(m, n, ldc);
  std::vector<float> expected = C;

  reference_sgemm(m, n, k, alpha, A, lda, B, ldb, beta, expected, ldc);

  float *d_A = nullptr;
  float *d_B = nullptr;
  float *d_C = nullptr;
  CHECK_CUDA(cudaMalloc(&d_A, A.size() * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&d_B, B.size() * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&d_C, C.size() * sizeof(float)));

  CHECK_CUDA(cudaMemcpy(d_A, A.data(), A.size() * sizeof(float),
                        cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_B, B.data(), B.size() * sizeof(float),
                        cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_C, C.data(), C.size() * sizeof(float),
                        cudaMemcpyHostToDevice));

  CHECK_CUDA(pmpp::gemm::sgemm_tiled_16_2x2_k32_coalesced(
      m, n, k, alpha, d_A, lda, d_B, ldb, beta, d_C, ldc));
  CHECK_CUDA(cudaDeviceSynchronize());

  CHECK_CUDA(cudaMemcpy(C.data(), d_C, C.size() * sizeof(float),
                        cudaMemcpyDeviceToHost));
  expect_active_matrix_near(expected, C, m, n, ldc, 1e-4f);

  CHECK_CUDA(cudaFree(d_A));
  CHECK_CUDA(cudaFree(d_B));
  CHECK_CUDA(cudaFree(d_C));
}

TEST(SgemmIjkTest, SupportsZeroKByScalingC) {
  SKIP_IF_CUDA_RUNTIME_UNAVAILABLE();

  constexpr int m = 3;
  constexpr int n = 4;
  constexpr int k = 0;
  constexpr int ldc = n;

  std::vector<float> C = random_matrix(m, n, ldc);
  std::vector<float> expected = C;
  reference_sgemm(m, n, k, 2.0f, {}, 0, {}, 0, 0.25f, expected, ldc);

  float *d_C = nullptr;
  CHECK_CUDA(cudaMalloc(&d_C, C.size() * sizeof(float)));
  CHECK_CUDA(cudaMemcpy(d_C, C.data(), C.size() * sizeof(float),
                        cudaMemcpyHostToDevice));

  CHECK_CUDA(pmpp::gemm::sgemm_ijk(m, n, k, 2.0f, nullptr, 0, nullptr, 0,
                                   0.25f, d_C, ldc));
  CHECK_CUDA(cudaDeviceSynchronize());

  CHECK_CUDA(cudaMemcpy(C.data(), d_C, C.size() * sizeof(float),
                        cudaMemcpyDeviceToHost));
  expect_active_matrix_near(expected, C, m, n, ldc, 1e-6f);

  CHECK_CUDA(cudaFree(d_C));
}

TEST(SgemmTiled16Test, SupportsZeroKByScalingC) {
  SKIP_IF_CUDA_RUNTIME_UNAVAILABLE();

  constexpr int m = 5;
  constexpr int n = 7;
  constexpr int k = 0;
  constexpr int ldc = 10;

  std::vector<float> C = random_matrix(m, n, ldc);
  std::vector<float> expected = C;
  reference_sgemm(m, n, k, 2.0f, {}, 0, {}, 0, -0.25f, expected, ldc);

  float *d_C = nullptr;
  CHECK_CUDA(cudaMalloc(&d_C, C.size() * sizeof(float)));
  CHECK_CUDA(cudaMemcpy(d_C, C.data(), C.size() * sizeof(float),
                        cudaMemcpyHostToDevice));

  CHECK_CUDA(pmpp::gemm::sgemm_tiled_16(m, n, k, 2.0f, nullptr, 0, nullptr, 0,
                                        -0.25f, d_C, ldc));
  CHECK_CUDA(cudaDeviceSynchronize());

  CHECK_CUDA(cudaMemcpy(C.data(), d_C, C.size() * sizeof(float),
                        cudaMemcpyDeviceToHost));
  expect_active_matrix_near(expected, C, m, n, ldc, 1e-6f);

  CHECK_CUDA(cudaFree(d_C));
}

TEST(SgemmTiled16_2x2Test, SupportsZeroKByScalingC) {
  SKIP_IF_CUDA_RUNTIME_UNAVAILABLE();

  constexpr int m = 33;
  constexpr int n = 35;
  constexpr int k = 0;
  constexpr int ldc = 39;

  std::vector<float> C = random_matrix(m, n, ldc);
  std::vector<float> expected = C;
  reference_sgemm(m, n, k, 2.0f, {}, 0, {}, 0, 0.75f, expected, ldc);

  float *d_C = nullptr;
  CHECK_CUDA(cudaMalloc(&d_C, C.size() * sizeof(float)));
  CHECK_CUDA(cudaMemcpy(d_C, C.data(), C.size() * sizeof(float),
                        cudaMemcpyHostToDevice));

  CHECK_CUDA(pmpp::gemm::sgemm_tiled_16_2x2(
      m, n, k, 2.0f, nullptr, 0, nullptr, 0, 0.75f, d_C, ldc));
  CHECK_CUDA(cudaDeviceSynchronize());

  CHECK_CUDA(cudaMemcpy(C.data(), d_C, C.size() * sizeof(float),
                        cudaMemcpyDeviceToHost));
  expect_active_matrix_near(expected, C, m, n, ldc, 1e-6f);

  CHECK_CUDA(cudaFree(d_C));
}

TEST(SgemmTiled16_2x2CoalescedTest, SupportsZeroKByScalingC) {
  SKIP_IF_CUDA_RUNTIME_UNAVAILABLE();

  constexpr int m = 33;
  constexpr int n = 35;
  constexpr int k = 0;
  constexpr int ldc = 39;

  std::vector<float> C = random_matrix(m, n, ldc);
  std::vector<float> expected = C;
  reference_sgemm(m, n, k, 2.0f, {}, 0, {}, 0, 0.75f, expected, ldc);

  float *d_C = nullptr;
  CHECK_CUDA(cudaMalloc(&d_C, C.size() * sizeof(float)));
  CHECK_CUDA(cudaMemcpy(d_C, C.data(), C.size() * sizeof(float),
                        cudaMemcpyHostToDevice));

  CHECK_CUDA(pmpp::gemm::sgemm_tiled_16_2x2_coalesced(
      m, n, k, 2.0f, nullptr, 0, nullptr, 0, 0.75f, d_C, ldc));
  CHECK_CUDA(cudaDeviceSynchronize());

  CHECK_CUDA(cudaMemcpy(C.data(), d_C, C.size() * sizeof(float),
                        cudaMemcpyDeviceToHost));
  expect_active_matrix_near(expected, C, m, n, ldc, 1e-6f);

  CHECK_CUDA(cudaFree(d_C));
}

TEST(SgemmTiled16_2x2K32CoalescedTest, SupportsZeroKByScalingC) {
  SKIP_IF_CUDA_RUNTIME_UNAVAILABLE();

  constexpr int m = 33;
  constexpr int n = 35;
  constexpr int k = 0;
  constexpr int ldc = 39;

  std::vector<float> C = random_matrix(m, n, ldc);
  std::vector<float> expected = C;
  reference_sgemm(m, n, k, 2.0f, {}, 0, {}, 0, 0.75f, expected, ldc);

  float *d_C = nullptr;
  CHECK_CUDA(cudaMalloc(&d_C, C.size() * sizeof(float)));
  CHECK_CUDA(cudaMemcpy(d_C, C.data(), C.size() * sizeof(float),
                        cudaMemcpyHostToDevice));

  CHECK_CUDA(pmpp::gemm::sgemm_tiled_16_2x2_k32_coalesced(
      m, n, k, 2.0f, nullptr, 0, nullptr, 0, 0.75f, d_C, ldc));
  CHECK_CUDA(cudaDeviceSynchronize());

  CHECK_CUDA(cudaMemcpy(C.data(), d_C, C.size() * sizeof(float),
                        cudaMemcpyDeviceToHost));
  expect_active_matrix_near(expected, C, m, n, ldc, 1e-6f);

  CHECK_CUDA(cudaFree(d_C));
}

TEST(SgemmIjkTest, RejectsInvalidLeadingDimensions) {
  float *dummy = reinterpret_cast<float *>(0x1);
  EXPECT_EQ(pmpp::gemm::sgemm_ijk(4, 4, 4, 1.0f, dummy, 3, dummy, 4, 0.0f,
                                  dummy, 4),
            cudaErrorInvalidValue);
  EXPECT_EQ(pmpp::gemm::sgemm_ijk(4, 4, 4, 1.0f, dummy, 4, dummy, 3, 0.0f,
                                  dummy, 4),
            cudaErrorInvalidValue);
  EXPECT_EQ(pmpp::gemm::sgemm_ijk(4, 4, 4, 1.0f, dummy, 4, dummy, 4, 0.0f,
                                  dummy, 3),
            cudaErrorInvalidValue);
}

TEST(SgemmTiled16Test, RejectsInvalidLeadingDimensions) {
  float *dummy = reinterpret_cast<float *>(0x1);
  EXPECT_EQ(pmpp::gemm::sgemm_tiled_16(4, 4, 4, 1.0f, dummy, 3, dummy, 4,
                                       0.0f, dummy, 4),
            cudaErrorInvalidValue);
  EXPECT_EQ(pmpp::gemm::sgemm_tiled_16(4, 4, 4, 1.0f, dummy, 4, dummy, 3,
                                       0.0f, dummy, 4),
            cudaErrorInvalidValue);
  EXPECT_EQ(pmpp::gemm::sgemm_tiled_16(4, 4, 4, 1.0f, dummy, 4, dummy, 4,
                                       0.0f, dummy, 3),
            cudaErrorInvalidValue);
}

TEST(SgemmTiled16_2x2Test, RejectsInvalidLeadingDimensions) {
  float *dummy = reinterpret_cast<float *>(0x1);
  EXPECT_EQ(pmpp::gemm::sgemm_tiled_16_2x2(4, 4, 4, 1.0f, dummy, 3, dummy, 4,
                                           0.0f, dummy, 4),
            cudaErrorInvalidValue);
  EXPECT_EQ(pmpp::gemm::sgemm_tiled_16_2x2(4, 4, 4, 1.0f, dummy, 4, dummy, 3,
                                           0.0f, dummy, 4),
            cudaErrorInvalidValue);
  EXPECT_EQ(pmpp::gemm::sgemm_tiled_16_2x2(4, 4, 4, 1.0f, dummy, 4, dummy, 4,
                                           0.0f, dummy, 3),
            cudaErrorInvalidValue);
}

TEST(SgemmTiled16_2x2CoalescedTest, RejectsInvalidLeadingDimensions) {
  float *dummy = reinterpret_cast<float *>(0x1);
  EXPECT_EQ(pmpp::gemm::sgemm_tiled_16_2x2_coalesced(
                4, 4, 4, 1.0f, dummy, 3, dummy, 4, 0.0f, dummy, 4),
            cudaErrorInvalidValue);
  EXPECT_EQ(pmpp::gemm::sgemm_tiled_16_2x2_coalesced(
                4, 4, 4, 1.0f, dummy, 4, dummy, 3, 0.0f, dummy, 4),
            cudaErrorInvalidValue);
  EXPECT_EQ(pmpp::gemm::sgemm_tiled_16_2x2_coalesced(
                4, 4, 4, 1.0f, dummy, 4, dummy, 4, 0.0f, dummy, 3),
            cudaErrorInvalidValue);
}

TEST(SgemmTiled16_2x2K32CoalescedTest, RejectsInvalidLeadingDimensions) {
  float *dummy = reinterpret_cast<float *>(0x1);
  EXPECT_EQ(pmpp::gemm::sgemm_tiled_16_2x2_k32_coalesced(
                4, 4, 4, 1.0f, dummy, 3, dummy, 4, 0.0f, dummy, 4),
            cudaErrorInvalidValue);
  EXPECT_EQ(pmpp::gemm::sgemm_tiled_16_2x2_k32_coalesced(
                4, 4, 4, 1.0f, dummy, 4, dummy, 3, 0.0f, dummy, 4),
            cudaErrorInvalidValue);
  EXPECT_EQ(pmpp::gemm::sgemm_tiled_16_2x2_k32_coalesced(
                4, 4, 4, 1.0f, dummy, 4, dummy, 4, 0.0f, dummy, 3),
            cudaErrorInvalidValue);
}

#undef CHECK_CUDA
#undef SKIP_IF_CUDA_RUNTIME_UNAVAILABLE

} // namespace
