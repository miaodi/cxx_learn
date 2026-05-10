#pragma once

#include <cuda_runtime_api.h>

namespace pmpp::gemm {

// Row-major, no-transpose SGEMM:
// C = alpha * A * B + beta * C
//
// A is M x K with row stride lda.
// B is K x N with row stride ldb.
// C is M x N with row stride ldc.
//
// All pointers are device pointers.
cudaError_t sgemm_ijk(int m, int n, int k, float alpha, const float *A,
                      int lda, const float *B, int ldb, float beta, float *C,
                      int ldc, cudaStream_t stream = nullptr);

cudaError_t sgemm_tiled_16(int m, int n, int k, float alpha, const float *A,
                           int lda, const float *B, int ldb, float beta,
                           float *C, int ldc,
                           cudaStream_t stream = nullptr);

cudaError_t sgemm_tiled_16_2x2(int m, int n, int k, float alpha,
                               const float *A, int lda, const float *B,
                               int ldb, float beta, float *C, int ldc,
                               cudaStream_t stream = nullptr);

cudaError_t sgemm_tiled_16_2x2_k32(int m, int n, int k, float alpha,
                                   const float *A, int lda, const float *B,
                                   int ldb, float beta, float *C, int ldc,
                                   cudaStream_t stream = nullptr);

cudaError_t sgemm_tiled_16_2x2_k64(int m, int n, int k, float alpha,
                                   const float *A, int lda, const float *B,
                                   int ldb, float beta, float *C, int ldc,
                                   cudaStream_t stream = nullptr);

cudaError_t sgemm_tiled_16_4x4(int m, int n, int k, float alpha,
                               const float *A, int lda, const float *B,
                               int ldb, float beta, float *C, int ldc,
                               cudaStream_t stream = nullptr);

cudaError_t sgemm_tiled_16_4x4_k32(int m, int n, int k, float alpha,
                                   const float *A, int lda, const float *B,
                                   int ldb, float beta, float *C, int ldc,
                                   cudaStream_t stream = nullptr);

cudaError_t sgemm_tiled_16_4x4_k64(int m, int n, int k, float alpha,
                                   const float *A, int lda, const float *B,
                                   int ldb, float beta, float *C, int ldc,
                                   cudaStream_t stream = nullptr);

cudaError_t sgemm_tiled_16_8x8(int m, int n, int k, float alpha,
                               const float *A, int lda, const float *B,
                               int ldb, float beta, float *C, int ldc,
                               cudaStream_t stream = nullptr);

cudaError_t sgemm_tiled_16_8x8_k32(int m, int n, int k, float alpha,
                                   const float *A, int lda, const float *B,
                                   int ldb, float beta, float *C, int ldc,
                                   cudaStream_t stream = nullptr);

cudaError_t sgemm_tiled_16_8x8_k64(int m, int n, int k, float alpha,
                                   const float *A, int lda, const float *B,
                                   int ldb, float beta, float *C, int ldc,
                                   cudaStream_t stream = nullptr);

} // namespace pmpp::gemm
