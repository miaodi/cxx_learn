#include "sgemm.h"

#include <cuda_runtime.h>

namespace pmpp::gemm {
namespace {

constexpr int kBlockSize = 256;

__global__ void sgemm_ijk_kernel(int m, int n, int k, float alpha,
                                 const float *__restrict__ A, int lda,
                                 const float *__restrict__ B, int ldb,
                                 float beta, float *__restrict__ C, int ldc) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= m * n) {
    return;
  }

  const int row = index / n;
  const int col = index % n;

  float sum = 0.0f;
  for (int kk = 0; kk < k; ++kk) {
    sum += A[row * lda + kk] * B[kk * ldb + col];
  }

  C[row * ldc + col] = alpha * sum + beta * C[row * ldc + col];
}

template <int Tile>
__global__ void sgemm_tiled_kernel(int m, int n, int k, float alpha,
                                   const float *__restrict__ A, int lda,
                                   const float *__restrict__ B, int ldb,
                                   float beta, float *__restrict__ C, int ldc) {
  static_assert(Tile > 0, "Tile size must be positive");
  static_assert(Tile * Tile <= 4096, "Tile uses too many CUDA threads");

  const int tileCols = (n + Tile - 1) / Tile;
  const int blockRow = blockIdx.x / tileCols;
  const int blockCol = blockIdx.x % tileCols;
  const int localRow = threadIdx.x / Tile;
  const int localCol = threadIdx.x % Tile;
  const int row = blockRow * Tile + localRow;
  const int col = blockCol * Tile + localCol;

  __shared__ float tileA[Tile][Tile];
  __shared__ float tileB[Tile][Tile];

  float sum = 0.0f;
  const int kTiles = (k + Tile - 1) / Tile;
  for (int tile = 0; tile < kTiles; ++tile) {
    const int aCol = tile * Tile + localCol;
    const int bRow = tile * Tile + localRow;

    tileA[localRow][localCol] =
        (row < m && aCol < k) ? A[row * lda + aCol] : 0.0f;
    tileB[localRow][localCol] =
        (bRow < k && col < n) ? B[bRow * ldb + col] : 0.0f;

    __syncthreads();

    for (int kk = 0; kk < Tile; ++kk) {
      sum += tileA[localRow][kk] * tileB[kk][localCol];
    }

    __syncthreads();
  }

  if (row < m && col < n) {
    C[row * ldc + col] = alpha * sum + beta * C[row * ldc + col];
  }
}

template <int Tile>
__global__ void sgemm_tiled_2x2_kernel(int m, int n, int k, float alpha,
                                       const float *__restrict__ A, int lda,
                                       const float *__restrict__ B, int ldb,
                                       float beta, float *__restrict__ C,
                                       int ldc) {
  static_assert(Tile > 0, "Tile size must be positive");
  static_assert(Tile * Tile <= 1024, "Tile uses too many CUDA threads");

  constexpr int ThreadRows = 2;
  constexpr int ThreadCols = 2;
  constexpr int OutputRows = Tile * ThreadRows;
  constexpr int OutputCols = Tile * ThreadCols;

  const int tileCols = (n + OutputCols - 1) / OutputCols;
  const int blockRow = blockIdx.x / tileCols;
  const int blockCol = blockIdx.x % tileCols;
  const int localThreadRow = threadIdx.x / Tile;
  const int localThreadCol = threadIdx.x % Tile;
  const int rowBase = blockRow * OutputRows + localThreadRow * ThreadRows;
  const int colBase = blockCol * OutputCols + localThreadCol * ThreadCols;

  __shared__ float tileA[OutputRows][Tile];
  __shared__ float tileB[Tile][OutputCols];

  float c00 = 0.0f;
  float c01 = 0.0f;
  float c10 = 0.0f;
  float c11 = 0.0f;

  const int kTiles = (k + Tile - 1) / Tile;
  for (int tile = 0; tile < kTiles; ++tile) {
    const int kCol = tile * Tile + localThreadCol;
    const int kRow = tile * Tile + localThreadRow;

    const int aRow0 = rowBase;
    const int aRow1 = rowBase + 1;
    tileA[localThreadRow * ThreadRows][localThreadCol] =
        (aRow0 < m && kCol < k) ? A[aRow0 * lda + kCol] : 0.0f;
    tileA[localThreadRow * ThreadRows + 1][localThreadCol] =
        (aRow1 < m && kCol < k) ? A[aRow1 * lda + kCol] : 0.0f;

    const int bCol0 = colBase;
    const int bCol1 = colBase + 1;
    tileB[localThreadRow][localThreadCol * ThreadCols] =
        (kRow < k && bCol0 < n) ? B[kRow * ldb + bCol0] : 0.0f;
    tileB[localThreadRow][localThreadCol * ThreadCols + 1] =
        (kRow < k && bCol1 < n) ? B[kRow * ldb + bCol1] : 0.0f;

    __syncthreads();

    for (int kk = 0; kk < Tile; ++kk) {
      const float a0 = tileA[localThreadRow * ThreadRows][kk];
      const float a1 = tileA[localThreadRow * ThreadRows + 1][kk];
      const float b0 = tileB[kk][localThreadCol * ThreadCols];
      const float b1 = tileB[kk][localThreadCol * ThreadCols + 1];
      c00 += a0 * b0;
      c01 += a0 * b1;
      c10 += a1 * b0;
      c11 += a1 * b1;
    }

    __syncthreads();
  }

  if (rowBase < m && colBase < n) {
    C[rowBase * ldc + colBase] = alpha * c00 + beta * C[rowBase * ldc + colBase];
  }
  if (rowBase < m && colBase + 1 < n) {
    C[rowBase * ldc + colBase + 1] =
        alpha * c01 + beta * C[rowBase * ldc + colBase + 1];
  }
  if (rowBase + 1 < m && colBase < n) {
    C[(rowBase + 1) * ldc + colBase] =
        alpha * c10 + beta * C[(rowBase + 1) * ldc + colBase];
  }
  if (rowBase + 1 < m && colBase + 1 < n) {
    C[(rowBase + 1) * ldc + colBase + 1] =
        alpha * c11 + beta * C[(rowBase + 1) * ldc + colBase + 1];
  }
}

template <int Tile, int KTile>
__global__ void
sgemm_tiled_2x2_coalesced_kernel(int m, int n, int k, float alpha,
                                 const float *__restrict__ A, int lda,
                                 const float *__restrict__ B, int ldb,
                                 float beta, float *__restrict__ C, int ldc) {
  static_assert(Tile > 0, "Tile size must be positive");
  static_assert(KTile > 0, "K tile size must be positive");
  static_assert(Tile * Tile <= 1024, "Tile uses too many CUDA threads");

  constexpr int ThreadRows = 2;
  constexpr int ThreadCols = 2;
  constexpr int OutputRows = Tile * ThreadRows;
  constexpr int OutputCols = Tile * ThreadCols;
  constexpr int ThreadsPerBlock = Tile * Tile;

  const int tileCols = (n + OutputCols - 1) / OutputCols;
  const int blockRow = blockIdx.x / tileCols;
  const int blockCol = blockIdx.x % tileCols;
  const int localThreadRow = threadIdx.x / Tile;
  const int localThreadCol = threadIdx.x % Tile;
  const int rowBase = blockRow * OutputRows + localThreadRow * ThreadRows;
  const int colBase = blockCol * OutputCols + localThreadCol * ThreadCols;

  __shared__ float tileA[OutputRows][KTile];
  __shared__ float tileB[KTile][OutputCols];

  float c00 = 0.0f;
  float c01 = 0.0f;
  float c10 = 0.0f;
  float c11 = 0.0f;

  const int kTiles = (k + KTile - 1) / KTile;
  for (int tile = 0; tile < kTiles; ++tile) {
    for (int offset = threadIdx.x; offset < OutputRows * KTile;
         offset += ThreadsPerBlock) {
      const int sharedRow = offset / KTile;
      const int sharedCol = offset % KTile;
      const int globalRow = blockRow * OutputRows + sharedRow;
      const int globalCol = tile * KTile + sharedCol;
      tileA[sharedRow][sharedCol] =
          (globalRow < m && globalCol < k) ? A[globalRow * lda + globalCol]
                                           : 0.0f;
    }

    for (int offset = threadIdx.x; offset < KTile * OutputCols;
         offset += ThreadsPerBlock) {
      const int sharedRow = offset / OutputCols;
      const int sharedCol = offset % OutputCols;
      const int globalRow = tile * KTile + sharedRow;
      const int globalCol = blockCol * OutputCols + sharedCol;
      tileB[sharedRow][sharedCol] =
          (globalRow < k && globalCol < n) ? B[globalRow * ldb + globalCol]
                                           : 0.0f;
    }

    __syncthreads();

    for (int kk = 0; kk < KTile; ++kk) {
      const float a0 = tileA[localThreadRow * ThreadRows][kk];
      const float a1 = tileA[localThreadRow * ThreadRows + 1][kk];
      const float b0 = tileB[kk][localThreadCol * ThreadCols];
      const float b1 = tileB[kk][localThreadCol * ThreadCols + 1];
      c00 += a0 * b0;
      c01 += a0 * b1;
      c10 += a1 * b0;
      c11 += a1 * b1;
    }

    __syncthreads();
  }

  if (rowBase < m && colBase < n) {
    C[rowBase * ldc + colBase] =
        alpha * c00 + beta * C[rowBase * ldc + colBase];
  }
  if (rowBase < m && colBase + 1 < n) {
    C[rowBase * ldc + colBase + 1] =
        alpha * c01 + beta * C[rowBase * ldc + colBase + 1];
  }
  if (rowBase + 1 < m && colBase < n) {
    C[(rowBase + 1) * ldc + colBase] =
        alpha * c10 + beta * C[(rowBase + 1) * ldc + colBase];
  }
  if (rowBase + 1 < m && colBase + 1 < n) {
    C[(rowBase + 1) * ldc + colBase + 1] =
        alpha * c11 + beta * C[(rowBase + 1) * ldc + colBase + 1];
  }
}

bool has_invalid_shape(int m, int n, int k, int lda, int ldb, int ldc) {
  if (m < 0 || n < 0 || k < 0) {
    return true;
  }
  if (m == 0 || n == 0) {
    return false;
  }
  if (ldc < n) {
    return true;
  }
  return k > 0 && (lda < k || ldb < n);
}

template <int Tile>
cudaError_t sgemm_tiled_impl(int m, int n, int k, float alpha, const float *A,
                             int lda, const float *B, int ldb, float beta,
                             float *C, int ldc, cudaStream_t stream) {
  if (has_invalid_shape(m, n, k, lda, ldb, ldc)) {
    return cudaErrorInvalidValue;
  }
  if (m == 0 || n == 0) {
    return cudaSuccess;
  }
  if (C == nullptr || (k > 0 && (A == nullptr || B == nullptr))) {
    return cudaErrorInvalidDevicePointer;
  }

  const int tileRows = (m + Tile - 1) / Tile;
  const int tileCols = (n + Tile - 1) / Tile;
  const int grid = tileRows * tileCols;
  sgemm_tiled_kernel<Tile><<<grid, Tile * Tile, 0, stream>>>(
      m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
  return cudaGetLastError();
}

template <int Tile>
cudaError_t sgemm_tiled_2x2_impl(int m, int n, int k, float alpha,
                                 const float *A, int lda, const float *B,
                                 int ldb, float beta, float *C, int ldc,
                                 cudaStream_t stream) {
  if (has_invalid_shape(m, n, k, lda, ldb, ldc)) {
    return cudaErrorInvalidValue;
  }
  if (m == 0 || n == 0) {
    return cudaSuccess;
  }
  if (C == nullptr || (k > 0 && (A == nullptr || B == nullptr))) {
    return cudaErrorInvalidDevicePointer;
  }

  constexpr int outputRows = Tile * 2;
  constexpr int outputCols = Tile * 2;
  const int tileRows = (m + outputRows - 1) / outputRows;
  const int tileCols = (n + outputCols - 1) / outputCols;
  const int grid = tileRows * tileCols;
  sgemm_tiled_2x2_kernel<Tile><<<grid, Tile * Tile, 0, stream>>>(
      m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
  return cudaGetLastError();
}

template <int Tile, int KTile>
cudaError_t sgemm_tiled_2x2_coalesced_impl(int m, int n, int k, float alpha,
                                           const float *A, int lda,
                                           const float *B, int ldb, float beta,
                                           float *C, int ldc,
                                           cudaStream_t stream) {
  if (has_invalid_shape(m, n, k, lda, ldb, ldc)) {
    return cudaErrorInvalidValue;
  }
  if (m == 0 || n == 0) {
    return cudaSuccess;
  }
  if (C == nullptr || (k > 0 && (A == nullptr || B == nullptr))) {
    return cudaErrorInvalidDevicePointer;
  }

  constexpr int outputRows = Tile * 2;
  constexpr int outputCols = Tile * 2;
  const int tileRows = (m + outputRows - 1) / outputRows;
  const int tileCols = (n + outputCols - 1) / outputCols;
  const int grid = tileRows * tileCols;
  sgemm_tiled_2x2_coalesced_kernel<Tile, KTile>
      <<<grid, Tile * Tile, 0, stream>>>(
      m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
  return cudaGetLastError();
}

} // namespace

cudaError_t sgemm_ijk(int m, int n, int k, float alpha, const float *A,
                      int lda, const float *B, int ldb, float beta, float *C,
                      int ldc, cudaStream_t stream) {
  if (has_invalid_shape(m, n, k, lda, ldb, ldc)) {
    return cudaErrorInvalidValue;
  }
  if (m == 0 || n == 0) {
    return cudaSuccess;
  }
  if (C == nullptr || (k > 0 && (A == nullptr || B == nullptr))) {
    return cudaErrorInvalidDevicePointer;
  }

  const int grid = (m * n + kBlockSize - 1) / kBlockSize;
  sgemm_ijk_kernel<<<grid, kBlockSize, 0, stream>>>(m, n, k, alpha, A, lda, B,
                                                    ldb, beta, C, ldc);
  return cudaGetLastError();
}

cudaError_t sgemm_tiled_16(int m, int n, int k, float alpha, const float *A,
                           int lda, const float *B, int ldb, float beta,
                           float *C, int ldc, cudaStream_t stream) {
  return sgemm_tiled_impl<16>(m, n, k, alpha, A, lda, B, ldb, beta, C, ldc,
                              stream);
}

cudaError_t sgemm_tiled_16_2x2(int m, int n, int k, float alpha,
                               const float *A, int lda, const float *B,
                               int ldb, float beta, float *C, int ldc,
                               cudaStream_t stream) {
  return sgemm_tiled_2x2_impl<16>(m, n, k, alpha, A, lda, B, ldb, beta, C, ldc,
                                  stream);
}

cudaError_t sgemm_tiled_16_2x2_coalesced(int m, int n, int k, float alpha,
                                         const float *A, int lda,
                                         const float *B, int ldb, float beta,
                                         float *C, int ldc,
                                         cudaStream_t stream) {
  return sgemm_tiled_2x2_coalesced_impl<16, 16>(
      m, n, k, alpha, A, lda, B, ldb, beta, C, ldc, stream);
}

cudaError_t sgemm_tiled_16_2x2_k32_coalesced(int m, int n, int k, float alpha,
                                             const float *A, int lda,
                                             const float *B, int ldb,
                                             float beta, float *C, int ldc,
                                             cudaStream_t stream) {
  return sgemm_tiled_2x2_coalesced_impl<16, 32>(
      m, n, k, alpha, A, lda, B, ldb, beta, C, ldc, stream);
}

} // namespace pmpp::gemm
