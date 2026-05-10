#include "sgemm.h"

#include <cuda_runtime.h>

namespace pmpp::gemm {
namespace {

constexpr int kBlockSize = 256;
constexpr int kDefaultMaxSharedMemoryBytes = 48 * 1024;

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

    tileA[localCol][localRow] =
        (row < m && aCol < k) ? A[row * lda + aCol] : 0.0f;
    tileB[localRow][localCol] =
        (bRow < k && col < n) ? B[bRow * ldb + col] : 0.0f;

    __syncthreads();

    for (int kk = 0; kk < Tile; ++kk) {
      sum += tileA[kk][localRow] * tileB[kk][localCol];
    }

    __syncthreads();
  }

  if (row < m && col < n) {
    C[row * ldc + col] = alpha * sum + beta * C[row * ldc + col];
  }
}

template <int Tile, int ThreadTile, int KTile>
__global__ void sgemm_tiled_thread_tile_kernel(
    int m, int n, int k, float alpha, const float *__restrict__ A, int lda,
    const float *__restrict__ B, int ldb, float beta, float *__restrict__ C,
    int ldc) {
  static_assert(Tile > 0, "Tile size must be positive");
  static_assert(ThreadTile > 0, "Thread tile size must be positive");
  static_assert(KTile > 0, "K tile size must be positive");
  static_assert(Tile * Tile <= 1024, "Tile uses too many CUDA threads");

  constexpr int OutputRows = Tile * ThreadTile;
  constexpr int OutputCols = Tile * ThreadTile;
  constexpr int ThreadsPerBlock = Tile * Tile;

  const int tileCols = (n + OutputCols - 1) / OutputCols;
  const int blockRow = blockIdx.x / tileCols;
  const int blockCol = blockIdx.x % tileCols;
  const int localThreadRow = threadIdx.x / Tile;
  const int localThreadCol = threadIdx.x % Tile;
  const int rowBase = blockRow * OutputRows + localThreadRow * ThreadTile;
  const int colBase = blockCol * OutputCols + localThreadCol * ThreadTile;

  extern __shared__ float shared[];
  float *const tileA = shared;
  float *const tileB = tileA + OutputRows * KTile;

  float acc[ThreadTile][ThreadTile] = {};

  const int kTiles = (k + KTile - 1) / KTile;
  for (int tile = 0; tile < kTiles; ++tile) {
    if constexpr (KTile == Tile) {
      const int kCol = tile * KTile + localThreadCol;
      const int kRow = tile * KTile + localThreadRow;

#pragma unroll
      for (int r = 0; r < ThreadTile; ++r) {
        const int aRow = rowBase + r;
        tileA[(localThreadRow * ThreadTile + r) * KTile + localThreadCol] =
            (aRow < m && kCol < k) ? A[aRow * lda + kCol] : 0.0f;
      }

#pragma unroll
      for (int c = 0; c < ThreadTile; ++c) {
        const int bCol = colBase + c;
        tileB[localThreadRow * OutputCols + localThreadCol * ThreadTile + c] =
            (kRow < k && bCol < n) ? B[kRow * ldb + bCol] : 0.0f;
      }
    } else {
      for (int offset = threadIdx.x; offset < OutputRows * KTile;
           offset += ThreadsPerBlock) {
        const int sharedRow = offset / KTile;
        const int sharedCol = offset % KTile;
        const int globalRow = blockRow * OutputRows + sharedRow;
        const int globalCol = tile * KTile + sharedCol;
        tileA[sharedRow * KTile + sharedCol] =
            (globalRow < m && globalCol < k) ? A[globalRow * lda + globalCol]
                                             : 0.0f;
      }

      for (int offset = threadIdx.x; offset < KTile * OutputCols;
           offset += ThreadsPerBlock) {
        const int sharedRow = offset / OutputCols;
        const int sharedCol = offset % OutputCols;
        const int globalRow = tile * KTile + sharedRow;
        const int globalCol = blockCol * OutputCols + sharedCol;
        tileB[sharedRow * OutputCols + sharedCol] =
            (globalRow < k && globalCol < n) ? B[globalRow * ldb + globalCol]
                                             : 0.0f;
      }
    }

    __syncthreads();

    for (int kk = 0; kk < KTile; ++kk) {
      float a[ThreadTile];
      float b[ThreadTile];
#pragma unroll
      for (int r = 0; r < ThreadTile; ++r) {
        a[r] = tileA[(localThreadRow * ThreadTile + r) * KTile + kk];
      }
#pragma unroll
      for (int c = 0; c < ThreadTile; ++c) {
        b[c] = tileB[kk * OutputCols + localThreadCol * ThreadTile + c];
      }
#pragma unroll
      for (int r = 0; r < ThreadTile; ++r) {
#pragma unroll
        for (int c = 0; c < ThreadTile; ++c) {
          acc[r][c] += a[r] * b[c];
        }
      }
    }

    __syncthreads();
  }

#pragma unroll
  for (int r = 0; r < ThreadTile; ++r) {
    const int row = rowBase + r;
    if (row < m) {
#pragma unroll
      for (int c = 0; c < ThreadTile; ++c) {
        const int col = colBase + c;
        if (col < n) {
          C[row * ldc + col] = alpha * acc[r][c] + beta * C[row * ldc + col];
        }
      }
    }
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

template <int Tile, int ThreadTile, int KTile>
cudaError_t sgemm_tiled_thread_tile_impl(int m, int n, int k, float alpha,
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

  constexpr int outputRows = Tile * ThreadTile;
  constexpr int outputCols = Tile * ThreadTile;
  constexpr int sharedMemoryBytes =
      (outputRows * KTile + KTile * outputCols) * sizeof(float);
  const int tileRows = (m + outputRows - 1) / outputRows;
  const int tileCols = (n + outputCols - 1) / outputCols;
  const int grid = tileRows * tileCols;

  if constexpr (sharedMemoryBytes > kDefaultMaxSharedMemoryBytes) {
    static const cudaError_t attributeStatus = cudaFuncSetAttribute(
        sgemm_tiled_thread_tile_kernel<Tile, ThreadTile, KTile>,
        cudaFuncAttributeMaxDynamicSharedMemorySize, sharedMemoryBytes);
    if (attributeStatus != cudaSuccess) {
      return attributeStatus;
    }
  }

  sgemm_tiled_thread_tile_kernel<Tile, ThreadTile, KTile>
      <<<grid, Tile * Tile, sharedMemoryBytes, stream>>>(
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
  return sgemm_tiled_thread_tile_impl<16, 2, 16>(
      m, n, k, alpha, A, lda, B, ldb, beta, C, ldc, stream);
}

cudaError_t sgemm_tiled_16_2x2_k32(int m, int n, int k, float alpha,
                                   const float *A, int lda, const float *B,
                                   int ldb, float beta, float *C, int ldc,
                                   cudaStream_t stream) {
  return sgemm_tiled_thread_tile_impl<16, 2, 32>(
      m, n, k, alpha, A, lda, B, ldb, beta, C, ldc, stream);
}

cudaError_t sgemm_tiled_16_2x2_k64(int m, int n, int k, float alpha,
                                   const float *A, int lda, const float *B,
                                   int ldb, float beta, float *C, int ldc,
                                   cudaStream_t stream) {
  return sgemm_tiled_thread_tile_impl<16, 2, 64>(
      m, n, k, alpha, A, lda, B, ldb, beta, C, ldc, stream);
}

cudaError_t sgemm_tiled_16_4x4(int m, int n, int k, float alpha,
                               const float *A, int lda, const float *B,
                               int ldb, float beta, float *C, int ldc,
                               cudaStream_t stream) {
  return sgemm_tiled_thread_tile_impl<16, 4, 16>(
      m, n, k, alpha, A, lda, B, ldb, beta, C, ldc, stream);
}

cudaError_t sgemm_tiled_16_4x4_k32(int m, int n, int k, float alpha,
                                   const float *A, int lda, const float *B,
                                   int ldb, float beta, float *C, int ldc,
                                   cudaStream_t stream) {
  return sgemm_tiled_thread_tile_impl<16, 4, 32>(
      m, n, k, alpha, A, lda, B, ldb, beta, C, ldc, stream);
}

cudaError_t sgemm_tiled_16_4x4_k64(int m, int n, int k, float alpha,
                                   const float *A, int lda, const float *B,
                                   int ldb, float beta, float *C, int ldc,
                                   cudaStream_t stream) {
  return sgemm_tiled_thread_tile_impl<16, 4, 64>(
      m, n, k, alpha, A, lda, B, ldb, beta, C, ldc, stream);
}

cudaError_t sgemm_tiled_16_8x8(int m, int n, int k, float alpha,
                               const float *A, int lda, const float *B,
                               int ldb, float beta, float *C, int ldc,
                               cudaStream_t stream) {
  return sgemm_tiled_thread_tile_impl<16, 8, 16>(
      m, n, k, alpha, A, lda, B, ldb, beta, C, ldc, stream);
}

cudaError_t sgemm_tiled_16_8x8_k32(int m, int n, int k, float alpha,
                                   const float *A, int lda, const float *B,
                                   int ldb, float beta, float *C, int ldc,
                                   cudaStream_t stream) {
  return sgemm_tiled_thread_tile_impl<16, 8, 32>(
      m, n, k, alpha, A, lda, B, ldb, beta, C, ldc, stream);
}

cudaError_t sgemm_tiled_16_8x8_k64(int m, int n, int k, float alpha,
                                   const float *A, int lda, const float *B,
                                   int ldb, float beta, float *C, int ldc,
                                   cudaStream_t stream) {
  return sgemm_tiled_thread_tile_impl<16, 8, 64>(
      m, n, k, alpha, A, lda, B, ldb, beta, C, ldc, stream);
}

} // namespace pmpp::gemm
