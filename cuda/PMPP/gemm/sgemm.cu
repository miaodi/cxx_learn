#include "sgemm.h"

#include <cuda_runtime.h>

namespace pmpp::gemm {
namespace {

constexpr int kBlockSize = 256;
constexpr int kDefaultMaxSharedMemoryBytes = 48 * 1024;

enum class SharedTileOptimization {
  None,
  PadA,
  PadAAndVectorizedC,
  PadAAndCoalescedB,
  PadAAndCoalescedBAndVectorizedC,
};

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

template <int Stride>
__device__ __forceinline__ float &shared_tile_at(float *tile, int row,
                                                 int col) {
  return tile[row * Stride + col];
}

template <int Tile, int ThreadTile, int KTile, SharedTileOptimization Opt>
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
  constexpr bool PadA = Opt != SharedTileOptimization::None;
  constexpr bool CoalesceB =
      Opt == SharedTileOptimization::PadAAndCoalescedB ||
      Opt == SharedTileOptimization::PadAAndCoalescedBAndVectorizedC;
  constexpr bool VectorizeC =
      (Opt == SharedTileOptimization::PadAAndVectorizedC ||
       Opt == SharedTileOptimization::PadAAndCoalescedBAndVectorizedC) &&
      ThreadTile == 4;
  constexpr int TileAPadding = PadA && ThreadTile > 1 ? 16 / ThreadTile : 0;
  constexpr int TileAStride = KTile + TileAPadding;

  const int tileCols = (n + OutputCols - 1) / OutputCols;
  const int blockRow = blockIdx.x / tileCols;
  const int blockCol = blockIdx.x % tileCols;
  const int localThreadRow = threadIdx.x / Tile;
  const int localThreadCol = threadIdx.x % Tile;
  const int rowBase = blockRow * OutputRows + localThreadRow * ThreadTile;
  const int colBase = blockCol * OutputCols + localThreadCol * ThreadTile;

  extern __shared__ float shared[];
  float *const tileA = shared;
  float *const tileB = tileA + OutputRows * TileAStride;

  float acc[ThreadTile][ThreadTile] = {};

  const int kTiles = (k + KTile - 1) / KTile;
  for (int tile = 0; tile < kTiles; ++tile) {
    if constexpr (KTile == Tile) {
      const int kCol = tile * KTile + localThreadCol;
      const int kRow = tile * KTile + localThreadRow;

#pragma unroll
      for (int r = 0; r < ThreadTile; ++r) {
        const int aRow = rowBase + r;
        shared_tile_at<TileAStride>(tileA, localThreadRow * ThreadTile + r,
                                    localThreadCol) =
            (aRow < m && kCol < k) ? A[aRow * lda + kCol] : 0.0f;
      }

      if constexpr (!CoalesceB) {
#pragma unroll
        for (int c = 0; c < ThreadTile; ++c) {
          const int bCol = colBase + c;
          shared_tile_at<OutputCols>(tileB, localThreadRow,
                                     localThreadCol * ThreadTile + c) =
              (kRow < k && bCol < n) ? B[kRow * ldb + bCol] : 0.0f;
        }
      }
    } else {
      for (int offset = threadIdx.x; offset < OutputRows * KTile;
           offset += ThreadsPerBlock) {
        const int sharedRow = offset / KTile;
        const int sharedCol = offset % KTile;
        const int globalRow = blockRow * OutputRows + sharedRow;
        const int globalCol = tile * KTile + sharedCol;
        shared_tile_at<TileAStride>(tileA, sharedRow, sharedCol) =
            (globalRow < m && globalCol < k) ? A[globalRow * lda + globalCol]
                                             : 0.0f;
      }
    }

    if constexpr (KTile != Tile || CoalesceB) {
      for (int offset = threadIdx.x; offset < KTile * OutputCols;
           offset += ThreadsPerBlock) {
        const int sharedRow = offset / OutputCols;
        const int sharedCol = offset % OutputCols;
        const int globalRow = tile * KTile + sharedRow;
        const int globalCol = blockCol * OutputCols + sharedCol;
        shared_tile_at<OutputCols>(tileB, sharedRow, sharedCol) =
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
        a[r] = shared_tile_at<TileAStride>(
            tileA, localThreadRow * ThreadTile + r, kk);
      }
#pragma unroll
      for (int c = 0; c < ThreadTile; ++c) {
        b[c] = shared_tile_at<OutputCols>(
            tileB, kk, localThreadCol * ThreadTile + c);
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

  if constexpr (VectorizeC) {
#pragma unroll
    for (int r = 0; r < ThreadTile; ++r) {
      const int row = rowBase + r;
      if (row < m) {
        float *const cAddress = C + row * ldc + colBase;
        const unsigned long long cAddressValue =
            reinterpret_cast<unsigned long long>(cAddress);
        if (colBase + 3 < n && (cAddressValue & 0xfULL) == 0) {
          float4 out = make_float4(alpha * acc[r][0], alpha * acc[r][1],
                                   alpha * acc[r][2], alpha * acc[r][3]);
          if (beta != 0.0f) {
            const float4 old = *reinterpret_cast<const float4 *>(cAddress);
            out.x += beta * old.x;
            out.y += beta * old.y;
            out.z += beta * old.z;
            out.w += beta * old.w;
          }
          *reinterpret_cast<float4 *>(cAddress) = out;
        } else {
#pragma unroll
          for (int c = 0; c < ThreadTile; ++c) {
            const int col = colBase + c;
            if (col < n) {
              C[row * ldc + col] =
                  alpha * acc[r][c] + beta * C[row * ldc + col];
            }
          }
        }
      }
    }
  } else {
#pragma unroll
    for (int r = 0; r < ThreadTile; ++r) {
      const int row = rowBase + r;
      if (row < m) {
#pragma unroll
        for (int c = 0; c < ThreadTile; ++c) {
          const int col = colBase + c;
          if (col < n) {
            C[row * ldc + col] =
                alpha * acc[r][c] + beta * C[row * ldc + col];
          }
        }
      }
    }
  }
}

__global__ void sgemm_tiled_64x128_8x8_paddedA_vectorizedC_kernel(
    int m, int n, int k, float alpha, const float *__restrict__ A, int lda,
    const float *__restrict__ B, int ldb, float beta, float *__restrict__ C,
    int ldc) {
  constexpr int BlockRows = 64;
  constexpr int BlockCols = 128;
  constexpr int KTile = 8;
  constexpr int ThreadRows = 8;
  constexpr int ThreadCols = 8;
  constexpr int RowGroups = BlockRows / ThreadRows;
  constexpr int ColGroups = BlockCols / ThreadCols;
  constexpr int ThreadsPerBlock = RowGroups * ColGroups;
  constexpr int TileAStride = KTile + 1;

  static_assert(ThreadsPerBlock == 128);

  const int tileCols = (n + BlockCols - 1) / BlockCols;
  const int blockRow = blockIdx.x / tileCols;
  const int blockCol = blockIdx.x % tileCols;
  const int localRowGroup = threadIdx.x / ColGroups;
  const int localColGroup = threadIdx.x % ColGroups;
  const int rowBase = blockRow * BlockRows + localRowGroup * ThreadRows;
  const int colBase = blockCol * BlockCols + localColGroup * ThreadCols;

  extern __shared__ float shared[];
  float *const tileA = shared;
  float *const tileB = tileA + BlockRows * TileAStride;

  float acc[ThreadRows][ThreadCols] = {};

  const int kTiles = (k + KTile - 1) / KTile;
  for (int tile = 0; tile < kTiles; ++tile) {
    for (int offset = threadIdx.x; offset < BlockRows * KTile;
         offset += ThreadsPerBlock) {
      const int sharedRow = offset / KTile;
      const int sharedCol = offset % KTile;
      const int globalRow = blockRow * BlockRows + sharedRow;
      const int globalCol = tile * KTile + sharedCol;
      shared_tile_at<TileAStride>(tileA, sharedRow, sharedCol) =
          (globalRow < m && globalCol < k) ? A[globalRow * lda + globalCol]
                                           : 0.0f;
    }

    for (int offset = threadIdx.x; offset < KTile * BlockCols;
         offset += ThreadsPerBlock) {
      const int sharedRow = offset / BlockCols;
      const int sharedCol = offset % BlockCols;
      const int globalRow = tile * KTile + sharedRow;
      const int globalCol = blockCol * BlockCols + sharedCol;
      shared_tile_at<BlockCols>(tileB, sharedRow, sharedCol) =
          (globalRow < k && globalCol < n) ? B[globalRow * ldb + globalCol]
                                           : 0.0f;
    }

    __syncthreads();

#pragma unroll
    for (int kk = 0; kk < KTile; ++kk) {
      float a[ThreadRows];
      float b[ThreadCols];
#pragma unroll
      for (int r = 0; r < ThreadRows; ++r) {
        a[r] = shared_tile_at<TileAStride>(tileA, localRowGroup * ThreadRows + r,
                                           kk);
      }
#pragma unroll
      for (int c = 0; c < ThreadCols; ++c) {
        b[c] = shared_tile_at<BlockCols>(tileB, kk,
                                         localColGroup * ThreadCols + c);
      }
#pragma unroll
      for (int r = 0; r < ThreadRows; ++r) {
#pragma unroll
        for (int c = 0; c < ThreadCols; ++c) {
          acc[r][c] += a[r] * b[c];
        }
      }
    }

    __syncthreads();
  }

#pragma unroll
  for (int r = 0; r < ThreadRows; ++r) {
    const int row = rowBase + r;
    if (row < m) {
#pragma unroll
      for (int chunk = 0; chunk < ThreadCols; chunk += 4) {
        const int col = colBase + chunk;
        if (col >= n) {
          continue;
        }
        float *const cAddress = C + row * ldc + col;
        const unsigned long long cAddressValue =
            reinterpret_cast<unsigned long long>(cAddress);
        if (col + 3 < n && (cAddressValue & 0xfULL) == 0) {
          float4 out = make_float4(alpha * acc[r][chunk + 0],
                                   alpha * acc[r][chunk + 1],
                                   alpha * acc[r][chunk + 2],
                                   alpha * acc[r][chunk + 3]);
          if (beta != 0.0f) {
            const float4 old = *reinterpret_cast<const float4 *>(cAddress);
            out.x += beta * old.x;
            out.y += beta * old.y;
            out.z += beta * old.z;
            out.w += beta * old.w;
          }
          *reinterpret_cast<float4 *>(cAddress) = out;
        } else {
#pragma unroll
          for (int c = 0; c < 4; ++c) {
            const int scalarCol = col + c;
            if (scalarCol < n) {
              C[row * ldc + scalarCol] =
                  alpha * acc[r][chunk + c] + beta * C[row * ldc + scalarCol];
            }
          }
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

template <int Tile, int ThreadTile, int KTile,
          SharedTileOptimization Opt = SharedTileOptimization::None>
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
  constexpr bool padA = Opt != SharedTileOptimization::None;
  constexpr int tileAPadding = padA && ThreadTile > 1 ? 16 / ThreadTile : 0;
  constexpr int tileAStride = KTile + tileAPadding;
  constexpr int sharedMemoryBytes =
      (outputRows * tileAStride + KTile * outputCols) * sizeof(float);
  const int tileRows = (m + outputRows - 1) / outputRows;
  const int tileCols = (n + outputCols - 1) / outputCols;
  const int grid = tileRows * tileCols;

  if constexpr (sharedMemoryBytes > kDefaultMaxSharedMemoryBytes) {
    static const cudaError_t attributeStatus = cudaFuncSetAttribute(
        sgemm_tiled_thread_tile_kernel<Tile, ThreadTile, KTile, Opt>,
        cudaFuncAttributeMaxDynamicSharedMemorySize, sharedMemoryBytes);
    if (attributeStatus != cudaSuccess) {
      return attributeStatus;
    }
  }

  sgemm_tiled_thread_tile_kernel<Tile, ThreadTile, KTile, Opt>
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
  return sgemm_tiled_thread_tile_impl<16, 1, 16>(
      m, n, k, alpha, A, lda, B, ldb, beta, C, ldc, stream);
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

cudaError_t sgemm_tiled_16_4x4_paddedA(int m, int n, int k, float alpha,
                                       const float *A, int lda,
                                       const float *B, int ldb, float beta,
                                       float *C, int ldc,
                                       cudaStream_t stream) {
  return sgemm_tiled_thread_tile_impl<16, 4, 16,
                                      SharedTileOptimization::PadA>(
      m, n, k, alpha, A, lda, B, ldb, beta, C, ldc, stream);
}

cudaError_t sgemm_tiled_16_4x4_paddedA_coalescedB(
    int m, int n, int k, float alpha, const float *A, int lda, const float *B,
    int ldb, float beta, float *C, int ldc, cudaStream_t stream) {
  return sgemm_tiled_thread_tile_impl<
      16, 4, 16, SharedTileOptimization::PadAAndCoalescedB>(
      m, n, k, alpha, A, lda, B, ldb, beta, C, ldc, stream);
}

cudaError_t sgemm_tiled_16_4x4_paddedA_vectorizedC(
    int m, int n, int k, float alpha, const float *A, int lda, const float *B,
    int ldb, float beta, float *C, int ldc, cudaStream_t stream) {
  return sgemm_tiled_thread_tile_impl<
      16, 4, 16, SharedTileOptimization::PadAAndVectorizedC>(
      m, n, k, alpha, A, lda, B, ldb, beta, C, ldc, stream);
}

cudaError_t sgemm_tiled_16_4x4_paddedA_coalescedB_vectorizedC(
    int m, int n, int k, float alpha, const float *A, int lda, const float *B,
    int ldb, float beta, float *C, int ldc, cudaStream_t stream) {
  return sgemm_tiled_thread_tile_impl<
      16, 4, 16, SharedTileOptimization::PadAAndCoalescedBAndVectorizedC>(
      m, n, k, alpha, A, lda, B, ldb, beta, C, ldc, stream);
}

cudaError_t sgemm_tiled_64x128_8x8_paddedA_vectorizedC(
    int m, int n, int k, float alpha, const float *A, int lda, const float *B,
    int ldb, float beta, float *C, int ldc, cudaStream_t stream) {
  if (has_invalid_shape(m, n, k, lda, ldb, ldc)) {
    return cudaErrorInvalidValue;
  }
  if (m == 0 || n == 0) {
    return cudaSuccess;
  }
  if (C == nullptr || (k > 0 && (A == nullptr || B == nullptr))) {
    return cudaErrorInvalidDevicePointer;
  }

  constexpr int blockRows = 64;
  constexpr int blockCols = 128;
  constexpr int kTile = 8;
  constexpr int tileAStride = kTile + 1;
  constexpr int threadsPerBlock = 128;
  constexpr int sharedMemoryBytes =
      (blockRows * tileAStride + kTile * blockCols) * sizeof(float);
  const int tileRows = (m + blockRows - 1) / blockRows;
  const int tileCols = (n + blockCols - 1) / blockCols;
  const int grid = tileRows * tileCols;

  sgemm_tiled_64x128_8x8_paddedA_vectorizedC_kernel
      <<<grid, threadsPerBlock, sharedMemoryBytes, stream>>>(
          m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
  return cudaGetLastError();
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
