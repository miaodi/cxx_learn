#pragma once
#include <algorithm>
#include <vector>

#if defined(__AVX2__)
#include <immintrin.h>
#endif

namespace matrix_transpose {

// ============================================================================
// Kernel Selection Tags
// ============================================================================

struct ScalarKernel {};
struct AVX2Kernel {};
struct AVX512Kernel {};

// ============================================================================
// Naive Matrix Transpose
// ============================================================================

/**
 * @brief Naive row-major to row-major transpose
 * Input: M×N matrix in row-major order
 * Output: N×M transposed matrix in row-major order
 * 
 * Performance: Poor cache locality - writes are strided
 */
template <typename T>
void NaiveTranspose(const T *input, T *output, int M, int N) {
  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < N; ++j) {
      output[j * M + i] = input[i * N + j];
    }
  }
}

// ============================================================================
// Tiled Matrix Transpose Kernels
// ============================================================================

/**
 * @brief Scalar tiled transpose kernel for a single square tile
 * Transposes a full TileSize×TileSize block from input to output
 * Assumes the tile is completely within matrix bounds
 */
template <typename T, int TileSize>
void TiledTransposeKernel(const T *input, T *output,
                          int inputRow, int inputCol,
                          int inputStride, int outputStride,
                          ScalarKernel) {
  for (int i = 0; i < TileSize; ++i) {
    for (int j = 0; j < TileSize; ++j) {
      output[(inputCol + j) * outputStride + (inputRow + i)] = 
          input[(inputRow + i) * inputStride + (inputCol + j)];
    }
  }
}

// Wrapper to call scalar kernel without specifying TileSize explicitly
template <int TileSize, typename T>
void TiledTransposeKernel(const T *input, T *output,
                          int inputRow, int inputCol,
                          int inputStride, int outputStride,
                          ScalarKernel tag) {
  TiledTransposeKernel<T, TileSize>(input, output, inputRow, inputCol, 
                                    inputStride, outputStride, tag);
}

#if defined(__AVX2__)
/**
 * @brief AVX2 tiled transpose kernel for float (8x8 tiles)
 * Uses 256-bit SIMD instructions for 8 floats at a time
 */
template <int TileSize>
inline void TiledTransposeKernel(const float *input, float *output,
                                 int inputRow, int inputCol,
                                 int inputStride, int outputStride,
                                 AVX2Kernel) {
  static_assert(TileSize == 8, "AVX2 float kernel requires 8x8 tiles");
  // Load 8x8 tile
  __m256 row[8];
  for (int i = 0; i < 8; ++i) {
    row[i] = _mm256_loadu_ps(&input[(inputRow + i) * inputStride + inputCol]);
  }
  
  // Transpose 8x8 using AVX2 shuffle instructions
  // Step 1: Transpose 4x4 blocks
  __m256 tmp[8];
  tmp[0] = _mm256_unpacklo_ps(row[0], row[1]);
  tmp[1] = _mm256_unpackhi_ps(row[0], row[1]);
  tmp[2] = _mm256_unpacklo_ps(row[2], row[3]);
  tmp[3] = _mm256_unpackhi_ps(row[2], row[3]);
  tmp[4] = _mm256_unpacklo_ps(row[4], row[5]);
  tmp[5] = _mm256_unpackhi_ps(row[4], row[5]);
  tmp[6] = _mm256_unpacklo_ps(row[6], row[7]);
  tmp[7] = _mm256_unpackhi_ps(row[6], row[7]);
  
  // Step 2: Shuffle 64-bit elements
  __m256 tmp2[8];
  tmp2[0] = _mm256_shuffle_ps(tmp[0], tmp[2], 0x44);
  tmp2[1] = _mm256_shuffle_ps(tmp[0], tmp[2], 0xEE);
  tmp2[2] = _mm256_shuffle_ps(tmp[1], tmp[3], 0x44);
  tmp2[3] = _mm256_shuffle_ps(tmp[1], tmp[3], 0xEE);
  tmp2[4] = _mm256_shuffle_ps(tmp[4], tmp[6], 0x44);
  tmp2[5] = _mm256_shuffle_ps(tmp[4], tmp[6], 0xEE);
  tmp2[6] = _mm256_shuffle_ps(tmp[5], tmp[7], 0x44);
  tmp2[7] = _mm256_shuffle_ps(tmp[5], tmp[7], 0xEE);
  
  // Step 3: Permute 128-bit lanes
  row[0] = _mm256_permute2f128_ps(tmp2[0], tmp2[4], 0x20);
  row[1] = _mm256_permute2f128_ps(tmp2[1], tmp2[5], 0x20);
  row[2] = _mm256_permute2f128_ps(tmp2[2], tmp2[6], 0x20);
  row[3] = _mm256_permute2f128_ps(tmp2[3], tmp2[7], 0x20);
  row[4] = _mm256_permute2f128_ps(tmp2[0], tmp2[4], 0x31);
  row[5] = _mm256_permute2f128_ps(tmp2[1], tmp2[5], 0x31);
  row[6] = _mm256_permute2f128_ps(tmp2[2], tmp2[6], 0x31);
  row[7] = _mm256_permute2f128_ps(tmp2[3], tmp2[7], 0x31);
  
  // Store transposed 8x8 tile
  for (int i = 0; i < 8; ++i) {
    _mm256_storeu_ps(&output[(inputCol + i) * outputStride + inputRow], row[i]);
  }
}

/**
 * @brief AVX2 tiled transpose kernel for double (4x4 tiles)
 * Uses 256-bit SIMD instructions for 4 doubles at a time
 */
template <int TileSize>
inline void TiledTransposeKernel(const double *input, double *output,
                                 int inputRow, int inputCol,
                                 int inputStride, int outputStride,
                                 AVX2Kernel) {
  static_assert(TileSize == 4, "AVX2 double kernel requires 4x4 tiles");
  // Load 4x4 tile
  __m256d row[4];
  for (int i = 0; i < 4; ++i) {
    row[i] = _mm256_loadu_pd(&input[(inputRow + i) * inputStride + inputCol]);
  }
  
  // Transpose 4x4
  __m256d tmp[4];
  tmp[0] = _mm256_unpacklo_pd(row[0], row[1]);
  tmp[1] = _mm256_unpackhi_pd(row[0], row[1]);
  tmp[2] = _mm256_unpacklo_pd(row[2], row[3]);
  tmp[3] = _mm256_unpackhi_pd(row[2], row[3]);
  
  row[0] = _mm256_permute2f128_pd(tmp[0], tmp[2], 0x20);
  row[1] = _mm256_permute2f128_pd(tmp[1], tmp[3], 0x20);
  row[2] = _mm256_permute2f128_pd(tmp[0], tmp[2], 0x31);
  row[3] = _mm256_permute2f128_pd(tmp[1], tmp[3], 0x31);
  
  // Store transposed 4x4 tile
  for (int i = 0; i < 4; ++i) {
    _mm256_storeu_pd(&output[(inputCol + i) * outputStride + inputRow], row[i]);
  }
}
#endif

#if defined(__AVX512F__)
/**
 * @brief AVX512 tiled transpose kernel for float (16x16 tiles)
 * Uses 512-bit SIMD instructions for 16 floats at a time
 */
template <int TileSize>
inline void TiledTransposeKernel(const float *input, float *output,
                                 int inputRow, int inputCol,
                                 int inputStride, int outputStride,
                                 AVX512Kernel) {
  static_assert(TileSize == 16, "AVX512 float kernel requires 16x16 tiles");
  // Load 16x16 tile
  __m512 row[16];
  for (int i = 0; i < 16; ++i) {
    row[i] = _mm512_loadu_ps(&input[(inputRow + i) * inputStride + inputCol]);
  }
  
  // Transpose using AVX512 permutations
  // This is a simplified version - full 16x16 transpose is complex
  // For production, consider using a more optimized approach
  __m512 tmp[16];
  
  // Step 1: Unpack pairs
  for (int i = 0; i < 16; i += 2) {
    tmp[i] = _mm512_unpacklo_ps(row[i], row[i + 1]);
    tmp[i + 1] = _mm512_unpackhi_ps(row[i], row[i + 1]);
  }
  
  // Step 2: Shuffle 4-element groups
  for (int i = 0; i < 16; i += 4) {
    row[i] = _mm512_shuffle_ps(tmp[i], tmp[i + 2], 0x44);
    row[i + 1] = _mm512_shuffle_ps(tmp[i], tmp[i + 2], 0xEE);
    row[i + 2] = _mm512_shuffle_ps(tmp[i + 1], tmp[i + 3], 0x44);
    row[i + 3] = _mm512_shuffle_ps(tmp[i + 1], tmp[i + 3], 0xEE);
  }
  
  // Step 3: Permute 128-bit lanes
  const __m512i idx1 = _mm512_setr_epi32(0, 1, 2, 3, 16, 17, 18, 19, 8, 9, 10, 11, 24, 25, 26, 27);
  const __m512i idx2 = _mm512_setr_epi32(4, 5, 6, 7, 20, 21, 22, 23, 12, 13, 14, 15, 28, 29, 30, 31);
  
  for (int i = 0; i < 8; ++i) {
    tmp[i] = _mm512_permutex2var_ps(row[i], idx1, row[i + 8]);
    tmp[i + 8] = _mm512_permutex2var_ps(row[i], idx2, row[i + 8]);
  }
  
  // Final permutation across 256-bit lanes
  for (int i = 0; i < 16; ++i) {
    row[i] = _mm512_shuffle_f32x4(tmp[i], tmp[i], 0xD8);
  }
  
  // Store transposed 16x16 tile
  for (int i = 0; i < 16; ++i) {
    _mm512_storeu_ps(&output[(inputCol + i) * outputStride + inputRow], row[i]);
  }
}

/**
 * @brief AVX512 tiled transpose kernel for double (8x8 tiles)
 * Uses 512-bit SIMD instructions for 8 doubles at a time
 */
template <int TileSize>
inline void TiledTransposeKernel(const double *input, double *output,
                                 int inputRow, int inputCol,
                                 int inputStride, int outputStride,
                                 AVX512Kernel) {
  static_assert(TileSize == 8, "AVX512 double kernel requires 8x8 tiles");
  // Load 8x8 tile
  __m512d row[8];
  for (int i = 0; i < 8; ++i) {
    row[i] = _mm512_loadu_pd(&input[(inputRow + i) * inputStride + inputCol]);
  }
  
  // Transpose 8x8
  __m512d tmp[8];
  tmp[0] = _mm512_unpacklo_pd(row[0], row[1]);
  tmp[1] = _mm512_unpackhi_pd(row[0], row[1]);
  tmp[2] = _mm512_unpacklo_pd(row[2], row[3]);
  tmp[3] = _mm512_unpackhi_pd(row[2], row[3]);
  tmp[4] = _mm512_unpacklo_pd(row[4], row[5]);
  tmp[5] = _mm512_unpackhi_pd(row[4], row[5]);
  tmp[6] = _mm512_unpacklo_pd(row[6], row[7]);
  tmp[7] = _mm512_unpackhi_pd(row[6], row[7]);
  
  // Shuffle 128-bit elements
  row[0] = _mm512_shuffle_f64x2(tmp[0], tmp[2], 0x44);
  row[1] = _mm512_shuffle_f64x2(tmp[1], tmp[3], 0x44);
  row[2] = _mm512_shuffle_f64x2(tmp[0], tmp[2], 0xEE);
  row[3] = _mm512_shuffle_f64x2(tmp[1], tmp[3], 0xEE);
  row[4] = _mm512_shuffle_f64x2(tmp[4], tmp[6], 0x44);
  row[5] = _mm512_shuffle_f64x2(tmp[5], tmp[7], 0x44);
  row[6] = _mm512_shuffle_f64x2(tmp[4], tmp[6], 0xEE);
  row[7] = _mm512_shuffle_f64x2(tmp[5], tmp[7], 0xEE);
  
  // Final permutation
  tmp[0] = _mm512_shuffle_f64x2(row[0], row[4], 0x88);
  tmp[1] = _mm512_shuffle_f64x2(row[1], row[5], 0x88);
  tmp[2] = _mm512_shuffle_f64x2(row[2], row[6], 0x88);
  tmp[3] = _mm512_shuffle_f64x2(row[3], row[7], 0x88);
  tmp[4] = _mm512_shuffle_f64x2(row[0], row[4], 0xDD);
  tmp[5] = _mm512_shuffle_f64x2(row[1], row[5], 0xDD);
  tmp[6] = _mm512_shuffle_f64x2(row[2], row[6], 0xDD);
  tmp[7] = _mm512_shuffle_f64x2(row[3], row[7], 0xDD);
  
  // Store transposed 8x8 tile
  for (int i = 0; i < 8; ++i) {
    _mm512_storeu_pd(&output[(inputCol + i) * outputStride + inputRow], tmp[i]);
  }
}
#endif

/**
 * @brief Tiled matrix transpose with cache-friendly access pattern
 * Processes matrix in tiles to improve cache locality
 * Handles boundary elements separately for partial tiles
 * 
 * @param TileSize Size of square tiles (typically 16, 32, or 64)
 * @param KernelTag Kernel implementation to use (ScalarKernel, AVX2Kernel, AVX512Kernel)
 */
template <typename T, int TileSize = 32, typename KernelTag = ScalarKernel>
void TiledTranspose(const T *input, T *output, int M, int N) {
  // Process complete tiles
  const int completeRowTiles = M / TileSize;
  const int completeColTiles = N / TileSize;
  
  for (int i = 0; i < completeRowTiles; ++i) {
    for (int j = 0; j < completeColTiles; ++j) {
      TiledTransposeKernel<TileSize>(input, output, 
                                     i * TileSize, j * TileSize, 
                                     N, M, KernelTag{});
    }
  }
  
  // Handle boundary regions with naive transpose
  const int completeRows = completeRowTiles * TileSize;
  const int completeCols = completeColTiles * TileSize;
  
  // Right edge (partial columns, full rows)
  if (completeCols < N) {
    for (int i = 0; i < completeRows; ++i) {
      for (int j = completeCols; j < N; ++j) {
        output[j * M + i] = input[i * N + j];
      }
    }
  }
  
  // Bottom edge (full columns, partial rows)  
  if (completeRows < M) {
    for (int i = completeRows; i < M; ++i) {
      for (int j = 0; j < completeCols; ++j) {
        output[j * M + i] = input[i * N + j];
      }
    }
  }
  
  // Bottom-right corner (both partial)
  if (completeRows < M && completeCols < N) {
    for (int i = completeRows; i < M; ++i) {
      for (int j = completeCols; j < N; ++j) {
        output[j * M + i] = input[i * N + j];
      }
    }
  }
}

// ============================================================================
// Cache-Oblivious Transpose (Recursive)
// ============================================================================

/**
 * @brief Cache-oblivious recursive transpose
 * Automatically adapts to cache hierarchy without tuning parameters
 */
template <typename T>
void CacheObliviousTransposeImpl(const T *input, T *output, 
                                  int M, int N, 
                                  int rowOffset, int colOffset,
                                  int inputStride, int outputStride) {
  constexpr int THRESHOLD = 16;
  
  if (M <= THRESHOLD && N <= THRESHOLD) {
    // Base case: transpose small block directly
    for (int i = 0; i < M; ++i) {
      for (int j = 0; j < N; ++j) {
        output[(colOffset + j) * outputStride + (rowOffset + i)] = 
            input[(rowOffset + i) * inputStride + (colOffset + j)];
      }
    }
  } else if (M >= N) {
    // Divide along M
    int M1 = M / 2;
    int M2 = M - M1;
    CacheObliviousTransposeImpl(input, output, M1, N, rowOffset, colOffset, inputStride, outputStride);
    CacheObliviousTransposeImpl(input, output, M2, N, rowOffset + M1, colOffset, inputStride, outputStride);
  } else {
    // Divide along N
    int N1 = N / 2;
    int N2 = N - N1;
    CacheObliviousTransposeImpl(input, output, M, N1, rowOffset, colOffset, inputStride, outputStride);
    CacheObliviousTransposeImpl(input, output, M, N2, rowOffset, colOffset + N1, inputStride, outputStride);
  }
}

template <typename T>
void CacheObliviousTranspose(const T *input, T *output, int M, int N) {
  CacheObliviousTransposeImpl(input, output, M, N, 0, 0, N, M);
}

// ============================================================================
// Helper Functions
// ============================================================================

/**
 * @brief Verify transpose correctness
 */
template <typename T>
bool VerifyTranspose(const T *input, const T *output, int M, int N, T tolerance = T(1e-5)) {
  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < N; ++j) {
      T expected = input[i * N + j];
      T actual = output[j * M + i];
      if (std::abs(expected - actual) > tolerance) {
        return false;
      }
    }
  }
  return true;
}

} // namespace matrix_transpose
