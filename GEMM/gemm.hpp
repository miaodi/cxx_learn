#pragma once
#include <algorithm>
#include <immintrin.h>
#include <vector>
namespace gemm {

namespace detail {

enum class BetaMode { Zero, One, General };

template <BetaMode betaMode, typename T>
void ScaleC(int M, int N, T beta, T *C, int ldc) {
  if constexpr (betaMode == BetaMode::Zero) {
    (void)beta;
    for (int i = 0; i < M; ++i) {
      std::fill(C + i * ldc, C + i * ldc + N, T(0));
    }
  } else if constexpr (betaMode == BetaMode::One) {
    (void)M;
    (void)N;
    (void)beta;
    (void)C;
    (void)ldc;
  } else {
    for (int i = 0; i < M; ++i) {
      for (int j = 0; j < N; ++j) {
        C[i * ldc + j] *= beta;
      }
    }
  }
}

template <BetaMode betaMode, typename T>
void gemm_naive_impl(int M, int N, int K, T alpha, const T *A, int lda,
                     const T *B, int ldb, T beta, T *C, int ldc) {
  ScaleC<betaMode>(M, N, beta, C, ldc);

  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < N; ++j) {
      T sum = T(0);
      for (int k = 0; k < K; ++k) {
        sum += A[i * lda + k] * B[k * ldb + j];
      }
      C[i * ldc + j] += alpha * sum;
    }
  }
}

template <BetaMode betaMode, typename T>
void gemm_ikj_impl(int M, int N, int K, T alpha, const T *A, int lda,
                   const T *B, int ldb, T beta, T *C, int ldc) {
  ScaleC<betaMode>(M, N, beta, C, ldc);

  for (int i = 0; i < M; ++i) {
    for (int k = 0; k < K; ++k) {
      const T aik = alpha * A[i * lda + k];
      for (int j = 0; j < N; ++j) {
        C[i * ldc + j] += aik * B[k * ldb + j];
      }
    }
  }
}

template <typename T>
void gemm_block_kernel(int mc, int nc, int kc, T alpha, const T *A, int lda,
                       const T *B, int ldb, T *C, int ldc) {
  for (int i = 0; i < mc; ++i) {
    for (int k = 0; k < kc; ++k) {
      const T aik = alpha * A[i * lda + k];
      for (int j = 0; j < nc; ++j) {
        C[i * ldc + j] += aik * B[k * ldb + j];
      }
    }
  }
}

template <BetaMode betaMode, typename T, int BM, int BN, int BK>
void gemm_blocked_impl(int M, int N, int K, T alpha, const T *A, int lda,
                       const T *B, int ldb, T beta, T *C, int ldc) {
  ScaleC<betaMode>(M, N, beta, C, ldc);

  for (int ii = 0; ii < M; ii += BM) {
    const int mc = std::min(BM, M - ii);
    for (int jj = 0; jj < N; jj += BN) {
      const int nc = std::min(BN, N - jj);
      for (int kk = 0; kk < K; kk += BK) {
        const int kc = std::min(BK, K - kk);
        gemm_block_kernel(mc, nc, kc, alpha, A + ii * lda + kk, lda,
                          B + kk * ldb + jj, ldb, C + ii * ldc + jj, ldc);
      }
    }
  }
}

template <typename T>
void PackBPanel(int kc, int nc, const T *B, int ldb, T *Bpack) {
  for (int k = 0; k < kc; ++k) {
    std::copy_n(B + k * ldb, nc, Bpack + k * nc);
  }
}

template <typename T>
void gemm_block_kernel_packed_b(int mc, int nc, int kc, T alpha, const T *A,
                                int lda, const T *Bpack, T *C, int ldc) {
  for (int i = 0; i < mc; ++i) {
    for (int k = 0; k < kc; ++k) {
      const T aik = alpha * A[i * lda + k];
      const T *BpackRow = Bpack + k * nc;
      for (int j = 0; j < nc; ++j) {
        C[i * ldc + j] += aik * BpackRow[j];
      }
    }
  }
}

template <BetaMode betaMode, typename T, int BM, int BN, int BK>
void gemm_packed_b_impl(int M, int N, int K, T alpha, const T *A, int lda,
                        const T *B, int ldb, T beta, T *C, int ldc) {
  ScaleC<betaMode>(M, N, beta, C, ldc);

  std::vector<T> Bpack(BK * BN);

  for (int jj = 0; jj < N; jj += BN) {
    const int nc = std::min(BN, N - jj);
    for (int kk = 0; kk < K; kk += BK) {
      const int kc = std::min(BK, K - kk);
      PackBPanel(kc, nc, B + kk * ldb + jj, ldb, Bpack.data());

      for (int ii = 0; ii < M; ii += BM) {
        const int mc = std::min(BM, M - ii);
        gemm_block_kernel_packed_b(mc, nc, kc, alpha, A + ii * lda + kk, lda,
                                   Bpack.data(), C + ii * ldc + jj, ldc);
      }
    }
  }
}

} // namespace detail

// Matrix multiplication: C = alpha * A * B + beta * C
// A is M x K with leading dimension lda.
// B is K x N with leading dimension ldb.
// C is M x N with leading dimension ldc.
// Assumes row-major storage and no transposition.
template <typename T>
void gemm_naive(int M, int N, int K, T alpha, const T *A, int lda,
                const T *B, int ldb, T beta, T *C, int ldc) {
  if (beta == T(0)) {
    detail::gemm_naive_impl<detail::BetaMode::Zero>(
        M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
    return;
  }

  if (beta == T(1)) {
    detail::gemm_naive_impl<detail::BetaMode::One>(
        M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
    return;
  }

  detail::gemm_naive_impl<detail::BetaMode::General>(
      M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
}

// Matrix multiplication: C = alpha * A * B + beta * C
// Same interface as gemm_naive, but uses i-k-j loop order.
template <typename T>
void gemm_ikj(int M, int N, int K, T alpha, const T *A, int lda, const T *B,
              int ldb, T beta, T *C, int ldc) {
  if (beta == T(0)) {
    detail::gemm_ikj_impl<detail::BetaMode::Zero>(
        M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
    return;
  }

  if (beta == T(1)) {
    detail::gemm_ikj_impl<detail::BetaMode::One>(
        M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
    return;
  }

  detail::gemm_ikj_impl<detail::BetaMode::General>(
      M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
}

// Matrix multiplication: C = alpha * A * B + beta * C
// Same interface as gemm_naive, but computes cache-sized output blocks.
template <typename T, int BM = 64, int BN = 64, int BK = 64>
void gemm_blocked(int M, int N, int K, T alpha, const T *A, int lda,
                  const T *B, int ldb, T beta, T *C, int ldc) {
  static_assert(BM > 0 && BN > 0 && BK > 0,
                "Block sizes must be positive");

  if (beta == T(0)) {
    detail::gemm_blocked_impl<detail::BetaMode::Zero, T, BM, BN, BK>(
        M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
    return;
  }

  if (beta == T(1)) {
    detail::gemm_blocked_impl<detail::BetaMode::One, T, BM, BN, BK>(
        M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
    return;
  }

  detail::gemm_blocked_impl<detail::BetaMode::General, T, BM, BN, BK>(
      M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
}

// Matrix multiplication: C = alpha * A * B + beta * C
// Same interface as gemm_blocked, but packs each B panel before reuse.
template <typename T, int BM = 64, int BN = 64, int BK = 64>
void gemm_packed_b(int M, int N, int K, T alpha, const T *A, int lda,
                   const T *B, int ldb, T beta, T *C, int ldc) {
  static_assert(BM > 0 && BN > 0 && BK > 0,
                "Block sizes must be positive");

  if (beta == T(0)) {
    detail::gemm_packed_b_impl<detail::BetaMode::Zero, T, BM, BN, BK>(
        M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
    return;
  }

  if (beta == T(1)) {
    detail::gemm_packed_b_impl<detail::BetaMode::One, T, BM, BN, BK>(
        M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
    return;
  }

  detail::gemm_packed_b_impl<detail::BetaMode::General, T, BM, BN, BK>(
      M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
}

template <typename T>
void MatMatTransMul(const T *A, const T *B, T *C, int M, int N, int K) {
  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < N; ++j) {
      C[i * N + j] = 0.0f;
      for (int k = 0; k < K; ++k) {
        C[i * N + j] += A[i * K + k] * B[j * K + k];
      }
    }
  }
}

// Tiled matrix multiplication with transposed B: C = A * B^T
// A is M x K, B is N x K (transposed), C is M x N
// Assumes row-major storage
template <typename T, int TileSize = 16>
void TiledMatMatTransMul(const T *A, const T *B, T *C, int M, int N, int K) {
  for (int i = 0; i < M; i += TileSize) {
    for (int j = 0; j < N; j += TileSize) {
      for (int k = 0; k < K; k += TileSize) {
        for (int ii = i; ii < std::min(i + TileSize, M); ++ii) {
          for (int jj = j; jj < std::min(j + TileSize, N); ++jj) {
            T sum = 0;
            for (int kk = k; kk < std::min(k + TileSize, K); ++kk) {
              sum += A[ii * K + kk] * B[jj * K + kk];
            }
            C[ii * N + jj] += sum;
          }
        }
      }
    }
  }
}

// Tiled matrix multiplication: C = A * B
// A is M x K, B is K x N, C is M x N
// Assumes row-major storage
template <typename T, int TileSize = 16>
void TiledMatMatMulInternalTrans(const T *A, const T *B, T *C, int M, int N,
                                 int K) {
  std::vector<T> B_transposed(N * K);
  for (int i = 0; i < K; ++i) {
    for (int j = 0; j < N; ++j) {
      B_transposed[j * K + i] = B[i * N + j];
    }
  }
  TiledMatMatTransMul<T, TileSize>(A, B_transposed.data(), C, M, N, K);
}

template <typename T, int TileSize = 16>
void ConvertToTiledRowMajorPadded(const T *input, T *output, int M, int N) {
  const int numTilesM = (M + TileSize - 1) / TileSize;
  const int numTilesN = (N + TileSize - 1) / TileSize;
  const int tiles = numTilesM * numTilesN;
  const int perTile = TileSize * TileSize;

  std::fill(output, output + tiles * perTile, T(0));

  for (int ti = 0; ti < numTilesM; ++ti) {
    const int rowStart = ti * TileSize;
    const int rowEnd = std::min(rowStart + TileSize, M);
    for (int tj = 0; tj < numTilesN; ++tj) {
      const int colStart = tj * TileSize;
      const int colEnd = std::min(colStart + TileSize, N);
      T *tilePtr = output + (ti * numTilesN + tj) * perTile;
      for (int i = rowStart; i < rowEnd; ++i) {
        const T *rowPtr = input + i * N + colStart;
        std::copy(rowPtr, rowPtr + (colEnd - colStart),
                  tilePtr + (i - rowStart) * TileSize);
      }
    }
  }
}

template <typename T, int TileSize = 16>
void ConvertToTiledRowMajorTranspadded(const T *input, T *output, int M,
                                       int N) {

  const int outM = N, outN = M;
  const int numTilesM = (outM + TileSize - 1) / TileSize; // along N
  const int numTilesN = (outN + TileSize - 1) / TileSize; // along M
  const size_t perTile = size_t(TileSize) * TileSize;
  std::fill(output, output + size_t(numTilesM) * numTilesN * perTile, T(0));
  for (int ti = 0; ti < numTilesM; ++ti) {
    const int rowStart = ti * TileSize;
    const int rowEnd = std::min(rowStart + TileSize, outM);
    for (int tj = 0; tj < numTilesN; ++tj) {
      const int colStart = tj * TileSize;
      const int colEnd = std::min(colStart + TileSize, outN);
      T *tilePtr = output + (ti * numTilesN + tj) * perTile;
      for (int i = rowStart; i < rowEnd; ++i) {
        T *rowPtr = tilePtr + (i - rowStart) * TileSize;
        for (int j = colStart; j < colEnd; ++j) {
          rowPtr[j - colStart] = input[j * N + i];
        }
      }
    }
  }
}

template <typename T, int TileSize = 16>
void ConvertFromTiledRowMajorPadded(const T *inputTiled, T *output, int M,
                                    int N) {
  const int numTilesM = (M + TileSize - 1) / TileSize;
  const int numTilesN = (N + TileSize - 1) / TileSize;
  const int perTile = TileSize * TileSize;

  for (int ti = 0; ti < numTilesM; ++ti) {
    const int row0 = ti * TileSize;
    const int rows = std::min(TileSize, M - row0);
    for (int tj = 0; tj < numTilesN; ++tj) {
      const int col0 = tj * TileSize;
      const int cols = std::min(TileSize, N - col0);

      const T *tileBase = inputTiled + (ti * numTilesN + tj) * perTile;

      for (int ii = 0; ii < rows; ++ii) {
        const T *src = tileBase + ii * TileSize;
        T *dst = output + (row0 + ii) * N + col0;
        std::copy(src, src + cols, dst);
      }
    }
  }
}

template <typename T, int TileSize = 16>
inline void Kernel_TileTileTrans(const T *__restrict__ Atile,
                                 const T *__restrict__ BTtile,
                                 T *__restrict__ Ctile) {
  for (int i = 0; i < TileSize; ++i) {
    for (int j = 0; j < TileSize; ++j) {
      T sum = 0;
      for (int k = 0; k < TileSize; ++k) {
        sum += Atile[i * TileSize + k] * BTtile[j * TileSize + k];
      }
      Ctile[i * TileSize + j] += sum;
    }
  }
}

template <typename T, int TileSize = 16>
void GemmTiledATiledBTToTiledC(const T *__restrict__ Atiles,
                               const T *__restrict__ BTiles,
                               T *__restrict__ Ctiles, int M, int N, int K) {
  const int ts = TileSize;
  const int nTM = (M + ts - 1) / ts; // tiles along M
  const int nTN = (N + ts - 1) / ts; // tiles along N
  const int nTK = (K + ts - 1) / ts; // tiles along K
  const std::size_t perTile = std::size_t(ts) * ts;

  auto A_tile_ptr = [&](int ti, int tk) {
    return Atiles + (std::size_t(ti) * nTK + tk) * perTile;
  };
  auto BT_tile_ptr = [&](int tj, int tk) {
    // BT is N×K tiled → tj over N, tk over K
    return BTiles + (std::size_t(tj) * nTK + tk) * perTile;
  };
  auto C_tile_ptr = [&](int ti, int tj) {
    return Ctiles + (std::size_t(ti) * nTN + tj) * perTile;
  };

  for (int ti = 0; ti < nTM; ++ti) {
    for (int tj = 0; tj < nTN; ++tj) {
      T *Ctile = C_tile_ptr(ti, tj);
      std::fill(Ctile, Ctile + perTile, T{0});
      for (int tk = 0; tk < nTK; ++tk) {
        const T *Atile = A_tile_ptr(ti, tk);
        const T *BTtile = BT_tile_ptr(tj, tk);
        Kernel_TileTileTrans<T, TileSize>(Atile, BTtile, Ctile);
      }
    }
  }
}

template <typename T, int TileSize = 4>
void TiledMatMatMulInternalTransTiledPadded(const T *A, const T *B, T *C, int M,
                                            int N, int K) {
  const int numTilesM = (M + TileSize - 1) / TileSize;
  const int numTilesN = (N + TileSize - 1) / TileSize;
  const int numTilesK = (K + TileSize - 1) / TileSize;
  constexpr int tileSize = TileSize * TileSize;
  std::vector<T> A_tiled(numTilesM * numTilesK * tileSize);
  std::vector<T> B_tiled(numTilesN * numTilesK * tileSize);
  std::vector<T> C_tiled(numTilesM * numTilesN * tileSize, 0);

  // Convert A to tiled layout
  ConvertToTiledRowMajorPadded<T, TileSize>(A, A_tiled.data(), M, K);
  // Convert B to tiled layout
  ConvertToTiledRowMajorTranspadded<T, TileSize>(B, B_tiled.data(), K, N);
  // Perform tiled multiplication

  GemmTiledATiledBTToTiledC<T, TileSize>(A_tiled.data(), B_tiled.data(),
                                         C_tiled.data(), M, N, K);
  // Convert C back to normal layout
  ConvertFromTiledRowMajorPadded<T, TileSize>(C_tiled.data(), C, M, N);
}

#if defined(AVX512_SUPPORTED) && defined(FMA_SUPPORTED)

template <typename T, int TileSize = 16>
inline void Kernel_TileTile(const T *__restrict__ Atile,
                            const T *__restrict__ BTtile,
                            T *__restrict__ Ctile) {
  for (int i = 0; i < TileSize; ++i) {
    for (int k = 0; k < TileSize; ++k) {
      const auto aik = Atile[i * TileSize + k];
      if constexpr (std::is_same_v<T, float>) {
        static_assert(TileSize % 16 == 0,
                      "TileSize must be a multiple of 16 for AVX512");
        for (int j = 0; j < TileSize; j += 16) {
          __m512 c_vec = _mm512_loadu_ps(&Ctile[i * TileSize + j]);
          __m512 b_vec = _mm512_loadu_ps(&BTtile[k * TileSize + j]);
          __m512 a_vec = _mm512_set1_ps(aik);
          c_vec = _mm512_fmadd_ps(a_vec, b_vec, c_vec);
          _mm512_storeu_ps(&Ctile[i * TileSize + j], c_vec);
        }
      } else if constexpr (std::is_same_v<T, double>) {
        static_assert(TileSize % 8 == 0,
                      "TileSize must be a multiple of 8 for AVX512");
        for (int j = 0; j < TileSize; j += 8) {
          __m512d c_vec = _mm512_loadu_pd(&Ctile[i * TileSize + j]);
          __m512d b_vec = _mm512_loadu_pd(&BTtile[k * TileSize + j]);
          __m512d a_vec = _mm512_set1_pd(aik);
          c_vec = _mm512_fmadd_pd(a_vec, b_vec, c_vec);
          _mm512_storeu_pd(&Ctile[i * TileSize + j], c_vec);
        }
      } else {
        for (int j = 0; j < TileSize; ++j) {
          Ctile[i * TileSize + j] += aik * BTtile[k * TileSize + j];
        }
      }
    }
  }
}

#elif defined(AVX2_SUPPORTED) && defined(FMA_SUPPORTED)

template <typename T, int TileSize = 16>
inline void Kernel_TileTile(const T *__restrict__ Atile,
                            const T *__restrict__ BTtile,
                            T *__restrict__ Ctile) {
  for (int i = 0; i < TileSize; ++i) {
    for (int k = 0; k < TileSize; ++k) {
      const auto aik = Atile[i * TileSize + k];
      if constexpr (std::is_same_v<T, float>) {
        static_assert(TileSize % 8 == 0,
                      "TileSize must be a multiple of 8 for AVX2");
        for (int j = 0; j < TileSize; j += 8) {
          __m256 c_vec = _mm256_loadu_ps(&Ctile[i * TileSize + j]);
          __m256 b_vec = _mm256_loadu_ps(&BTtile[k * TileSize + j]);
          __m256 a_vec = _mm256_set1_ps(aik);
          c_vec = _mm256_fmadd_ps(a_vec, b_vec, c_vec);
          _mm256_storeu_ps(&Ctile[i * TileSize + j], c_vec);
        }
      } else if constexpr (std::is_same_v<T, double>) {
        static_assert(TileSize % 4 == 0,
                      "TileSize must be a multiple of 4 for AVX2");
        for (int j = 0; j < TileSize; j += 4) {
          __m256d c_vec = _mm256_loadu_pd(&Ctile[i * TileSize + j]);
          __m256d b_vec = _mm256_loadu_pd(&BTtile[k * TileSize + j]);
          __m256d a_vec = _mm256_set1_pd(aik);
          c_vec = _mm256_fmadd_pd(a_vec, b_vec, c_vec);
          _mm256_storeu_pd(&Ctile[i * TileSize + j], c_vec);
        }
      } else {
        for (int j = 0; j < TileSize; ++j) {
          Ctile[i * TileSize + j] += aik * BTtile[k * TileSize + j];
        }
      }
    }
  }
}

#else
template <typename T, int TileSize = 16>
inline void Kernel_TileTile(const T *__restrict__ Atile,
                            const T *__restrict__ BTtile,
                            T *__restrict__ Ctile) {
  for (int i = 0; i < TileSize; ++i) {
    for (int k = 0; k < TileSize; ++k) {
      const auto aik = Atile[i * TileSize + k];
      for (int j = 0; j < TileSize; ++j) {
        Ctile[i * TileSize + j] += aik * BTtile[k * TileSize + j];
      }
    }
  }
}

#endif

template <typename T, int TileSize = 16>
void GemmTiledATiledBToTiledC(const T *__restrict__ Atiles,
                              const T *__restrict__ BTiles,
                              T *__restrict__ Ctiles, int M, int N, int K) {
  const int ts = TileSize;
  const int nTM = (M + ts - 1) / ts; // tiles along M
  const int nTN = (N + ts - 1) / ts; // tiles along N
  const int nTK = (K + ts - 1) / ts; // tiles along K
  const std::size_t perTile = std::size_t(ts) * ts;

  auto A_tile_ptr = [&](int ti, int tk) {
    return Atiles + (std::size_t(ti) * nTK + tk) * perTile;
  };
  auto B_tile_ptr = [&](int tk, int tj) {
    // B is K×N tiled → tk over K, tj over N
    return BTiles + (std::size_t(tk) * nTN + tj) * perTile;
  };
  auto C_tile_ptr = [&](int ti, int tj) {
    return Ctiles + (std::size_t(ti) * nTN + tj) * perTile;
  };

  for (int ti = 0; ti < nTM; ++ti) {
    for (int tj = 0; tj < nTN; ++tj) {
      T *Ctile = C_tile_ptr(ti, tj);
      std::fill(Ctile, Ctile + perTile, T{0});
      for (int tk = 0; tk < nTK; ++tk) {
        const T *Atile = A_tile_ptr(ti, tk);
        const T *BTtile = B_tile_ptr(tk, tj);
        Kernel_TileTile<T, TileSize>(Atile, BTtile, Ctile);
      }
    }
  }
}

template <typename T, int TileSize = 16>
void TiledMatMatMulInternalTiledPadded(const T *A, const T *B, T *C, int M,
                                       int N, int K) {
  const int numTilesM = (M + TileSize - 1) / TileSize;
  const int numTilesN = (N + TileSize - 1) / TileSize;
  const int numTilesK = (K + TileSize - 1) / TileSize;
  constexpr int tileSize = TileSize * TileSize;
  std::vector<T> A_tiled(numTilesM * numTilesK * tileSize);
  std::vector<T> B_tiled(numTilesN * numTilesK * tileSize);
  std::vector<T> C_tiled(numTilesM * numTilesN * tileSize);

  // Convert A to tiled layout
  ConvertToTiledRowMajorPadded<T, TileSize>(A, A_tiled.data(), M, K);
  // Convert B to tiled layout
  ConvertToTiledRowMajorPadded<T, TileSize>(B, B_tiled.data(), K, N);
  // Perform tiled multiplication

  GemmTiledATiledBToTiledC<T, TileSize>(A_tiled.data(), B_tiled.data(),
                                        C_tiled.data(), M, N, K);
  // Convert C back to normal layout
  ConvertFromTiledRowMajorPadded<T, TileSize>(C_tiled.data(), C, M, N);
}
} // namespace gemm
