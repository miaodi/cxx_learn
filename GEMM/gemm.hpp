#pragma once
#include <algorithm>
#include <immintrin.h>
#include <vector>

#if defined(GEMM_ENABLE_PROFILING) && (defined(__GNUC__) || defined(__clang__))
#define GEMM_NOINLINE __attribute__((noinline))
#else
#define GEMM_NOINLINE
#endif

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
GEMM_NOINLINE
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
GEMM_NOINLINE
void PackRowMajorPanel(int rows, int cols, const T *src, int ldsrc, T *dst) {
  for (int row = 0; row < rows; ++row) {
    std::copy_n(src + row * ldsrc, cols, dst + row * cols);
  }
}

template <typename T>
GEMM_NOINLINE
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

template <typename T>
GEMM_NOINLINE
void gemm_block_kernel_packed_ab(int mc, int nc, int kc, T alpha,
                                 const T *Apack, const T *Bpack, T *C,
                                 int ldc) {
  for (int i = 0; i < mc; ++i) {
    const T *ApackRow = Apack + i * kc;
    for (int k = 0; k < kc; ++k) {
      const T aik = alpha * ApackRow[k];
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
      PackRowMajorPanel(kc, nc, B + kk * ldb + jj, ldb, Bpack.data());

      for (int ii = 0; ii < M; ii += BM) {
        const int mc = std::min(BM, M - ii);
        gemm_block_kernel_packed_b(mc, nc, kc, alpha, A + ii * lda + kk, lda,
                                   Bpack.data(), C + ii * ldc + jj, ldc);
      }
    }
  }
}

template <BetaMode betaMode, typename T, int BM, int BN, int BK>
void gemm_packed_ab_impl(int M, int N, int K, T alpha, const T *A, int lda,
                         const T *B, int ldb, T beta, T *C, int ldc) {
  ScaleC<betaMode>(M, N, beta, C, ldc);

  std::vector<T> Apack(BM * BK);
  std::vector<T> Bpack(BK * BN);

  for (int jj = 0; jj < N; jj += BN) {
    const int nc = std::min(BN, N - jj);
    for (int kk = 0; kk < K; kk += BK) {
      const int kc = std::min(BK, K - kk);
      PackRowMajorPanel(kc, nc, B + kk * ldb + jj, ldb, Bpack.data());

      for (int ii = 0; ii < M; ii += BM) {
        const int mc = std::min(BM, M - ii);
        PackRowMajorPanel(mc, kc, A + ii * lda + kk, lda, Apack.data());
        gemm_block_kernel_packed_ab(mc, nc, kc, alpha, Apack.data(),
                                    Bpack.data(), C + ii * ldc + jj, ldc);
      }
    }
  }
}

template <typename T, int BM, int BK>
GEMM_NOINLINE
void PrepackAPanels(int M, int K, const T *A, int lda, T *Apack) {
  constexpr int panelSize = BM * BK;
  const int numBlockM = (M + BM - 1) / BM;

  for (int kb = 0, kk = 0; kk < K; ++kb, kk += BK) {
    const int kc = std::min(BK, K - kk);
    for (int ib = 0, ii = 0; ii < M; ++ib, ii += BM) {
      const int mc = std::min(BM, M - ii);
      T *ApackPanel =
          Apack + (std::size_t(kb) * numBlockM + ib) * panelSize;
      PackRowMajorPanel(mc, kc, A + ii * lda + kk, lda, ApackPanel);
    }
  }
}

template <BetaMode betaMode, typename T, int BM, int BN, int BK>
void gemm_packed_ab_prepack_a_impl(int M, int N, int K, T alpha, const T *A,
                                   int lda, const T *B, int ldb, T beta, T *C,
                                   int ldc) {
  ScaleC<betaMode>(M, N, beta, C, ldc);

  constexpr int aPanelSize = BM * BK;
  const int numBlockM = (M + BM - 1) / BM;
  const int numBlockK = (K + BK - 1) / BK;
  std::vector<T> Apack(std::size_t(numBlockK) * numBlockM * aPanelSize);
  std::vector<T> Bpack(BK * BN);

  PrepackAPanels<T, BM, BK>(M, K, A, lda, Apack.data());

  for (int jj = 0; jj < N; jj += BN) {
    const int nc = std::min(BN, N - jj);
    for (int kb = 0, kk = 0; kk < K; ++kb, kk += BK) {
      const int kc = std::min(BK, K - kk);
      PackRowMajorPanel(kc, nc, B + kk * ldb + jj, ldb, Bpack.data());

      for (int ib = 0, ii = 0; ii < M; ++ib, ii += BM) {
        const int mc = std::min(BM, M - ii);
        const T *ApackPanel =
            Apack.data() + (std::size_t(kb) * numBlockM + ib) * aPanelSize;
        gemm_block_kernel_packed_ab(mc, nc, kc, alpha, ApackPanel,
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

// Matrix multiplication: C = alpha * A * B + beta * C
// Same interface as gemm_packed_b, but packs both A blocks and B panels.
template <typename T, int BM = 64, int BN = 64, int BK = 64>
void gemm_packed_ab(int M, int N, int K, T alpha, const T *A, int lda,
                    const T *B, int ldb, T beta, T *C, int ldc) {
  static_assert(BM > 0 && BN > 0 && BK > 0,
                "Block sizes must be positive");

  if (beta == T(0)) {
    detail::gemm_packed_ab_impl<detail::BetaMode::Zero, T, BM, BN, BK>(
        M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
    return;
  }

  if (beta == T(1)) {
    detail::gemm_packed_ab_impl<detail::BetaMode::One, T, BM, BN, BK>(
        M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
    return;
  }

  detail::gemm_packed_ab_impl<detail::BetaMode::General, T, BM, BN, BK>(
      M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
}

// Matrix multiplication: C = alpha * A * B + beta * C
// Same interface as gemm_packed_ab, but pre-packs all A block panels once.
template <typename T, int BM = 64, int BN = 64, int BK = 64>
void gemm_packed_ab_prepack_a(int M, int N, int K, T alpha, const T *A,
                              int lda, const T *B, int ldb, T beta, T *C,
                              int ldc) {
  static_assert(BM > 0 && BN > 0 && BK > 0,
                "Block sizes must be positive");

  if (beta == T(0)) {
    detail::gemm_packed_ab_prepack_a_impl<detail::BetaMode::Zero, T, BM, BN,
                                          BK>(M, N, K, alpha, A, lda, B, ldb,
                                              beta, C, ldc);
    return;
  }

  if (beta == T(1)) {
    detail::gemm_packed_ab_prepack_a_impl<detail::BetaMode::One, T, BM, BN,
                                          BK>(M, N, K, alpha, A, lda, B, ldb,
                                              beta, C, ldc);
    return;
  }

  detail::gemm_packed_ab_prepack_a_impl<detail::BetaMode::General, T, BM, BN,
                                        BK>(M, N, K, alpha, A, lda, B, ldb,
                                            beta, C, ldc);
}
} // namespace gemm
