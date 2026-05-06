#pragma once
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <immintrin.h>
#include <limits>
#include <new>
#include <type_traits>
#include <vector>

#if defined(GEMM_ENABLE_PROFILING) && (defined(__GNUC__) || defined(__clang__))
#define GEMM_NOINLINE __attribute__((noinline))
#else
#define GEMM_NOINLINE
#endif

#if defined(__GNUC__) || defined(__clang__)
#define GEMM_TARGET_AVX2_FMA __attribute__((target("avx2,fma")))
#else
#define GEMM_TARGET_AVX2_FMA
#endif

// #define GEMM_DISABLE_EXPLICIT_FMA

namespace gemm {

namespace detail {

enum class BetaMode { Zero, One, General };

template <typename T, std::size_t Alignment> class AlignedAllocator {
public:
  using value_type = T;

  static_assert(Alignment >= alignof(T),
                "Alignment must satisfy value_type alignment");
  static_assert((Alignment & (Alignment - 1)) == 0,
                "Alignment must be a power of two");

  AlignedAllocator() noexcept = default;

  template <typename U>
  AlignedAllocator(const AlignedAllocator<U, Alignment> &) noexcept {}

  [[nodiscard]] T *allocate(std::size_t n) {
    if (n > std::numeric_limits<std::size_t>::max() / sizeof(T)) {
      throw std::bad_array_new_length();
    }

    void *ptr = nullptr;
    if (posix_memalign(&ptr, Alignment, n * sizeof(T)) != 0) {
      throw std::bad_alloc();
    }
    return static_cast<T *>(ptr);
  }

  void deallocate(T *p, std::size_t) noexcept { std::free(p); }

  template <typename U> struct rebind {
    using other = AlignedAllocator<U, Alignment>;
  };
};

template <typename T, typename U, std::size_t Alignment>
bool operator==(const AlignedAllocator<T, Alignment> &,
                const AlignedAllocator<U, Alignment> &) {
  return true;
}

template <typename T, typename U, std::size_t Alignment>
bool operator!=(const AlignedAllocator<T, Alignment> &,
                const AlignedAllocator<U, Alignment> &) {
  return false;
}

template <typename T>
T Fma(T a, T b, T c) {
#if defined(GEMM_DISABLE_EXPLICIT_FMA)
  return c + a * b;
#else
  if constexpr (std::is_floating_point_v<T>) {
    return std::fma(a, b, c);
  } else {
    return c + a * b;
  }
#endif
}

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
void PackRowMajorPanelPaddedCols(int rows, int cols, int paddedCols,
                                 const T *src, int ldsrc, T *dst) {
  for (int row = 0; row < rows; ++row) {
    T *dstRow = dst + row * paddedCols;
    std::copy_n(src + row * ldsrc, cols, dstRow);
    std::fill(dstRow + cols, dstRow + paddedCols, T(0));
  }
}

template <typename T, int MR>
GEMM_NOINLINE
void PackAMicroPanels(int rows, int cols, const T *src, int ldsrc, T *dst) {
  static_assert(MR > 0, "Micro-panel row count must be positive");

  for (int rowBlock = 0; rowBlock < rows; rowBlock += MR) {
    const int panelRows = std::min(MR, rows - rowBlock);
    for (int col = 0; col < cols; ++col) {
      for (int row = 0; row < panelRows; ++row) {
        *dst++ = src[(rowBlock + row) * ldsrc + col];
      }
    }
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

template <typename T>
GEMM_NOINLINE
void gemm_block_kernel_packed_ab_strided_b(int mc, int nc, int kc, T alpha,
                                           const T *Apack, const T *Bpack,
                                           int bpackStride, T *C, int ldc) {
  for (int i = 0; i < mc; ++i) {
    const T *ApackRow = Apack + i * kc;
    for (int k = 0; k < kc; ++k) {
      const T aik = alpha * ApackRow[k];
      const T *BpackRow = Bpack + k * bpackStride;
      for (int j = 0; j < nc; ++j) {
        C[i * ldc + j] += aik * BpackRow[j];
      }
    }
  }
}

template <typename T>
GEMM_NOINLINE
void gemm_block_kernel_packed_ab_micro_a(int mc, int nc, int kc, T alpha,
                                         const T *Apack, int aPanelRows,
                                         const T *Bpack, int bpackStride,
                                         T *C, int ldc) {
  for (int i = 0; i < mc; ++i) {
    for (int k = 0; k < kc; ++k) {
      const T aik = alpha * Apack[k * aPanelRows + i];
      const T *BpackRow = Bpack + k * bpackStride;
      for (int j = 0; j < nc; ++j) {
        C[i * ldc + j] += aik * BpackRow[j];
      }
    }
  }
}

template <typename T>
GEMM_NOINLINE
void gemm_micro_kernel_4x4_packed_ab(int kc, T alpha, const T *Apack,
                                     const T *Bpack, int bpackStride, T *C,
                                     int ldc) {
  T c00 = C[0 * ldc + 0];
  T c01 = C[0 * ldc + 1];
  T c02 = C[0 * ldc + 2];
  T c03 = C[0 * ldc + 3];
  T c10 = C[1 * ldc + 0];
  T c11 = C[1 * ldc + 1];
  T c12 = C[1 * ldc + 2];
  T c13 = C[1 * ldc + 3];
  T c20 = C[2 * ldc + 0];
  T c21 = C[2 * ldc + 1];
  T c22 = C[2 * ldc + 2];
  T c23 = C[2 * ldc + 3];
  T c30 = C[3 * ldc + 0];
  T c31 = C[3 * ldc + 1];
  T c32 = C[3 * ldc + 2];
  T c33 = C[3 * ldc + 3];

  for (int k = 0; k < kc; ++k) {
    const T *ApackCol = Apack + k * 4;
    const T a0 = alpha * ApackCol[0];
    const T a1 = alpha * ApackCol[1];
    const T a2 = alpha * ApackCol[2];
    const T a3 = alpha * ApackCol[3];
    const T *BpackRow = Bpack + k * bpackStride;
    const T b0 = BpackRow[0];
    const T b1 = BpackRow[1];
    const T b2 = BpackRow[2];
    const T b3 = BpackRow[3];

    c00 = Fma(a0, b0, c00);
    c01 = Fma(a0, b1, c01);
    c02 = Fma(a0, b2, c02);
    c03 = Fma(a0, b3, c03);
    c10 = Fma(a1, b0, c10);
    c11 = Fma(a1, b1, c11);
    c12 = Fma(a1, b2, c12);
    c13 = Fma(a1, b3, c13);
    c20 = Fma(a2, b0, c20);
    c21 = Fma(a2, b1, c21);
    c22 = Fma(a2, b2, c22);
    c23 = Fma(a2, b3, c23);
    c30 = Fma(a3, b0, c30);
    c31 = Fma(a3, b1, c31);
    c32 = Fma(a3, b2, c32);
    c33 = Fma(a3, b3, c33);
  }

  C[0 * ldc + 0] = c00;
  C[0 * ldc + 1] = c01;
  C[0 * ldc + 2] = c02;
  C[0 * ldc + 3] = c03;
  C[1 * ldc + 0] = c10;
  C[1 * ldc + 1] = c11;
  C[1 * ldc + 2] = c12;
  C[1 * ldc + 3] = c13;
  C[2 * ldc + 0] = c20;
  C[2 * ldc + 1] = c21;
  C[2 * ldc + 2] = c22;
  C[2 * ldc + 3] = c23;
  C[3 * ldc + 0] = c30;
  C[3 * ldc + 1] = c31;
  C[3 * ldc + 2] = c32;
  C[3 * ldc + 3] = c33;
}

template <typename T>
GEMM_NOINLINE
void gemm_block_kernel_packed_ab_register_blocked(int mc, int nc, int kc,
                                                  T alpha, const T *Apack,
                                                  const T *Bpack, T *C,
                                                  int ldc) {
  constexpr int MR = 4;
  constexpr int NR = 4;
  const int fullRows = mc - mc % MR;
  const int fullCols = nc - nc % NR;

  for (int i = 0; i < fullRows; i += MR) {
    const T *ApackPanel = Apack + (i / MR) * MR * kc;
    for (int j = 0; j < fullCols; j += NR) {
      gemm_micro_kernel_4x4_packed_ab(kc, alpha, ApackPanel, Bpack + j,
                                      nc, C + i * ldc + j, ldc);
    }
  }

  if (fullCols < nc) {
    const int edgeCols = nc - fullCols;
    for (int i = 0; i < fullRows; i += MR) {
      const T *ApackPanel = Apack + (i / MR) * MR * kc;
      gemm_block_kernel_packed_ab_micro_a(
          MR, edgeCols, kc, alpha, ApackPanel, MR, Bpack + fullCols, nc,
          C + i * ldc + fullCols, ldc);
    }
  }

  if (fullRows < mc) {
    const int edgeRows = mc - fullRows;
    const T *ApackPanel = Apack + (fullRows / MR) * MR * kc;
    gemm_block_kernel_packed_ab_micro_a(edgeRows, nc, kc, alpha, ApackPanel,
                                        edgeRows, Bpack, nc,
                                        C + fullRows * ldc, ldc);
  }
}

GEMM_NOINLINE GEMM_TARGET_AVX2_FMA
inline void gemm_micro_kernel_4x8_packed_ab_avx2_float(
    int kc, float alpha, const float *Apack, const float *Bpack,
    int bpackStride, float *C, int ldc) {
  __m256 c0 = _mm256_loadu_ps(C + 0 * ldc);
  __m256 c1 = _mm256_loadu_ps(C + 1 * ldc);
  __m256 c2 = _mm256_loadu_ps(C + 2 * ldc);
  __m256 c3 = _mm256_loadu_ps(C + 3 * ldc);

  for (int k = 0; k < kc; ++k) {
    const float *ApackCol = Apack + k * 4;
    const __m256 b = _mm256_load_ps(Bpack + k * bpackStride);
    const __m256 a0 = _mm256_set1_ps(alpha * ApackCol[0]);
    const __m256 a1 = _mm256_set1_ps(alpha * ApackCol[1]);
    const __m256 a2 = _mm256_set1_ps(alpha * ApackCol[2]);
    const __m256 a3 = _mm256_set1_ps(alpha * ApackCol[3]);

    c0 = _mm256_fmadd_ps(a0, b, c0);
    c1 = _mm256_fmadd_ps(a1, b, c1);
    c2 = _mm256_fmadd_ps(a2, b, c2);
    c3 = _mm256_fmadd_ps(a3, b, c3);
  }

  _mm256_storeu_ps(C + 0 * ldc, c0);
  _mm256_storeu_ps(C + 1 * ldc, c1);
  _mm256_storeu_ps(C + 2 * ldc, c2);
  _mm256_storeu_ps(C + 3 * ldc, c3);
}

GEMM_NOINLINE GEMM_TARGET_AVX2_FMA
inline void gemm_block_kernel_packed_ab_register_blocked_avx2_float(
    int mc, int nc, int kc, float alpha, const float *Apack,
    const float *Bpack, int bpackStride, float *C, int ldc) {
  constexpr int MR = 4;
  constexpr int NR = 8;
  const int fullRows = mc - mc % MR;
  const int fullCols = nc - nc % NR;

  for (int i = 0; i < fullRows; i += MR) {
    const float *ApackPanel = Apack + (i / MR) * MR * kc;
    for (int j = 0; j < fullCols; j += NR) {
      gemm_micro_kernel_4x8_packed_ab_avx2_float(
          kc, alpha, ApackPanel, Bpack + j, bpackStride, C + i * ldc + j,
          ldc);
    }
  }

  if (fullCols < nc) {
    const int edgeCols = nc - fullCols;
    for (int i = 0; i < fullRows; i += MR) {
      const float *ApackPanel = Apack + (i / MR) * MR * kc;
      gemm_block_kernel_packed_ab_micro_a(
          MR, edgeCols, kc, alpha, ApackPanel, MR, Bpack + fullCols,
          bpackStride, C + i * ldc + fullCols, ldc);
    }
  }

  if (fullRows < mc) {
    const int edgeRows = mc - fullRows;
    const float *ApackPanel = Apack + (fullRows / MR) * MR * kc;
    gemm_block_kernel_packed_ab_micro_a(edgeRows, nc, kc, alpha, ApackPanel,
                                        edgeRows, Bpack, bpackStride,
                                        C + fullRows * ldc, ldc);
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

template <BetaMode betaMode, typename T, int BM, int BN, int BK>
void gemm_packed_ab_register_blocked_impl(int M, int N, int K, T alpha,
                                          const T *A, int lda, const T *B,
                                          int ldb, T beta, T *C, int ldc) {
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
        PackAMicroPanels<T, 4>(mc, kc, A + ii * lda + kk, lda, Apack.data());
        gemm_block_kernel_packed_ab_register_blocked(
            mc, nc, kc, alpha, Apack.data(), Bpack.data(),
            C + ii * ldc + jj, ldc);
      }
    }
  }
}

template <BetaMode betaMode, int BM, int BN, int BK>
void gemm_packed_ab_register_blocked_avx2_float_impl(
    int M, int N, int K, float alpha, const float *A, int lda, const float *B,
    int ldb, float beta, float *C, int ldc) {
  ScaleC<betaMode>(M, N, beta, C, ldc);

  std::vector<float> Apack(BM * BK);
  std::vector<float, AlignedAllocator<float, 32>> Bpack(BK * BN);

  for (int jj = 0; jj < N; jj += BN) {
    const int nc = std::min(BN, N - jj);
    const int paddedNc = ((nc + 7) / 8) * 8;
    for (int kk = 0; kk < K; kk += BK) {
      const int kc = std::min(BK, K - kk);
      PackRowMajorPanelPaddedCols(kc, nc, paddedNc, B + kk * ldb + jj, ldb,
                                  Bpack.data());

      for (int ii = 0; ii < M; ii += BM) {
        const int mc = std::min(BM, M - ii);
        PackAMicroPanels<float, 4>(mc, kc, A + ii * lda + kk, lda,
                                   Apack.data());
        gemm_block_kernel_packed_ab_register_blocked_avx2_float(
            mc, nc, kc, alpha, Apack.data(), Bpack.data(), paddedNc,
            C + ii * ldc + jj, ldc);
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
// Same interface as gemm_packed_ab, but uses a scalar 4x4 register-blocked
// micro-kernel for full output tiles.
template <typename T, int BM = 64, int BN = 64, int BK = 64>
void gemm_packed_ab_register_blocked(int M, int N, int K, T alpha, const T *A,
                                     int lda, const T *B, int ldb, T beta,
                                     T *C, int ldc) {
  static_assert(BM > 0 && BN > 0 && BK > 0,
                "Block sizes must be positive");

  if (beta == T(0)) {
    detail::gemm_packed_ab_register_blocked_impl<detail::BetaMode::Zero, T, BM,
                                                 BN, BK>(
        M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
    return;
  }

  if (beta == T(1)) {
    detail::gemm_packed_ab_register_blocked_impl<detail::BetaMode::One, T, BM,
                                                 BN, BK>(
        M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
    return;
  }

  detail::gemm_packed_ab_register_blocked_impl<detail::BetaMode::General, T,
                                               BM, BN, BK>(
      M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
}

// Matrix multiplication: C = alpha * A * B + beta * C
// Same interface as gemm_packed_ab_register_blocked, but specialized for
// float and uses a 4x8 AVX2/FMA micro-kernel for full output tiles.
template <int BM = 64, int BN = 64, int BK = 64>
void gemm_packed_ab_register_blocked_avx2_float(
    int M, int N, int K, float alpha, const float *A, int lda, const float *B,
    int ldb, float beta, float *C, int ldc) {
  static_assert(BM > 0 && BN > 0 && BK > 0,
                "Block sizes must be positive");
  static_assert(BN % 8 == 0,
                "AVX2 float B panel width must be a multiple of 8");

  if (beta == 0.0f) {
    detail::gemm_packed_ab_register_blocked_avx2_float_impl<
        detail::BetaMode::Zero, BM, BN, BK>(M, N, K, alpha, A, lda, B, ldb,
                                            beta, C, ldc);
    return;
  }

  if (beta == 1.0f) {
    detail::gemm_packed_ab_register_blocked_avx2_float_impl<
        detail::BetaMode::One, BM, BN, BK>(M, N, K, alpha, A, lda, B, ldb,
                                           beta, C, ldc);
    return;
  }

  detail::gemm_packed_ab_register_blocked_avx2_float_impl<
      detail::BetaMode::General, BM, BN, BK>(M, N, K, alpha, A, lda, B, ldb,
                                             beta, C, ldc);
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

#undef GEMM_TARGET_AVX2_FMA
