#pragma once
#include <algorithm>
#include <array>
#include <immintrin.h>
namespace gemm {

// assume row-major storage
template <typename T> class gemm_pure_c {
public:
  gemm_pure_c() {}
  void operator()(const T *A, const T *B, T *C, int M, int N, int K) {
    // Initialize C matrix to zero
    std::fill(C, C + M * N, T(0));

    const int mb = (M + _MC - 1) / _MC;
    const int nb = (N + _NC - 1) / _NC;
    const int kb = (K + _KC - 1) / _KC;

    // Pre-compute remainders to avoid modulo in loops
    const int M_rem = M % _MC;
    const int N_rem = N % _NC;
    const int K_rem = K % _KC;
    const int MC_last = (M_rem == 0) ? _MC : M_rem;
    const int NC_last = (N_rem == 0) ? _NC : N_rem;
    const int KC_last = (K_rem == 0) ? _KC : K_rem;

    for (int j = 0; j < nb; ++j) {
      int NC = (j == nb - 1) ? NC_last : _NC;
      for (int k = 0; k < kb; ++k) {
        int KC = (k == kb - 1) ? KC_last : _KC;
        pack_B(&B[k * _KC * N + j * _NC], KC, NC, N);
        for (int i = 0; i < mb; ++i) {
          int MC = (i == mb - 1) ? MC_last : _MC;
          pack_A(&A[i * _MC * K + k * _KC], MC, KC, K);
          gemm_macro_kernel(MC, NC, KC, &C[i * _MC * N + j * _NC], N);
        }
      }
    }
  }

protected:
  void pack_B(const T *B, int KC, int NC, int N) {
    const int ib = (NC + _NR - 1) / _NR;
    const int _nr = NC % _NR;
    for (int j = 0; j < ib; j++) {
      int jb = (j == ib - 1 && _nr != 0) ? _nr : _NR;
      for (int k = 0; k < KC; ++k) {
        for (int jj = 0; jj < jb; ++jj) {
          _B[(j * KC + k) * _NR + jj] = B[k * N + j * _NR + jj];
        }
        for (int jj = jb; jj < _NR; ++jj) {
          _B[(j * KC + k) * _NR + jj] = T(0);
        }
      }
    }
  }

  void pack_A(const T *A, int MC, int KC, int K) {
    const int ia = (MC + _MR - 1) / _MR;
    const int _mr = MC % _MR;
    for (int i = 0; i < ia; i++) {
      int ib = (i == ia - 1 && _mr != 0) ? _mr : _MR;
      for (int k = 0; k < KC; ++k) {
        for (int ii = 0; ii < ib; ++ii) {
          _A[(i * KC + k) * _MR + ii] = A[(i * _MR + ii) * K + k];
        }
        for (int ii = ib; ii < _MR; ++ii) {
          _A[(i * KC + k) * _MR + ii] = T(0);
        }
      }
    }
  }

  void gemm_macro_kernel(int MC, int NC, int KC, T *C, int N) {
    for (int j = 0; j < NC; j += _NR) {
      int jb = std::min(_NR, NC - j);
      for (int i = 0; i < MC; i += _MR) {
        int ib = std::min(_MR, MC - i);
        // compute micro kernel
        const T *A_panel = &_A[(i / _MR) * KC * _MR];
        const T *B_panel = &_B[(j / _NR) * KC * _NR];
        Kernel_Micro(_MR, _NR, KC, N, A_panel, B_panel, _AB.data());

        for (int i = 0; i < MR; ++i) {
          for (int j = 0; j < NR; ++j) {
            C[i * N + j] += _AB[i * _NR + j];
          }
        }
      }
    }
  }

  void Kernel_Micro(int MR, int NR, int KC, int N, const T *A, const T *B,
                    T *C) {
    // std::fill(C, C + _MR * _NR, T(0));
    for (int k = 0; k < KC; ++k) {
      // #if defined(AVX512_SUPPORTED) && defined(FMA_SUPPORTED)
      //       for (int i = 0; i < MR; ++i) {
      //         const auto aik = A[k * _MR + i];
      //         if constexpr (std::is_same_v<T, float>) {
      //           static_assert(_NR % 16 == 0,
      //                         "_NR must be a multiple of 16 for AVX512");
      //           for (int j = 0; j < NR; j += 16) {
      //             __m512 c_vec = _mm512_loadu_ps(&C[i * _NR + j]);
      //             __m512 b_vec = _mm512_loadu_ps(&B[k * _NR + j]);
      //             __m512 a_vec = _mm512_set1_ps(aik);
      //             c_vec = _mm512_fmadd_ps(a_vec, b_vec, c_vec);
      //             _mm512_storeu_ps(&C[i * _NR + j], c_vec);
      //           }
      //         } else if constexpr (std::is_same_v<T, double>) {
      //           static_assert(_NR % 8 == 0, "_NR must be a multiple of 8 for
      //           AVX512"); for (int j = 0; j < NR; j += 8) {
      //             __m512d c_vec = _mm512_loadu_pd(&C[i * _NR + j]);
      //             __m512d b_vec = _mm512_loadu_pd(&B[k * _NR + j]);
      //             __m512d a_vec = _mm512_set1_pd(aik);
      //             c_vec = _mm512_fmadd_pd(a_vec, b_vec, c_vec);
      //             _mm512_storeu_pd(&C[i * _NR + j], c_vec);
      //           }
      //         } else {
      //           for (int j = 0; j < NR; ++j) {
      //             C[i * _NR + j] += aik * B[k * _NR + j];
      //           }
      //         }
      //       }
      // #elif defined(AVX2_SUPPORTED) && defined(FMA_SUPPORTED)
      //       for (int i = 0; i < MR; ++i) {
      //         const auto aik = A[k * _MR + i];
      //         if constexpr (std::is_same_v<T, float>) {
      //           static_assert(_NR % 8 == 0, "_NR must be a multiple of 8 for
      //           AVX2"); for (int j = 0; j < NR; j += 8) {
      //             __m256 c_vec = _mm256_loadu_ps(&C[i * _NR + j]);
      //             __m256 b_vec = _mm256_loadu_ps(&B[k * _NR + j]);
      //             __m256 a_vec = _mm256_set1_ps(aik);
      //             c_vec = _mm256_fmadd_ps(a_vec, b_vec, c_vec);
      //             _mm256_storeu_ps(&C[i * _NR + j], c_vec);
      //           }
      //         } else if constexpr (std::is_same_v<T, double>) {
      //           static_assert(_NR % 4 == 0, "_NR must be a multiple of 4 for
      //           AVX2"); for (int j = 0; j < NR; j += 4) {
      //             __m256d c_vec = _mm256_loadu_pd(&C[i * _NR + j]);
      //             __m256d b_vec = _mm256_loadu_pd(&B[k * _NR + j]);
      //             __m256d a_vec = _mm256_set1_pd(aik);
      //             c_vec = _mm256_fmadd_pd(a_vec, b_vec, c_vec);
      //             _mm256_storeu_pd(&C[i * _NR + j], c_vec);
      //           }
      //         } else {
      //           for (int j = 0; j < NR; ++j) {
      //             C[i * _NR + j] += aik * B[k * _NR + j];
      //           }
      //         }
      //       }
      // #else
      for (int i = 0; i < MR; ++i) {
        const auto aik = A[k * _MR + i];
        for (int j = 0; j < NR; ++j) {
          _AB[i * _NR + j] += aik * B[k * _NR + j];
        }
      }
      // #endif
    }
  }

protected:
  static constexpr int _MC{384};  // block size along M
  static constexpr int _KC{384};  // block size along K
  static constexpr int _NC{4096}; // block size along N
  static constexpr int _MR{4};    // micro-panel size along M
  static constexpr int _NR{4};    // micro-panel size along N
  std::array<T, _MC * _KC> _A;
  std::array<T, _KC * _NC> _B;
  std::array<T, _MR * _NR> _AB;
};
} // namespace gemm