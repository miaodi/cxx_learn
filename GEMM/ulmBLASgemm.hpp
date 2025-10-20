#pragma once
#include <algorithm>
#include <array>
#include <immintrin.h>
#include <iostream>
#include <vector>
namespace gemm {

// assume row-major storage
template <typename T> class gemm_pure_c {
public:
  gemm_pure_c() {}
  void operator()(const T *A, const T *B, T *C, int M, int N, int K) {
    // // Initialize C matrix to zero
    // std::fill(C, C + M * N, T(0));

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
    const int mp = (MC + _MR - 1) / _MR;
    const int np = (NC + _NR - 1) / _NR;
    const int _mr = MC % _MR;
    const int _nr = NC % _NR;

    for (int j = 0; j < np; ++j) {
      int jb = (j == np - 1 && _nr != 0) ? _nr : _NR;
      const T *B_panel = &_B[(j * KC * _NR)];
      for (int i = 0; i < mp; ++i) {
        int ib = (i == mp - 1 && _mr != 0) ? _mr : _MR;
        // compute micro kernel
        const T *A_panel = &_A[(i * KC * _MR)];
        Kernel_Micro(_MR, _NR, KC, N, A_panel, B_panel, &C[i * _MR * N]);
      }
      C += _NR;
    }
  }

  void Kernel_Micro(int MR, int NR, int KC, int N, const T *A, const T *B,
                    T *C) {
    T _C[_MR * _NR] = {0};
    for (int k = 0; k < KC; ++k) {
#if defined(AVX512_SUPPORTED) && defined(FMA_SUPPORTED)
      // AVX512 vectorized version
      for (int i = 0; i < MR; ++i) {
        const auto aik = A[i];
        auto c_row = &_C[i * NR];

        if constexpr (std::is_same_v<T, float>) {
          static_assert(_NR % 16 == 0,
                        "_NR must be a multiple of 16 for AVX512");
          for (int j = 0; j < NR; j += 16) {
            __m512 c_vec = _mm512_loadu_ps(&c_row[j]);
            __m512 b_vec = _mm512_loadu_ps(&B[j]);
            __m512 a_vec = _mm512_set1_ps(aik);
            c_vec = _mm512_fmadd_ps(a_vec, b_vec, c_vec);
            _mm512_storeu_ps(&c_row[j], c_vec);
          }
        } else if constexpr (std::is_same_v<T, double>) {
          static_assert(_NR % 8 == 0, "_NR must be a multiple of 8 for AVX512");
          for (int j = 0; j < NR; j += 8) {
            __m512d c_vec = _mm512_loadu_pd(&c_row[j]);
            __m512d b_vec = _mm512_loadu_pd(&B[j]);
            __m512d a_vec = _mm512_set1_pd(aik);
            c_vec = _mm512_fmadd_pd(a_vec, b_vec, c_vec);
            _mm512_storeu_pd(&c_row[j], c_vec);
          }
        } else {
          // Fallback for other types
          for (int j = 0; j < NR; ++j) {
            c_row[j] += aik * B[j];
          }
        }
      }
#elif defined(AVX2_SUPPORTED) && defined(FMA_SUPPORTED)
      // AVX2 vectorized version
      for (int i = 0; i < MR; ++i) {
        const auto aik = A[i];
        auto c_row = &_C[i * NR];

        if constexpr (std::is_same_v<T, float>) {
          static_assert(_NR % 8 == 0, "_NR must be a multiple of 8 for AVX2");
          for (int j = 0; j < NR; j += 8) {
            __m256 c_vec = _mm256_loadu_ps(&c_row[j]);
            __m256 b_vec = _mm256_loadu_ps(&B[j]);
            __m256 a_vec = _mm256_set1_ps(aik);
            c_vec = _mm256_fmadd_ps(a_vec, b_vec, c_vec);
            _mm256_storeu_ps(&c_row[j], c_vec);
          }
        } else if constexpr (std::is_same_v<T, double>) {
          static_assert(_NR % 4 == 0, "_NR must be a multiple of 4 for AVX2");
          for (int j = 0; j < NR; j += 4) {
            __m256d c_vec = _mm256_loadu_pd(&c_row[j]);
            __m256d b_vec = _mm256_loadu_pd(&B[j]);
            __m256d a_vec = _mm256_set1_pd(aik);
            c_vec = _mm256_fmadd_pd(a_vec, b_vec, c_vec);
            _mm256_storeu_pd(&c_row[j], c_vec);
          }
        } else {
          // Fallback for other types
          for (int j = 0; j < NR; ++j) {
            c_row[j] += aik * B[j];
          }
        }
      }

#else
      // Scalar fallback version (your original code)
      for (int i = 0; i < MR; ++i) {
        const auto aik = A[i];
        auto c_row = &_C[i * NR];
        for (int j = 0; j < NR; ++j) {
          c_row[j] += aik * B[j];
        }
      }
#endif
      A += MR;
      B += NR;
    }
    for (int i = 0; i < MR; ++i) {
      auto c_row = C + i * N;
      auto c_buf_row = &_C[i * NR];
      for (int j = 0; j < NR; ++j) {
        c_row[j] += c_buf_row[j];
      }
    }
  }

protected:
  static constexpr int _MC{384};   // block size along M
  static constexpr int _KC{384};   // block size along K
  static constexpr int _NC{4096};  // block size along N
  #if defined(AVX512_SUPPORTED)
    static constexpr int _MR{16}; // micro-panel size along M
    static constexpr int _NR{16}; // micro-panel size along N
  #else
    static constexpr int _MR{8};  // micro-panel size along M
    static constexpr int _NR{8};  // micro-panel size along N
  #endif
  std::vector<T> _A = std::vector<T>(_MC * _KC);
  std::vector<T> _B = std::vector<T>(_KC * _NC);
  // T _C[_MR * _NR];
};
} // namespace gemm