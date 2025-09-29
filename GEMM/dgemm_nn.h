#pragma once
#include <stdio.h>

namespace gemm {

// Block sizes for cache optimization
constexpr int MC = 384;  // M block size (rows of A)
constexpr int KC = 384;  // K block size (shared dimension)
constexpr int NC = 4096; // N block size (columns of B)

// Micro-kernel sizes for register blocking
constexpr int MR = 4; // Micro-panel rows
constexpr int NR = 4; // Micro-panel columns

//
//  Local buffers for storing panels from A, B and C (now templated)
//
template <typename T> struct gemm_buffers {
  static thread_local T _A[MC * KC];
  static thread_local T _B[KC * NC];
  static thread_local T _C[MR * NR];
};

template <typename T> thread_local T gemm_buffers<T>::_A[MC * KC];

template <typename T> thread_local T gemm_buffers<T>::_B[KC * NC];

template <typename T> thread_local T gemm_buffers<T>::_C[MR * NR];

//
//  Packing complete panels from A (i.e. without padding)
//
template <typename T>
static void pack_MRxk(int k, const T *A, int incRowA, int incColA, T *buffer) {
  int i, j;

  for (j = 0; j < k; ++j) {
    for (i = 0; i < MR; ++i) {
      buffer[i] = A[i * incRowA];
    }
    buffer += MR;
    A += incColA;
  }
}

//
//  Packing panels from A with padding if required
//
template <typename T>
static void pack_A(int mc, int kc, const T *A, int incRowA, int incColA,
                   T *buffer) {
  int mp = mc / MR;
  int _mr = mc % MR;

  int i, j;

  for (i = 0; i < mp; ++i) {
    pack_MRxk<T>(kc, A, incRowA, incColA, buffer);
    buffer += kc * MR;
    A += MR * incRowA;
  }
  if (_mr > 0) {
    for (j = 0; j < kc; ++j) {
      for (i = 0; i < _mr; ++i) {
        buffer[i] = A[i * incRowA];
      }
      for (i = _mr; i < MR; ++i) {
        buffer[i] = T(0);
      }
      buffer += MR;
      A += incColA;
    }
  }
}

//
//  Packing complete panels from B (i.e. without padding)
//
template <typename T>
static void pack_kxNR(int k, const T *B, int incRowB, int incColB, T *buffer) {
  int i, j;

  for (i = 0; i < k; ++i) {
    for (j = 0; j < NR; ++j) {
      buffer[j] = B[j * incColB];
    }
    buffer += NR;
    B += incRowB;
  }
}

//
//  Packing panels from B with padding if required
//
template <typename T>
static void pack_B(int kc, int nc, const T *B, int incRowB, int incColB,
                   T *buffer) {
  int np = nc / NR;
  int _nr = nc % NR;

  int i, j;

  for (j = 0; j < np; ++j) {
    pack_kxNR<T>(kc, B, incRowB, incColB, buffer);
    buffer += kc * NR;
    B += NR * incColB;
  }
  if (_nr > 0) {
    for (i = 0; i < kc; ++i) {
      for (j = 0; j < _nr; ++j) {
        buffer[j] = B[j * incColB];
      }
      for (j = _nr; j < NR; ++j) {
        buffer[j] = T(0);
      }
      buffer += NR;
      B += incRowB;
    }
  }
}

//
//  Micro kernel for multiplying panels from A and B.
//
template <typename T>
static void gemm_micro_kernel(int kc, T alpha, const T *A, const T *B, T beta,
                              T *C, int incRowC, int incColC) {
  T AB[MR * NR];

  int i, j, l;

  //
  //  Compute AB = A*B
  //
  for (l = 0; l < MR * NR; ++l) {
    AB[l] = T(0);
  }
  for (l = 0; l < kc; ++l) {
    for (j = 0; j < NR; ++j) {
      for (i = 0; i < MR; ++i) {
        AB[i + j * MR] += A[i] * B[j];
      }
    }
    A += MR;
    B += NR;
  }

  //
  //  Update C <- beta*C
  //
  if (beta == T(0)) {
    for (j = 0; j < NR; ++j) {
      for (i = 0; i < MR; ++i) {
        C[i * incRowC + j * incColC] = T(0);
      }
    }
  } else if (beta != T(1)) {
    for (j = 0; j < NR; ++j) {
      for (i = 0; i < MR; ++i) {
        C[i * incRowC + j * incColC] *= beta;
      }
    }
  }

  //
  //  Update C <- C + alpha*AB (note: the case alpha==0.0 was already treated in
  //                                  the above layer gemm_nn)
  //
  if (alpha == T(1)) {
    for (j = 0; j < NR; ++j) {
      for (i = 0; i < MR; ++i) {
        C[i * incRowC + j * incColC] += AB[i + j * MR];
      }
    }
  } else {
    for (j = 0; j < NR; ++j) {
      for (i = 0; i < MR; ++i) {
        C[i * incRowC + j * incColC] += alpha * AB[i + j * MR];
      }
    }
  }
}

//
//  Compute Y += alpha*X
//
template <typename T>
static void geaxpy(int m, int n, T alpha, const T *X, int incRowX, int incColX,
                   T *Y, int incRowY, int incColY) {
  int i, j;

  if (alpha != T(1)) {
    for (j = 0; j < n; ++j) {
      for (i = 0; i < m; ++i) {
        Y[i * incRowY + j * incColY] += alpha * X[i * incRowX + j * incColX];
      }
    }
  } else {
    for (j = 0; j < n; ++j) {
      for (i = 0; i < m; ++i) {
        Y[i * incRowY + j * incColY] += X[i * incRowX + j * incColX];
      }
    }
  }
}

//
//  Compute X *= alpha
//
template <typename T>
static void gescal(int m, int n, T alpha, T *X, int incRowX, int incColX) {
  int i, j;

  if (alpha != T(0)) {
    for (j = 0; j < n; ++j) {
      for (i = 0; i < m; ++i) {
        X[i * incRowX + j * incColX] *= alpha;
      }
    }
  } else {
    for (j = 0; j < n; ++j) {
      for (i = 0; i < m; ++i) {
        X[i * incRowX + j * incColX] = T(0);
      }
    }
  }
}

//
//  Macro Kernel for the multiplication of blocks of A and B.  We assume that
//  these blocks were previously packed to buffers _A and _B.
//
template <typename T>
static void gemm_macro_kernel(int mc, int nc, int kc, T alpha, T beta, T *C,
                              int incRowC, int incColC) {
  int mp = (mc + MR - 1) / MR;
  int np = (nc + NR - 1) / NR;

  int _mr = mc % MR;
  int _nr = nc % NR;

  int mr, nr;
  int i, j;

  T *_A = gemm_buffers<T>::_A;
  T *_B = gemm_buffers<T>::_B;
  T *_C = gemm_buffers<T>::_C;

  for (j = 0; j < np; ++j) {
    nr = (j != np - 1 || _nr == 0) ? NR : _nr;

    for (i = 0; i < mp; ++i) {
      mr = (i != mp - 1 || _mr == 0) ? MR : _mr;

      if (mr == MR && nr == NR) {
        gemm_micro_kernel<T>(kc, alpha, &_A[i * kc * MR], &_B[j * kc * NR],
                             beta, &C[i * MR * incRowC + j * NR * incColC],
                             incRowC, incColC);
      } else {
        gemm_micro_kernel<T>(kc, alpha, &_A[i * kc * MR], &_B[j * kc * NR],
                             T(0), _C, 1, MR);
        gescal<T>(mr, nr, beta, &C[i * MR * incRowC + j * NR * incColC],
                  incRowC, incColC);
        geaxpy<T>(mr, nr, T(1), _C, 1, MR,
                  &C[i * MR * incRowC + j * NR * incColC], incRowC, incColC);
      }
    }
  }
}

//
//  Compute C <- beta*C + alpha*A*B
//
template <typename T>
void gemm_nn(int m, int n, int k, T alpha, const T *A, int incRowA, int incColA,
             const T *B, int incRowB, int incColB, T beta, T *C, int incRowC,
             int incColC) {
  int mb = (m + MC - 1) / MC;
  int nb = (n + NC - 1) / NC;
  int kb = (k + KC - 1) / KC;

  int _mc = m % MC;
  int _nc = n % NC;
  int _kc = k % KC;

  int mc, nc, kc;
  int i, j, l;

  T _beta;

  T *_A = gemm_buffers<T>::_A;
  T *_B = gemm_buffers<T>::_B;

  if (alpha == T(0) || k == 0) {
    gescal<T>(m, n, beta, C, incRowC, incColC);
    return;
  }

  for (j = 0; j < nb; ++j) {
    nc = (j != nb - 1 || _nc == 0) ? NC : _nc;

    for (l = 0; l < kb; ++l) {
      kc = (l != kb - 1 || _kc == 0) ? KC : _kc;
      _beta = (l == 0) ? beta : T(1);

      pack_B<T>(kc, nc, &B[l * KC * incRowB + j * NC * incColB], incRowB,
                incColB, _B);

      for (i = 0; i < mb; ++i) {
        mc = (i != mb - 1 || _mc == 0) ? MC : _mc;

        pack_A<T>(mc, kc, &A[i * MC * incRowA + l * KC * incColA], incRowA,
                  incColA, _A);

        gemm_macro_kernel<T>(mc, nc, kc, alpha, _beta,
                             &C[i * MC * incRowC + j * NC * incColC], incRowC,
                             incColC);
      }
    }
  }
}

} // namespace gemm
