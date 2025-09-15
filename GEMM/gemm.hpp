#pragma once
#include <algorithm>
#include <immintrin.h>
#include <vector>
namespace gemm {

// Matrix multiplication: C = A * B
// A is M x K, B is K x N, C is M x N
// Assumes row-major storage
template <typename T>
void MatMatMul(const T *A, const T *B, T *C, int M, int N, int K) {
  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < N; ++j) {
      C[i * N + j] = 0.0f;
      for (int k = 0; k < K; ++k) {
        C[i * N + j] += A[i * K + k] * B[k * N + j];
      }
    }
  }
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

// Tiled matrix multiplication: C = A * B
// A is M x K, B is K x N, C is M x N
// Assumes row-major storage
template <typename T, int TileSize = 16>
void TiledMatMatMul(const T *A, const T *B, T *C, int M, int N, int K) {
  for (int i = 0; i < M; i += TileSize) {
    for (int j = 0; j < N; j += TileSize) {
      for (int k = 0; k < K; k += TileSize) {
        for (int ii = i; ii < std::min(i + TileSize, M); ++ii) {
          for (int jj = j; jj < std::min(j + TileSize, N); ++jj) {
            T sum = 0;
            for (int kk = k; kk < std::min(k + TileSize, K); ++kk) {
              sum += A[ii * K + kk] * B[kk * N + jj];
            }
            C[ii * N + jj] += sum;
          }
        }
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
inline void Kernel_TileTile(const T *__restrict__ Atile,
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
        Kernel_TileTile<T, TileSize>(Atile, BTtile, Ctile);
      }
    }
  }
}

template <typename T, int TileSize = 4>
void TiledMatMatMulInternalTiledPadded(const T *A, const T *B, T *C, int M,
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
} // namespace gemm