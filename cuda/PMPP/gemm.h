#pragma once
#include <algorithm>
#include <cuda_runtime.h>
#include <omp.h>

// Matrix multiplication: C = A * B
// A is M x K, B is K x N, C is M x N
// Assumes row-major storage
__global__ void gemm_kernel(const float *A, const float *B, float *C, int M,
                            int N, int K);

void gpu_gemm(const float *A, const float *B, float *C, int M, int N, int K);

__global__ void transpose_kernel(const float *input, float *output, int rows,
                                 int cols);

void gpu_gemm_tiled(const float *A, const float *B, float *C, int M, int N,
                    int K, int tileSize);

template <int TILE>
__global__ void gemm_tiled_kernel(
    const float *__restrict__ A, // M×K
    const float *__restrict__ B, // K×N
    float *__restrict__ C,       // M×N (output)
    int M, int N, int K);

template <int TILE, int MICRO_TILE, bool USE_COALESCED = true>
void gpu_gemm_micro_tiled(
    const float *A, const float *B, float *C, int M, int N,
    int K);

// Explicit instantiation declarations for micro-tiled variants
extern template void gpu_gemm_micro_tiled<32, 2, true>(const float*, const float*, float*, int, int, int);
extern template void gpu_gemm_micro_tiled<32, 2, false>(const float*, const float*, float*, int, int, int);
extern template void gpu_gemm_micro_tiled<32, 4, true>(const float*, const float*, float*, int, int, int);
extern template void gpu_gemm_micro_tiled<32, 4, false>(const float*, const float*, float*, int, int, int);
extern template void gpu_gemm_micro_tiled<64, 2, true>(const float*, const float*, float*, int, int, int);
extern template void gpu_gemm_micro_tiled<64, 2, false>(const float*, const float*, float*, int, int, int);
extern template void gpu_gemm_micro_tiled<64, 4, true>(const float*, const float*, float*, int, int, int);
extern template void gpu_gemm_micro_tiled<64, 4, false>(const float*, const float*, float*, int, int, int);

template <int TILE, int MICRO_TILE, bool USE_COALESCED = true>
__global__ void gemm_micro_tiled_kernel(
    const float *__restrict__ A, // M×K
    const float *__restrict__ B, // K×N
    float *__restrict__ C,       // M×N (output)
    int M, int N, int K);
