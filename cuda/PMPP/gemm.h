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