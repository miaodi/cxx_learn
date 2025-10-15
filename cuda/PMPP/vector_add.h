#pragma once
#include <cuda_runtime.h>
#include <omp.h>

void cpu_vector_add(const float *A, const float *B, float *C, int N);

__global__ void gpu_vector_add_kernel(const float *A, const float *B, float *C,
                                      int N);

void gpu_vector_add(const float *A, const float *B, float *C, int N);