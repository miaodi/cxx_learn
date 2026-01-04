#pragma once
#include <cuda_runtime.h>

__global__ void simple_reduction_kernel(const float *input, float *output,
                                        int size);

float simple_reduction(const float *input, int size);

__global__ void coarsened_reduction_kernel(const float *input, float *output,
                                           int size, int coarseningFactor);

float coarsened_reduction(const float *input, int size);

// Thrust-based reduction as reference
float thrust_reduction(const float *input, int size);