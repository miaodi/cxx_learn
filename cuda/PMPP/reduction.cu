#include "reduction.h"
#include <algorithm>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/reduce.h>

__global__ void simple_reduction_kernel(const float *input, float *output,
                                        int size) {
  extern __shared__ float shared_data[];
  int tid = threadIdx.x;
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  // Load input into shared memory and perform first reduction step
  float value = (idx < size) ? input[idx] : 0.0f;
  int idx2 = idx + blockDim.x / 2;
  if (idx2 < size) {
    value += input[idx2];
  }
  shared_data[tid] = value;
  __syncthreads();

  // Perform reduction in shared memory (one less iteration needed)
#pragma unroll (16)
  for (int stride = blockDim.x / 4; stride > 0; stride /= 2) {
    if (tid < stride) {
      shared_data[tid] += shared_data[tid + stride];
    }
    __syncthreads();
  }

  // Write result for this block to output using atomic operation
  if (tid == 0) {
    atomicAdd(output, shared_data[0]);
  }
}

float simple_reduction(const float *input, int size) {
  float *d_input, *d_output;
  float result = 0.0f;
  const int blockSize = 256;

  // Allocate device memory
  cudaMalloc(&d_input, size * sizeof(float));
  cudaMalloc(&d_output, sizeof(float));

  // Copy input to device
  cudaMemcpy(d_input, input, size * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemset(d_output, 0, sizeof(float));

  // Launch kernel
  int numBlocks = (size + blockSize - 1) / blockSize;
  int sharedMemSize = blockSize * sizeof(float);
  simple_reduction_kernel<<<numBlocks, blockSize, sharedMemSize>>>(
      d_input, d_output, size);

  // Copy result back to host
  cudaMemcpy(&result, d_output, sizeof(float), cudaMemcpyDeviceToHost);

  // Free device memory
  cudaFree(d_input);
  cudaFree(d_output);

  return result;
}

__global__ void coarsened_reduction_kernel(const float *input, float *output,
                                           int size, int coarseningFactor) {
  extern __shared__ float shared_data[];
  int tid = threadIdx.x;
  int blockStart = blockIdx.x * blockDim.x * coarseningFactor;
  
  // Each thread processes multiple elements (coarsening)
  float sum = 0.0f;
#pragma unroll (16)
  for (int i = 0; i < coarseningFactor; ++i) {
    int idx = blockStart + i * blockDim.x + tid;
    if (idx < size) {
      sum += input[idx];
    }
  }
  
  // Store partial sum in shared memory
  shared_data[tid] = sum;
  __syncthreads();

  // Perform reduction in shared memory
  for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
    if (tid < stride) {
      shared_data[tid] += shared_data[tid + stride];
    }
    __syncthreads();
  }

  // Write result for this block to output using atomic operation
  if (tid == 0) {
    atomicAdd(output, shared_data[0]);
  }
}

float coarsened_reduction(const float *input, int size) {
  float *d_input, *d_output;
  float result = 0.0f;
  const int blockSize = 256;

  // Allocate device memory
  cudaMalloc(&d_input, size * sizeof(float));
  cudaMalloc(&d_output, sizeof(float));

  // Copy input to device
  cudaMemcpy(d_input, input, size * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemset(d_output, 0, sizeof(float));

  // Automatically determine coarsening factor based on problem size
  // Goal: Keep number of blocks reasonable (between 128 and 4096 blocks)
  // to ensure good GPU utilization without too much overhead
  int coarseningFactor = 1;
  int maxBlocks = 4096;
  int minBlocks = 128;
  
  // Calculate how many blocks we'd have without coarsening
  int blocksWithoutCoarsening = (size + blockSize - 1) / blockSize;
  
  if (blocksWithoutCoarsening > maxBlocks) {
    // Too many blocks, increase coarsening factor
    coarseningFactor = (blocksWithoutCoarsening + maxBlocks - 1) / maxBlocks;
    // Clamp to reasonable values (power of 2 for better performance)
    if (coarseningFactor > 16) coarseningFactor = 16;
    else if (coarseningFactor > 8) coarseningFactor = 8;
    else if (coarseningFactor > 4) coarseningFactor = 4;
    else if (coarseningFactor > 2) coarseningFactor = 2;
  } else if (blocksWithoutCoarsening > minBlocks) {
    // Good range, use moderate coarsening for better memory bandwidth
    coarseningFactor = 2;
  }
  // else: Small problem, no coarsening needed (factor = 1)

  // Launch kernel with coarsening
  // Each block processes blockSize * coarseningFactor elements
  int numBlocks = (size + blockSize * coarseningFactor - 1) / (blockSize * coarseningFactor);
  int sharedMemSize = blockSize * sizeof(float);
  coarsened_reduction_kernel<<<numBlocks, blockSize, sharedMemSize>>>(
      d_input, d_output, size, coarseningFactor);

  // Copy result back to host
  cudaMemcpy(&result, d_output, sizeof(float), cudaMemcpyDeviceToHost);

  // Free device memory
  cudaFree(d_input);
  cudaFree(d_output);

  return result;
}

float thrust_reduction(const float *input, int size) {
  // Copy data to device using Thrust
  thrust::device_vector<float> d_vec(input, input + size);
  
  // Perform reduction using Thrust
  float result = thrust::reduce(d_vec.begin(), d_vec.end(), 0.0f, cuda::std::plus<float>());
  
  return result;
}