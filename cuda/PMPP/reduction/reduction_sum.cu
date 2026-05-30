#include "reduction_sum.h"

#include <cuda/std/functional>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>
#include <thrust/reduce.h>

#include <algorithm>
#include <cstddef>
#include <stdexcept>
#include <string>

namespace pmpp::reduction {
namespace {

constexpr int kBlockSize = 256;
constexpr int kCoarseningFactor = 4;

enum class SumKernel {
  SimpleStride,
  SequentialAddressing,
  CoarsenedSharedMemory,
  CoarsenedSharedMemoryOptimized,
};

std::size_t elements_per_block(SumKernel kernel) {
  switch (kernel) {
  case SumKernel::SimpleStride:
  case SumKernel::SequentialAddressing:
    return 2 * kBlockSize;
  case SumKernel::CoarsenedSharedMemory:
  case SumKernel::CoarsenedSharedMemoryOptimized:
    return kCoarseningFactor * kBlockSize;
  }
  return 2 * kBlockSize;
}

void check_cuda(cudaError_t status, const char *operation) {
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string(operation) + ": " +
                             cudaGetErrorString(status));
  }
}

std::size_t padded_block_elements(std::size_t size) {
  const std::size_t min_block_elements = 2 * kBlockSize;
  return std::max<std::size_t>(min_block_elements,
                               ((size + min_block_elements - 1) /
                                min_block_elements) * min_block_elements);
}

__global__ void simple_stride_sum_kernel(const float *input, float *scratch,
                                         float *partials, std::size_t size) {
  const unsigned int tid = threadIdx.x;
  const std::size_t block_start = 2 * blockIdx.x * blockDim.x;
  const std::size_t first = block_start + tid;
  const std::size_t second = first + blockDim.x;

  float value = (first < size) ? input[first] : 0.0f;
  if (second < size) {
    value += input[second];
  }
  scratch[first] = value;
  __syncthreads();

  // PMPP's simple interleaved-addressing reduction. It intentionally reduces
  // in global memory so the next variant can motivate shared-memory staging.
  for (unsigned int stride = 1; stride < blockDim.x; stride *= 2) {
    if (tid % (2 * stride) == 0) {
      const std::size_t other = first + stride;
      scratch[first] += scratch[other];
    }
    __syncthreads();
  }

  if (tid == 0) {
    partials[blockIdx.x] = scratch[block_start];
  }
}

__global__ void sequential_addressing_sum_kernel(const float *input,
                                                 float *scratch, float *partials,
                                                 std::size_t size) {
  const unsigned int tid = threadIdx.x;
  const std::size_t block_start = 2 * blockIdx.x * blockDim.x;
  const std::size_t first = block_start + tid;
  const std::size_t second = first + blockDim.x;

  float value = (first < size) ? input[first] : 0.0f;
  if (second < size) {
    value += input[second];
  }
  scratch[first] = value;
  __syncthreads();

  // Same reduction tree as simple_stride_sum_kernel, but the active threads are
  // contiguous at every step: first 256 add second 256, first 128 add second
  // 128, and so on.
  for (unsigned int stride = blockDim.x / 2; stride > 0; stride /= 2) {
    if (tid < stride) {
      const std::size_t other = first + stride;
      scratch[first] += scratch[other];
    }
    __syncthreads();
  }

  if (tid == 0) {
    partials[blockIdx.x] = scratch[block_start];
  }
}

__global__ void coarsened_shared_memory_sum_kernel(const float *input,
                                                    float *partials,
                                                    std::size_t size) {
  __shared__ float shared[kBlockSize];

  const unsigned int tid = threadIdx.x;
  const std::size_t block_start =
      kCoarseningFactor * blockIdx.x * blockDim.x;

  float local_sum = 0.0f;
  for (int round = 0; round < kCoarseningFactor; ++round) {
    const std::size_t index = block_start + round * blockDim.x + tid;
    if (index < size) {
      local_sum += input[index];
    }
  }

  shared[tid] = local_sum;
  __syncthreads();

  for (unsigned int stride = blockDim.x / 2; stride > 0; stride /= 2) {
    if (tid < stride) {
      shared[tid] += shared[tid + stride];
    }
    __syncthreads();
  }

  if (tid == 0) {
    partials[blockIdx.x] = shared[0];
  }
}

__global__ void coarsened_shared_memory_sum_optimized_kernel(
    const float *input, float *partials, std::size_t size) {
  __shared__ float shared[kBlockSize];

  const unsigned int tid = threadIdx.x;
  const std::size_t block_start =
      kCoarseningFactor * blockIdx.x * blockDim.x;

  float local_sum = 0.0f;
#pragma unroll
  for (int round = 0; round < kCoarseningFactor; ++round) {
    const std::size_t index = block_start + round * blockDim.x + tid;
    if (index < size) {
      local_sum += input[index];
    }
  }

  shared[tid] = local_sum;
  __syncthreads();

#pragma unroll
  for (unsigned int stride = blockDim.x / 2; stride >= 32; stride /= 2) {
    if (tid < stride) {
      shared[tid] += shared[tid + stride];
    }
    __syncthreads();
  }

  float value = shared[tid];
  if (tid < 32) {
    value += __shfl_down_sync(0xffffffff, value, 16);
    value += __shfl_down_sync(0xffffffff, value, 8);
    value += __shfl_down_sync(0xffffffff, value, 4);
    value += __shfl_down_sync(0xffffffff, value, 2);
    value += __shfl_down_sync(0xffffffff, value, 1);
  }

  if (tid == 0) {
    partials[blockIdx.x] = value;
  }
}

void launch_sum_kernel(SumKernel kernel, const float *input, float *scratch,
                       float *partials, std::size_t size, std::size_t blocks) {
  switch (kernel) {
  case SumKernel::SimpleStride:
    simple_stride_sum_kernel<<<static_cast<unsigned int>(blocks), kBlockSize>>>(
        input, scratch, partials, size);
    check_cuda(cudaGetLastError(), "simple_stride_sum_kernel launch");
    break;
  case SumKernel::SequentialAddressing:
    sequential_addressing_sum_kernel<<<static_cast<unsigned int>(blocks),
                                       kBlockSize>>>(input, scratch, partials,
                                                     size);
    check_cuda(cudaGetLastError(), "sequential_addressing_sum_kernel launch");
    break;
  case SumKernel::CoarsenedSharedMemory:
    coarsened_shared_memory_sum_kernel<<<static_cast<unsigned int>(blocks),
                                          kBlockSize>>>(input, partials, size);
    check_cuda(cudaGetLastError(),
               "coarsened_shared_memory_sum_kernel launch");
    break;
  case SumKernel::CoarsenedSharedMemoryOptimized:
    coarsened_shared_memory_sum_optimized_kernel<<<
        static_cast<unsigned int>(blocks), kBlockSize>>>(input, partials, size);
    check_cuda(cudaGetLastError(),
               "coarsened_shared_memory_sum_optimized_kernel launch");
    break;
  }
}

float run_device_sum(const float *device_input, std::size_t size,
                     SumKernel kernel, DeviceReductionBuffers &buffers) {
  if (size == 0) {
    return 0.0f;
  }

  const std::size_t block_elements = elements_per_block(kernel);

  if (buffers.scratch_capacity < padded_block_elements(size) ||
      buffers.partial_capacity == 0 ||
      buffers.scratch == nullptr || buffers.partials_a == nullptr ||
      buffers.partials_b == nullptr) {
    throw std::runtime_error("device reduction buffers are too small");
  }

  const float *current_input = device_input;
  float *current_output = buffers.partials_a;
  std::size_t current_size = size;

  while (current_size > 1) {
    const std::size_t blocks =
        (current_size + block_elements - 1) / block_elements;
    launch_sum_kernel(kernel, current_input, buffers.scratch, current_output,
                      current_size, blocks);

    current_size = blocks;
    current_input = current_output;
    current_output = (current_output == buffers.partials_a) ? buffers.partials_b
                                                            : buffers.partials_a;
  }

  float result = 0.0f;
  check_cuda(cudaMemcpy(&result, current_input, sizeof(float),
                        cudaMemcpyDeviceToHost),
             "cudaMemcpy result");
  return result;
}

} // namespace

void allocate_device_reduction_buffers(DeviceReductionBuffers &buffers,
                                       std::size_t size) {
  free_device_reduction_buffers(buffers);

  if (size == 0) {
    return;
  }

  const std::size_t scratch_capacity = padded_block_elements(size);
  const std::size_t min_block_elements = 2 * kBlockSize;
  const std::size_t partial_capacity =
      std::max<std::size_t>(1, (size + min_block_elements - 1) /
                                   min_block_elements);

  check_cuda(cudaMalloc(&buffers.scratch, scratch_capacity * sizeof(float)),
             "cudaMalloc scratch");
  check_cuda(cudaMalloc(&buffers.partials_a, partial_capacity * sizeof(float)),
             "cudaMalloc partials A");
  check_cuda(cudaMalloc(&buffers.partials_b, partial_capacity * sizeof(float)),
             "cudaMalloc partials B");

  buffers.scratch_capacity = scratch_capacity;
  buffers.partial_capacity = partial_capacity;
}

void free_device_reduction_buffers(DeviceReductionBuffers &buffers) {
  cudaFree(buffers.scratch);
  cudaFree(buffers.partials_a);
  cudaFree(buffers.partials_b);

  buffers = {};
}

float simple_stride_sum(const float *input, std::size_t size) {
  if (size == 0) {
    return 0.0f;
  }

  float *device_input = nullptr;
  DeviceReductionBuffers buffers;

  check_cuda(cudaMalloc(&device_input, size * sizeof(float)),
             "cudaMalloc input");
  try {
    check_cuda(cudaMemcpy(device_input, input, size * sizeof(float),
                          cudaMemcpyHostToDevice),
               "cudaMemcpy input");
    allocate_device_reduction_buffers(buffers, size);
    const float result =
        simple_stride_sum_device(device_input, size, buffers);
    free_device_reduction_buffers(buffers);
    check_cuda(cudaFree(device_input), "cudaFree input");
    return result;
  } catch (...) {
    free_device_reduction_buffers(buffers);
    cudaFree(device_input);
    throw;
  }
}

float sequential_addressing_sum(const float *input, std::size_t size) {
  if (size == 0) {
    return 0.0f;
  }

  float *device_input = nullptr;
  DeviceReductionBuffers buffers;

  check_cuda(cudaMalloc(&device_input, size * sizeof(float)),
             "cudaMalloc input");
  try {
    check_cuda(cudaMemcpy(device_input, input, size * sizeof(float),
                          cudaMemcpyHostToDevice),
               "cudaMemcpy input");
    allocate_device_reduction_buffers(buffers, size);
    const float result =
        sequential_addressing_sum_device(device_input, size, buffers);
    free_device_reduction_buffers(buffers);
    check_cuda(cudaFree(device_input), "cudaFree input");
    return result;
  } catch (...) {
    free_device_reduction_buffers(buffers);
    cudaFree(device_input);
    throw;
  }
}

float coarsened_shared_memory_sum(const float *input, std::size_t size) {
  if (size == 0) {
    return 0.0f;
  }

  float *device_input = nullptr;
  DeviceReductionBuffers buffers;

  check_cuda(cudaMalloc(&device_input, size * sizeof(float)),
             "cudaMalloc input");
  try {
    check_cuda(cudaMemcpy(device_input, input, size * sizeof(float),
                          cudaMemcpyHostToDevice),
               "cudaMemcpy input");
    allocate_device_reduction_buffers(buffers, size);
    const float result =
        coarsened_shared_memory_sum_device(device_input, size, buffers);
    free_device_reduction_buffers(buffers);
    check_cuda(cudaFree(device_input), "cudaFree input");
    return result;
  } catch (...) {
    free_device_reduction_buffers(buffers);
    cudaFree(device_input);
    throw;
  }
}

float coarsened_shared_memory_sum_optimized(const float *input,
                                            std::size_t size) {
  if (size == 0) {
    return 0.0f;
  }

  float *device_input = nullptr;
  DeviceReductionBuffers buffers;

  check_cuda(cudaMalloc(&device_input, size * sizeof(float)),
             "cudaMalloc input");
  try {
    check_cuda(cudaMemcpy(device_input, input, size * sizeof(float),
                          cudaMemcpyHostToDevice),
               "cudaMemcpy input");
    allocate_device_reduction_buffers(buffers, size);
    const float result = coarsened_shared_memory_sum_optimized_device(
        device_input, size, buffers);
    free_device_reduction_buffers(buffers);
    check_cuda(cudaFree(device_input), "cudaFree input");
    return result;
  } catch (...) {
    free_device_reduction_buffers(buffers);
    cudaFree(device_input);
    throw;
  }
}

float thrust_reference_sum(const float *input, std::size_t size) {
  if (size == 0) {
    return 0.0f;
  }

  thrust::device_vector<float> values(input, input + size);
  return thrust::reduce(values.begin(), values.end(), 0.0f,
                        cuda::std::plus<float>{});
}

float simple_stride_sum_device(const float *device_input, std::size_t size,
                               DeviceReductionBuffers &buffers) {
  return run_device_sum(device_input, size, SumKernel::SimpleStride, buffers);
}

float sequential_addressing_sum_device(const float *device_input,
                                       std::size_t size,
                                       DeviceReductionBuffers &buffers) {
  return run_device_sum(device_input, size, SumKernel::SequentialAddressing,
                        buffers);
}

float coarsened_shared_memory_sum_device(const float *device_input,
                                          std::size_t size,
                                          DeviceReductionBuffers &buffers) {
  return run_device_sum(device_input, size, SumKernel::CoarsenedSharedMemory,
                        buffers);
}

float coarsened_shared_memory_sum_optimized_device(
    const float *device_input, std::size_t size,
    DeviceReductionBuffers &buffers) {
  return run_device_sum(device_input, size,
                        SumKernel::CoarsenedSharedMemoryOptimized, buffers);
}

float thrust_reference_sum_device(const float *device_input, std::size_t size) {
  if (size == 0) {
    return 0.0f;
  }

  thrust::device_ptr<const float> first =
      thrust::device_pointer_cast(device_input);
  return thrust::reduce(first, first + size, 0.0f, cuda::std::plus<float>{});
}

} // namespace pmpp::reduction
