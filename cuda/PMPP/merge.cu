#include "merge.cuh"

#include <algorithm>
#include <climits>
#include <cuda_runtime.h>
#include <iostream>

namespace PMPP {
__device__ __forceinline__ size_t merge_path(const int *A, size_t m, const int *B,
                                          size_t n, size_t diag) {
  size_t low = (diag > n) ? (diag - n) : 0;
  size_t high = (diag < m) ? diag : m;

  while (low < high) {
    size_t mid = (low + high) >> 1;
    int a_key = (mid < m) ? A[mid] : INT_MAX;
    int b_key = (mid < diag) ? B[diag - mid - 1] : INT_MIN;

    if (a_key < b_key) {
      low = mid + 1;
    } else {
      high = mid;
    }
  }
  return low;
}

__global__ void merge_path_partitions_kernel(const int *A, size_t m, const int *B,
                                             size_t n, size_t *a_offsets,
                                             size_t *b_offsets,
                                             size_t num_partitions) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= num_partitions + 1) {
    return;
  }

  size_t total = m + n;
  size_t diag =
      static_cast<size_t>((static_cast<unsigned long long>(idx) * total) / num_partitions);
  size_t a_idx = merge_path(A, m, B, n, diag);
  size_t b_idx = diag - a_idx;

  a_offsets[idx] = a_idx;
  b_offsets[idx] = b_idx;
}

__device__ __host__ void merge_sequential(const int *A_start,
                                          const int *B_start, int *C_start,
                                          size_t a_size, size_t b_size) {
  size_t i = 0;
  size_t j = 0;

  while (i < a_size && j < b_size) {
    if (A_start[i] <= B_start[j]) {
      C_start[i + j] = A_start[i];
      ++i;
    } else {
      C_start[i + j] = B_start[j];
      ++j;
    }
  }

  while (i < a_size) {
    C_start[i + j] = A_start[i];
    ++i;
  }

  while (j < b_size) {
    C_start[i + j] = B_start[j];
    ++j;
  }
}

// Shared by both shared-memory kernels: merge one tile from A_shared/B_shared
// to C.
__device__ void merge_tile_from_shared(const int *A_shared, size_t a_tile_count,
                                       const int *B_shared, size_t b_tile_count,
                                       int *C_dst) {
  size_t t = threadIdx.x;
  size_t block_threads = blockDim.x;
  size_t tile_total = a_tile_count + b_tile_count;

  size_t diag_start = (static_cast<unsigned long long>(t) * tile_total) / block_threads;
  size_t diag_end = (static_cast<unsigned long long>(t + 1) * tile_total) / block_threads;

  size_t a_start =
      merge_path(A_shared, a_tile_count, B_shared, b_tile_count, diag_start);
  size_t a_end =
      merge_path(A_shared, a_tile_count, B_shared, b_tile_count, diag_end);
  size_t b_start = diag_start - a_start;
  size_t b_end = diag_end - a_end;

  const size_t a_thread_count = a_end - a_start;
  const size_t b_thread_count = b_end - b_start;

  if (a_thread_count + b_thread_count > 0) {
    merge_sequential(A_shared + a_start, B_shared + b_start, C_dst + diag_start,
                     a_thread_count, b_thread_count);
  }
}

__global__ void merge_partitions_kernel(const int *A, const int *B, int *C,
                                        const size_t *a_offsets,
                                        const size_t *b_offsets) {
  size_t block = blockIdx.x;
  size_t a0 = a_offsets[block];
  size_t a1 = a_offsets[block + 1];
  size_t b0 = b_offsets[block];
  size_t b1 = b_offsets[block + 1];

  size_t a_count = a1 - a0;
  size_t b_count = b1 - b0;
  size_t local_total = a_count + b_count;
  if (local_total == 0) {
    return;
  }

  const int *A_block = A + a0;
  const int *B_block = B + b0;
  int *C_block = C + a0 + b0;

  size_t t = threadIdx.x;
  size_t block_threads = blockDim.x;

  size_t diag_start = (static_cast<unsigned long long>(t) * local_total) / block_threads;
  size_t diag_end = (static_cast<unsigned long long>(t + 1) * local_total) / block_threads;

  size_t a_start = merge_path(A_block, a_count, B_block, b_count, diag_start);
  size_t a_end = merge_path(A_block, a_count, B_block, b_count, diag_end);
  size_t b_start = diag_start - a_start;
  size_t b_end = diag_end - a_end;

  const size_t a_thread_count = a_end - a_start;
  const size_t b_thread_count = b_end - b_start;
  if (a_thread_count + b_thread_count == 0) {
    return;
  }

  const int *A_thread_start = A_block + a_start;
  const int *B_thread_start = B_block + b_start;
  int *C_thread_start = C_block + diag_start;

  merge_sequential(A_thread_start, B_thread_start, C_thread_start,
                   a_thread_count, b_thread_count);
}

// Original shared version: load up to tile_size from A and B each, merge one
// tile (tile_size output elements), advance. Uses 2*tile_size shared memory for
// A_shared and B_shared.
__global__ void merge_partitions_kernel_shared_tiled(const int *A, const int *B,
                                                     int *C,
                                                     const size_t *a_offsets,
                                                     const size_t *b_offsets,
                                                     size_t tile_size) {
  extern __shared__ int shared_mem[];

  size_t block = blockIdx.x;
  size_t a0 = a_offsets[block];
  size_t a1 = a_offsets[block + 1];
  size_t b0 = b_offsets[block];
  size_t b1 = b_offsets[block + 1];

  size_t a_count = a1 - a0;
  size_t b_count = b1 - b0;
  size_t local_total = a_count + b_count;
  if (local_total == 0) {
    return;
  }

  int *A_shared = shared_mem;
  int *B_shared = A_shared + tile_size;

  const int *A_ptr = A + a0;
  const int *B_ptr = B + b0;
  int *C_ptr = C + a0 + b0;

  size_t t = threadIdx.x;
  size_t block_threads = blockDim.x;

  size_t a_remaining = a_count;
  size_t b_remaining = b_count;
  const size_t total_iterations = (local_total + tile_size - 1) / tile_size;

  for (size_t iter = 0; iter < total_iterations; ++iter) {
    size_t a_tile = (tile_size < a_remaining) ? tile_size : a_remaining;
    size_t b_tile = (tile_size < b_remaining) ? tile_size : b_remaining;
    const size_t merge_size = a_tile + b_tile;

    for (size_t i = t; i < a_tile; i += block_threads) {
      A_shared[i] = A_ptr[i];
    }
    for (size_t i = t; i < b_tile; i += block_threads) {
      B_shared[i] = B_ptr[i];
    }
    __syncthreads();

    merge_tile_from_shared(A_shared, a_tile, B_shared, b_tile, C_ptr);

    __syncthreads();

    size_t output_count = (tile_size < merge_size) ? tile_size : merge_size;
    size_t A_processed =
        merge_path(A_shared, a_tile, B_shared, b_tile, output_count);
    size_t B_processed = output_count - A_processed;
    A_ptr += A_processed;
    B_ptr += B_processed;
    C_ptr += output_count;
    a_remaining -= A_processed;
    b_remaining -= B_processed;
  }
}

// Partitioned shared version: pre-partition block into tile_size chunks, then
// load exactly tile_size elements per tile. Uses extra shared memory for tile
// offsets.
__global__ void merge_partitions_kernel_shared_partitioned(
    const int *A, const int *B, int *C, const size_t *a_offsets,
    const size_t *b_offsets, size_t tile_size, size_t max_num_tiles) {
  extern __shared__ int shared_mem[];

  size_t block = blockIdx.x;
  size_t a0 = a_offsets[block];
  size_t a1 = a_offsets[block + 1];
  size_t b0 = b_offsets[block];
  size_t b1 = b_offsets[block + 1];

  size_t a_count = a1 - a0;
  size_t b_count = b1 - b0;
  size_t local_total = a_count + b_count;
  if (local_total == 0) {
    return;
  }

  const int *A_block = A + a0;
  const int *B_block = B + b0;
  int *C_block = C + a0 + b0;

  size_t t = threadIdx.x;
  size_t block_threads = blockDim.x;

  const size_t num_tiles = (local_total + tile_size - 1) / tile_size;

  int *A_shared = shared_mem;
  int *B_shared = nullptr;
  size_t *a_tile_offsets = reinterpret_cast<size_t*>(shared_mem + tile_size);
  size_t *b_tile_offsets = a_tile_offsets + (max_num_tiles + 1);

  for (size_t i = t; i <= num_tiles; i += block_threads) {
    size_t diag = (i == num_tiles) ? local_total : (i * tile_size);
    size_t a_idx = merge_path(A_block, a_count, B_block, b_count, diag);
    size_t b_idx = diag - a_idx;
    a_tile_offsets[i] = a_idx;
    b_tile_offsets[i] = b_idx;
  }
  __syncthreads();

  for (size_t tile = 0; tile < num_tiles; ++tile) {
    size_t a_tile_start = a_tile_offsets[tile];
    size_t a_tile_end = a_tile_offsets[tile + 1];
    size_t b_tile_start = b_tile_offsets[tile];
    size_t b_tile_end = b_tile_offsets[tile + 1];

    size_t a_tile_count = a_tile_end - a_tile_start;
    size_t b_tile_count = b_tile_end - b_tile_start;

    const int *A_tile_src = A_block + a_tile_start;
    const int *B_tile_src = B_block + b_tile_start;
    int *C_tile_dst = C_block + (tile * tile_size);

    for (size_t i = t; i < a_tile_count; i += block_threads) {
      A_shared[i] = A_tile_src[i];
    }
    B_shared = A_shared + a_tile_count;
    for (size_t i = t; i < b_tile_count; i += block_threads) {
      B_shared[i] = B_tile_src[i];
    }
    __syncthreads();

    merge_tile_from_shared(A_shared, a_tile_count, B_shared, b_tile_count,
                           C_tile_dst);

    __syncthreads();
  }
}

enum class MergeKernelType { Simple, Shared, SharedPartitioned };

int get_sm_count() {
  cudaDeviceProp prop{};
  int device = 0;
  cudaGetDevice(&device);
  cudaGetDeviceProperties(&prop, device);
  return prop.multiProcessorCount;
}

int get_shared_memory_size() {
  cudaDeviceProp prop{};
  int device = 0;
  cudaGetDevice(&device);
  cudaGetDeviceProperties(&prop, device);
  return prop.sharedMemPerBlock;
}

template <MergeKernelType KernelType>
void merge_impl_device(int *d_input1, size_t size1, int *d_input2, size_t size2,
                       int *d_output) {
  size_t total = size1 + size2;
  if (total == 0) {
    return;
  }

  const size_t block_size = 256;
  const size_t elements_per_block = block_size * 8;

  int sm_count = std::max(1, get_sm_count());
  size_t desired_blocks = (total + elements_per_block - 1) / elements_per_block;
  size_t num_blocks = std::min(static_cast<size_t>(sm_count), std::max(static_cast<size_t>(1), desired_blocks));

  size_t tile_size = 0;
  size_t shared_mem_size = 0;
  size_t max_num_tiles = 0;

  if constexpr (KernelType == MergeKernelType::Shared) {
    int shared_mem_bytes = get_shared_memory_size();
    tile_size = static_cast<size_t>(0.9 * shared_mem_bytes / (2 * sizeof(int)));
    tile_size = std::max(static_cast<size_t>(256), std::min(tile_size, static_cast<size_t>(4096)));
    shared_mem_size = tile_size * 2 * sizeof(int);
  } else if constexpr (KernelType == MergeKernelType::SharedPartitioned) {
    int shared_mem_bytes = get_shared_memory_size();
    const size_t max_partition_size = (total + num_blocks - 1) / num_blocks;
    tile_size = 256;
    for (size_t test_tile_size = 256; test_tile_size <= 9192;
         test_tile_size += 256) {
      size_t test_max_num_tiles =
          (max_partition_size + test_tile_size - 1) / test_tile_size;
      size_t needed_bytes =
          (test_tile_size * sizeof(int) + (test_max_num_tiles + 1) * 2 * sizeof(size_t));
      if (needed_bytes <= static_cast<size_t>(0.9 * shared_mem_bytes)) {
        tile_size = test_tile_size;
        max_num_tiles = test_max_num_tiles;
      } else {
        break;
      }
    }
    shared_mem_size = (tile_size * sizeof(int) + (max_num_tiles + 1) * 2 * sizeof(size_t));
  }

  size_t *d_a_offsets = nullptr;
  size_t *d_b_offsets = nullptr;

  cudaMalloc(&d_a_offsets, (num_blocks + 1) * sizeof(size_t));
  cudaMalloc(&d_b_offsets, (num_blocks + 1) * sizeof(size_t));

  size_t partition_threads = 256;
  size_t partition_blocks =
      (num_blocks + 1 + partition_threads - 1) / partition_threads;
  merge_path_partitions_kernel<<<partition_blocks, partition_threads>>>(
      d_input1, size1, d_input2, size2, d_a_offsets, d_b_offsets, num_blocks);

  if constexpr (KernelType == MergeKernelType::Simple) {
    merge_partitions_kernel<<<num_blocks, block_size>>>(
        d_input1, d_input2, d_output, d_a_offsets, d_b_offsets);
  } else if constexpr (KernelType == MergeKernelType::Shared) {
    merge_partitions_kernel_shared_tiled<<<num_blocks, block_size,
                                           shared_mem_size>>>(
        d_input1, d_input2, d_output, d_a_offsets, d_b_offsets, tile_size);
  } else if constexpr (KernelType == MergeKernelType::SharedPartitioned) {
    merge_partitions_kernel_shared_partitioned<<<num_blocks, block_size,
                                                 shared_mem_size>>>(
        d_input1, d_input2, d_output, d_a_offsets, d_b_offsets, tile_size,
        max_num_tiles);
  }

  cudaDeviceSynchronize();

  cudaFree(d_a_offsets);
  cudaFree(d_b_offsets);
}

template <MergeKernelType KernelType>
void merge_impl(int *input1, size_t size1, int *input2, size_t size2, int *output) {
  size_t total = size1 + size2;
  if (total == 0) {
    return;
  }

  int *d_a = nullptr;
  int *d_b = nullptr;
  int *d_c = nullptr;

  cudaMalloc(&d_a, size1 * sizeof(int));
  cudaMalloc(&d_b, size2 * sizeof(int));
  cudaMalloc(&d_c, total * sizeof(int));

  if (size1 > 0) {
    cudaMemcpy(d_a, input1, size1 * sizeof(int), cudaMemcpyHostToDevice);
  }
  if (size2 > 0) {
    cudaMemcpy(d_b, input2, size2 * sizeof(int), cudaMemcpyHostToDevice);
  }

  merge_impl_device<KernelType>(d_a, size1, d_b, size2, d_c);

  cudaMemcpy(output, d_c, total * sizeof(int), cudaMemcpyDeviceToHost);

  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);
}

void merge(int *input1, size_t size1, int *input2, size_t size2, int *output) {
  merge_impl<MergeKernelType::Simple>(input1, size1, input2, size2, output);
}

void merge_shared(int *input1, size_t size1, int *input2, size_t size2, int *output) {
  merge_impl<MergeKernelType::Shared>(input1, size1, input2, size2, output);
}

void merge_shared_partitioned(int *input1, size_t size1, int *input2, size_t size2,
                              int *output) {
  merge_impl<MergeKernelType::SharedPartitioned>(input1, size1, input2, size2,
                                                 output);
}

void merge_device(int *d_input1, size_t size1, int *d_input2, size_t size2,
                  int *d_output) {
  merge_impl_device<MergeKernelType::Simple>(d_input1, size1, d_input2, size2,
                                             d_output);
}

void merge_shared_device(int *d_input1, size_t size1, int *d_input2, size_t size2,
                         int *d_output) {
  merge_impl_device<MergeKernelType::Shared>(d_input1, size1, d_input2, size2,
                                             d_output);
}

void merge_shared_partitioned_device(int *d_input1, size_t size1, int *d_input2,
                                     size_t size2, int *d_output) {
  merge_impl_device<MergeKernelType::SharedPartitioned>(
      d_input1, size1, d_input2, size2, d_output);
}
} // namespace PMPP