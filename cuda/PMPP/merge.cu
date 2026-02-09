#include "merge.cuh"

#include <algorithm>
#include <climits>
#include <cuda_runtime.h>
#include <iostream>

namespace PMPP {
__device__ __forceinline__ int merge_path(const int *A, int m, const int *B,
                                          int n, int diag) {
  int low = max(0, diag - n);
  int high = min(diag, m);

  while (low < high) {
    int mid = (low + high) >> 1;
    int a_key = (mid < m) ? A[mid] : INT_MAX;
    int b_key = (diag - mid - 1 >= 0) ? B[diag - mid - 1] : INT_MIN;

    if (a_key < b_key) {
      low = mid + 1;
    } else {
      high = mid;
    }
  }
  return low;
}

__global__ void merge_path_partitions_kernel(const int *A, int m, const int *B,
                                             int n, int *a_offsets,
                                             int *b_offsets,
                                             int num_partitions) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= num_partitions + 1) {
    return;
  }

  int total = m + n;
  int diag =
      static_cast<int>((static_cast<long long>(idx) * total) / num_partitions);
  int a_idx = merge_path(A, m, B, n, diag);
  int b_idx = diag - a_idx;

  a_offsets[idx] = a_idx;
  b_offsets[idx] = b_idx;
}

__device__ __host__ void merge_sequential(const int *A_start, const int *B_start,
                                           int *C_start, int a_size, int b_size) {
  int i = 0;
  int j = 0;

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

__global__ void merge_partitions_kernel(const int *A, const int *B, int *C,
                                        const int *a_offsets,
                                        const int *b_offsets) {
  int block = blockIdx.x;
  int a0 = a_offsets[block];
  int a1 = a_offsets[block + 1];
  int b0 = b_offsets[block];
  int b1 = b_offsets[block + 1];

  int a_count = a1 - a0;
  int b_count = b1 - b0;
  int local_total = a_count + b_count;
  if (local_total == 0) {
    return;
  }

  const int *A_block = A + a0;
  const int *B_block = B + b0;
  int *C_block = C + a0 + b0;

  int t = threadIdx.x;
  int block_threads = blockDim.x;

  int diag_start = (static_cast<long long>(t) * local_total) / block_threads;
  int diag_end = (static_cast<long long>(t + 1) * local_total) / block_threads;

  int a_start = merge_path(A_block, a_count, B_block, b_count, diag_start);
  int a_end = merge_path(A_block, a_count, B_block, b_count, diag_end);
  int b_start = diag_start - a_start;
  int b_end = diag_end - a_end;

  const int a_thread_count = a_end - a_start;
  const int b_thread_count = b_end - b_start;
  if (a_thread_count + b_thread_count == 0) {
    return;
  }

  const int *A_thread_start = A_block + a_start;
  const int *B_thread_start = B_block + b_start;
  int *C_thread_start = C_block + diag_start;

  merge_sequential(A_thread_start, B_thread_start, C_thread_start,
                   a_thread_count, b_thread_count);
}

__global__ void merge_partitions_kernel_shared(const int *A, const int *B,
                                               int *C, const int *a_offsets,
                                               const int *b_offsets,
                                               int tile_size) {
  extern __shared__ int shared_mem[];

  int block = blockIdx.x;
  int a0 = a_offsets[block];
  int a1 = a_offsets[block + 1];
  int b0 = b_offsets[block];
  int b1 = b_offsets[block + 1];

  int a_count = a1 - a0;
  int b_count = b1 - b0;
  int local_total = a_count + b_count;
  if (local_total == 0) {
    return;
  }

  int *A_shared = shared_mem;
  int *B_shared = A_shared + tile_size;

  const int *A_ptr = A + a0;
  const int *B_ptr = B + b0;
  int *C_ptr = C + a0 + b0;

  int t = threadIdx.x;
  int block_threads = blockDim.x;

  int a_remaining = a_count;
  int b_remaining = b_count;

  const int total_iterations = (local_total + tile_size - 1) / tile_size;

  for (int iter = 0; iter < total_iterations; ++iter) {
    int a_tile = min(tile_size, a_remaining);
    int b_tile = min(tile_size, b_remaining);

    const int merge_size = a_tile + b_tile;

    // Copy A tile to shared memory
    for (int i = t; i < a_tile; i += block_threads) {
      A_shared[i] = A_ptr[i];
    }

    // Copy B tile to shared memory
    for (int i = t; i < b_tile; i += block_threads) {
      B_shared[i] = B_ptr[i];
    }

    __syncthreads();

    // Partition the tile using merge_path
    int diag_start = (static_cast<long long>(t) * tile_size) / block_threads;
    int diag_end = (static_cast<long long>(t + 1) * tile_size) / block_threads;

    diag_start = min(diag_start, merge_size);
    diag_end = min(diag_end, merge_size);

    int a_start = merge_path(A_shared, a_tile, B_shared, b_tile, diag_start);
    int a_end = merge_path(A_shared, a_tile, B_shared, b_tile, diag_end);
    int b_start = diag_start - a_start;
    int b_end = diag_end - a_end;

    const int a_thread_count = a_end - a_start;
    const int b_thread_count = b_end - b_start;

    if (a_thread_count + b_thread_count == 0) {
      continue;
    }

    // Merge this thread's partition into output
    merge_sequential(A_shared + a_start, B_shared + b_start, C_ptr + diag_start,
                     a_thread_count, b_thread_count);

    __syncthreads();

    // Advance pointers
    const int A_processed =
        merge_path(A_shared, a_tile, B_shared, b_tile, tile_size);
    const int B_processed = tile_size - A_processed;
    A_ptr += A_processed;
    B_ptr += B_processed;
    C_ptr += tile_size;
    a_remaining -= A_processed;
    b_remaining -= B_processed;
  }
}

enum class MergeKernelType { Simple, Shared };

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
void merge_impl_device(int *d_input1, int size1, int *d_input2, int size2, int *d_output) {
  int total = size1 + size2;
  if (total == 0) {
    return;
  }

  const int block_size = 256;
  const int elements_per_block = block_size * 8;

  int sm_count = std::max(1, get_sm_count());
  int desired_blocks = (total + elements_per_block - 1) / elements_per_block;
  int num_blocks = std::min(sm_count, std::max(1, desired_blocks));

  int tile_size = 0;
  int shared_mem_size = 0;

  if constexpr (KernelType == MergeKernelType::Shared) {
    int shared_mem_bytes = get_shared_memory_size();
    tile_size = static_cast<int>(0.9 * shared_mem_bytes / (2 * sizeof(int)));
    tile_size = std::max(256, std::min(tile_size, 4096));
    shared_mem_size = tile_size * 2 * sizeof(int);
  }

  int *d_a_offsets = nullptr;
  int *d_b_offsets = nullptr;

  cudaMalloc(&d_a_offsets, (num_blocks + 1) * sizeof(int));
  cudaMalloc(&d_b_offsets, (num_blocks + 1) * sizeof(int));

  int partition_threads = 256;
  int partition_blocks =
      (num_blocks + 1 + partition_threads - 1) / partition_threads;
  merge_path_partitions_kernel<<<partition_blocks, partition_threads>>>(
      d_input1, size1, d_input2, size2, d_a_offsets, d_b_offsets, num_blocks);

  if constexpr (KernelType == MergeKernelType::Simple) {
    merge_partitions_kernel<<<num_blocks, block_size>>>(d_input1, d_input2, d_output,
                                                        d_a_offsets, d_b_offsets);
  } else if constexpr (KernelType == MergeKernelType::Shared) {
    merge_partitions_kernel_shared<<<num_blocks, block_size, shared_mem_size>>>(
        d_input1, d_input2, d_output, d_a_offsets, d_b_offsets, tile_size);
  }

  cudaDeviceSynchronize();

  cudaFree(d_a_offsets);
  cudaFree(d_b_offsets);
}

template <MergeKernelType KernelType>
void merge_impl(int *input1, int size1, int *input2, int size2, int *output) {
  int total = size1 + size2;
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

void merge(int *input1, int size1, int *input2, int size2, int *output) {
  merge_impl<MergeKernelType::Simple>(input1, size1, input2, size2, output);
}

void merge_shared(int *input1, int size1, int *input2, int size2, int *output) {
  merge_impl<MergeKernelType::Shared>(input1, size1, input2, size2, output);
}

void merge_device(int *d_input1, int size1, int *d_input2, int size2, int *d_output) {
  merge_impl_device<MergeKernelType::Simple>(d_input1, size1, d_input2, size2, d_output);
}

void merge_shared_device(int *d_input1, int size1, int *d_input2, int size2, int *d_output) {
  merge_impl_device<MergeKernelType::Shared>(d_input1, size1, d_input2, size2, d_output);
}
} // namespace PMPP