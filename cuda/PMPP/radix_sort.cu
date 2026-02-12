#include "radix_sort.cuh"

#include <cuda_runtime.h>
#include <cub/cub.cuh>
#include <cstdio>
#include <algorithm>

namespace pmpp {
namespace radix_sort {

// CUDA error checking macro
#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(error)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

constexpr int BLOCK_SIZE = 256;

/**
 * @brief Calculate optimal grid size based on hardware capabilities
 */
int get_optimal_num_blocks(size_t n, int items_per_thread = 4) {
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    
    // Calculate number of threads needed
    size_t total_threads = (n + items_per_thread - 1) / items_per_thread;
    int num_blocks = (total_threads + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
    // Limit based on GPU capability
    // Heuristic: aim for multiple waves of blocks per SM
    int max_blocks_per_sm = prop.maxThreadsPerMultiProcessor / BLOCK_SIZE;
    int suggested_blocks = prop.multiProcessorCount * max_blocks_per_sm * 2;
    
    return std::min(num_blocks, suggested_blocks);
}

/**
 * @brief Compute exclusive prefix sum using CUB
 */
void exclusive_scan(uint32_t* d_in, uint32_t* d_out, size_t n) {
    void* d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;
    
    // Determine temporary device storage requirements
    cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, d_in, d_out, n);
    
    // Allocate temporary storage
    CUDA_CHECK(cudaMalloc(&d_temp_storage, temp_storage_bytes));
    
    // Run exclusive prefix sum
    cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, d_in, d_out, n);
    
    // Free temporary storage
    CUDA_CHECK(cudaFree(d_temp_storage));
}

/**
 * @brief Extract N-bit digit values for all elements with block partitioning
 * Each block processes a contiguous chunk of data for better memory coalescing
 * 
 * @tparam RADIX_BITS Number of bits to extract
 */
template<int RADIX_BITS>
__global__ void extract_digits(const uint32_t* input, uint32_t* digits, size_t n, int start_bit) {
    constexpr uint32_t MASK = (1u << RADIX_BITS) - 1;
    
    // Compute chunk boundaries for this block
    size_t chunk_size = (n + gridDim.x - 1) / gridDim.x;
    size_t block_start = blockIdx.x * chunk_size;
    size_t block_end = min(block_start + chunk_size, n);
    
    // Each thread processes elements within this block's chunk
    for (size_t idx = block_start + threadIdx.x; idx < block_end; idx += blockDim.x) {
        digits[idx] = (input[idx] >> start_bit) & MASK;
    }
}

/**
 * @brief Compute per-block histograms of digit values using shared memory.
 *
 * Each block processes a contiguous chunk of data and builds a local histogram
 * in shared memory, then writes it to a global 2D array `block_hist` of size
 * (num_blocks x NUM_BUCKETS).
 *
 * No global atomics are used; only shared-memory atomics inside a block.
 *
 * @tparam RADIX_BITS Number of bits in the digit
 */
template<int RADIX_BITS>
__global__ void compute_block_histogram(const uint32_t* digits, size_t n,
                                        uint32_t* block_hist) {
    constexpr int NUM_BUCKETS = 1 << RADIX_BITS;

    __shared__ uint32_t local_hist[NUM_BUCKETS];

    // Initialize shared histogram
    for (int i = threadIdx.x; i < NUM_BUCKETS; i += blockDim.x) {
        local_hist[i] = 0;
    }
    __syncthreads();

    // Compute chunk boundaries for this block
    size_t chunk_size = (n + gridDim.x - 1) / gridDim.x;
    size_t block_start = blockIdx.x * chunk_size;
    size_t block_end = min(block_start + chunk_size, n);

    // Compute local histogram for this block's chunk
    for (size_t idx = block_start + threadIdx.x; idx < block_end; idx += blockDim.x) {
        uint32_t d = digits[idx];
        atomicAdd(&local_hist[d], 1);
    }
    __syncthreads();

    // Write this block's histogram to global memory
    uint32_t* block_hist_row = block_hist + blockIdx.x * NUM_BUCKETS;
    for (int i = threadIdx.x; i < NUM_BUCKETS; i += blockDim.x) {
        block_hist_row[i] = local_hist[i];
    }
}

/**
 * @brief Reduce per-block histograms into a single global histogram.
 *
 * For each bucket, a single thread sums that bucket's counts across all blocks.
 *
 * @tparam RADIX_BITS Number of bits in the digit
 */
template<int RADIX_BITS>
__global__ void reduce_block_histogram(const uint32_t* block_hist,
                                       uint32_t* histogram,
                                       int num_blocks) {
    constexpr int NUM_BUCKETS = 1 << RADIX_BITS;

    int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= NUM_BUCKETS) return;

    uint32_t sum = 0;
    for (int blk = 0; blk < num_blocks; ++blk) {
        sum += block_hist[blk * NUM_BUCKETS + b];
    }
    histogram[b] = sum;
}

/**
 * @brief Compute per-block prefix offsets for each bucket.
 *
 * For each bucket b, we compute an exclusive prefix sum over blocks:
 *   block_offsets[blk, b] = sum_{k < blk} block_hist[k, b]
 *
 * @tparam RADIX_BITS Number of bits in the digit
 */
template<int RADIX_BITS>
__global__ void compute_block_offsets(const uint32_t* block_hist,
                                      uint32_t* block_offsets,
                                      int num_blocks) {
    constexpr int NUM_BUCKETS = 1 << RADIX_BITS;

    int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= NUM_BUCKETS) return;

    uint32_t running = 0;
    for (int blk = 0; blk < num_blocks; ++blk) {
        int idx = blk * NUM_BUCKETS + b;
        uint32_t count = block_hist[idx];
        block_offsets[idx] = running;
        running += count;
    }
}

/**
 * @brief Scatter elements based on digit values using hierarchical offsets.
 *
 * This kernel uses:
 *  - `bucket_positions[b]`: global start offset of bucket b
 *  - `block_offsets[blk, b]`: prefix sum of counts for bucket b over blocks < blk
 *
 * Each block processes a contiguous chunk of data for better memory coalescing.
 * Within a block, we maintain shared-memory counters per bucket to obtain
 * the intra-block offset. No global atomics are used.
 * 
 * @tparam RADIX_BITS Number of bits in the digit
 * @param input Input array
 * @param output Output array
 * @param n Number of elements
 * @param digits Digit values for each element
 * @param bucket_positions Starting position for each bucket (global prefix sum)
 * @param block_offsets Per-block prefix offsets for each bucket
 */
template<int RADIX_BITS>
__global__ void scatter_by_digit_hierarchical(const uint32_t* input, uint32_t* output,
                                              size_t n,
                                              const uint32_t* digits,
                                              const uint32_t* bucket_positions,
                                              const uint32_t* block_offsets) {
    constexpr int NUM_BUCKETS = 1 << RADIX_BITS;

    __shared__ uint32_t block_bucket_base[NUM_BUCKETS];
    __shared__ uint32_t local_bucket_counters[NUM_BUCKETS];

    // Compute this block's base offset for each bucket and zero local counters
    const uint32_t* block_offset_row = block_offsets + blockIdx.x * NUM_BUCKETS;

    for (int i = threadIdx.x; i < NUM_BUCKETS; i += blockDim.x) {
        block_bucket_base[i] = bucket_positions[i] + block_offset_row[i];
        local_bucket_counters[i] = 0;
    }
    __syncthreads();

    // Compute chunk boundaries for this block
    size_t chunk_size = (n + gridDim.x - 1) / gridDim.x;
    size_t block_start = blockIdx.x * chunk_size;
    size_t block_end = min(block_start + chunk_size, n);

    // Process elements within this block's chunk
    for (size_t idx = block_start + threadIdx.x; idx < block_end; idx += blockDim.x) {
        uint32_t value = input[idx];
        uint32_t digit = digits[idx];

        // Get intra-block position within this bucket (shared memory only)
        uint32_t local_idx = atomicAdd(&local_bucket_counters[digit], 1);
        uint32_t out_pos = block_bucket_base[digit] + local_idx;
        output[out_pos] = value;
    }
}

template<int RADIX_BITS>
void radix_sort(const uint32_t* d_input, uint32_t* d_output, size_t n) {
    static_assert(32 % RADIX_BITS == 0, "RADIX_BITS must divide 32 evenly");
    
    if (n == 0) return;
    
    constexpr int NUM_BUCKETS = 1 << RADIX_BITS;
    constexpr int NUM_ITERATIONS = 32 / RADIX_BITS;
    
    // Calculate optimal number of blocks based on hardware
    int num_blocks = get_optimal_num_blocks(n);

    // Allocate temporary buffers
    uint32_t *d_temp, *d_digits, *d_histogram, *d_bucket_positions;
    uint32_t *d_block_hist = nullptr, *d_block_offsets = nullptr;

    CUDA_CHECK(cudaMalloc(&d_temp, n * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_digits, n * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_histogram, NUM_BUCKETS * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_bucket_positions, NUM_BUCKETS * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_block_hist, num_blocks * NUM_BUCKETS * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_block_offsets, num_blocks * NUM_BUCKETS * sizeof(uint32_t)));
    
    // Copy input to output initially
    CUDA_CHECK(cudaMemcpy(d_output, d_input, n * sizeof(uint32_t), cudaMemcpyDeviceToDevice));
    
    // Process RADIX_BITS bits per iteration from LSB to MSB
    for (int iter = 0; iter < NUM_ITERATIONS; iter++) {
        int start_bit = iter * RADIX_BITS;
        
        // Step 1: Extract digit values
        extract_digits<RADIX_BITS><<<num_blocks, BLOCK_SIZE>>>(d_output, d_digits, n, start_bit);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        
        // Step 2: Compute per-block histograms (no global atomics)
        compute_block_histogram<RADIX_BITS><<<num_blocks, BLOCK_SIZE>>>(d_digits, n, d_block_hist);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        
        // Step 3: Reduce per-block histograms to global histogram
        {
            int threads = NUM_BUCKETS < BLOCK_SIZE ? NUM_BUCKETS : BLOCK_SIZE;
            int blocks = 1;
            reduce_block_histogram<RADIX_BITS><<<blocks, threads>>>(d_block_hist, d_histogram, num_blocks);
        }
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        
        // Step 4: Compute exclusive prefix sum of histogram to get bucket start positions
        exclusive_scan(d_histogram, d_bucket_positions, NUM_BUCKETS);
        CUDA_CHECK(cudaDeviceSynchronize());
        
        // Step 5: Compute per-block prefix offsets for each bucket
        {
            int threads = NUM_BUCKETS < BLOCK_SIZE ? NUM_BUCKETS : BLOCK_SIZE;
            int blocks = 1;
            compute_block_offsets<RADIX_BITS><<<blocks, threads>>>(d_block_hist, d_block_offsets, num_blocks);
        }
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        
        // Step 6: Scatter elements to correct positions using hierarchical offsets
        scatter_by_digit_hierarchical<RADIX_BITS><<<num_blocks, BLOCK_SIZE>>>(
            d_output, d_temp, n, d_digits, d_bucket_positions, d_block_offsets);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        
        // Swap buffers
        uint32_t* swap = d_output;
        d_output = d_temp;
        d_temp = swap;
    }
    
    // After NUM_ITERATIONS, check if result is in right place
    if (NUM_ITERATIONS % 2 == 1) {
        // Odd iterations: result is in d_temp, need to copy to d_output
        CUDA_CHECK(cudaMemcpy((uint32_t*)d_input, d_temp, n * sizeof(uint32_t), cudaMemcpyDeviceToDevice));
    } else {
        // Even iterations: result is already in d_output (which is now d_temp after swaps)
        CUDA_CHECK(cudaMemcpy((uint32_t*)d_input, d_temp, n * sizeof(uint32_t), cudaMemcpyDeviceToDevice));
    }
    
    CUDA_CHECK(cudaFree(d_temp));
    CUDA_CHECK(cudaFree(d_digits));
    CUDA_CHECK(cudaFree(d_histogram));
    CUDA_CHECK(cudaFree(d_bucket_positions));
    CUDA_CHECK(cudaFree(d_block_hist));
    CUDA_CHECK(cudaFree(d_block_offsets));
}

template<int RADIX_BITS>
void radix_sort_inplace(uint32_t* d_data, size_t n) {
    static_assert(32 % RADIX_BITS == 0, "RADIX_BITS must divide 32 evenly");
    
    if (n == 0) return;
    
    constexpr int NUM_BUCKETS = 1 << RADIX_BITS;
    constexpr int NUM_ITERATIONS = 32 / RADIX_BITS;
    
    // Allocate temporary buffers
    uint32_t *d_temp, *d_digits, *d_histogram, *d_bucket_positions;
    uint32_t *d_block_hist = nullptr, *d_block_offsets = nullptr;

    // Calculate optimal number of blocks based on hardware
    int num_blocks = get_optimal_num_blocks(n);

    CUDA_CHECK(cudaMalloc(&d_temp, n * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_digits, n * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_histogram, NUM_BUCKETS * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_bucket_positions, NUM_BUCKETS * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_block_hist, num_blocks * NUM_BUCKETS * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_block_offsets, num_blocks * NUM_BUCKETS * sizeof(uint32_t)));
    
    uint32_t* current = d_data;
    uint32_t* next = d_temp;
    
    // Process RADIX_BITS bits per iteration from LSB to MSB
    for (int iter = 0; iter < NUM_ITERATIONS; iter++) {
        int start_bit = iter * RADIX_BITS;
        
        // Step 1: Extract digit values
        extract_digits<RADIX_BITS><<<num_blocks, BLOCK_SIZE>>>(current, d_digits, n, start_bit);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        
        // Step 2: Clear histogram
        // (no need to clear histogram when using per-block histograms)
        
        // Step 3: Compute per-block histograms (no global atomics)
        compute_block_histogram<RADIX_BITS><<<num_blocks, BLOCK_SIZE>>>(d_digits, n, d_block_hist);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        
        // Step 4: Reduce per-block histograms to global histogram
        {
            int threads = NUM_BUCKETS < BLOCK_SIZE ? NUM_BUCKETS : BLOCK_SIZE;
            int blocks = 1;
            reduce_block_histogram<RADIX_BITS><<<blocks, threads>>>(d_block_hist, d_histogram, num_blocks);
        }
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        
        // Step 5: Compute exclusive prefix sum of histogram to get bucket start positions
        exclusive_scan(d_histogram, d_bucket_positions, NUM_BUCKETS);
        CUDA_CHECK(cudaDeviceSynchronize());
        
        // Step 6: Compute per-block prefix offsets for each bucket
        {
            int threads = NUM_BUCKETS < BLOCK_SIZE ? NUM_BUCKETS : BLOCK_SIZE;
            int blocks = 1;
            compute_block_offsets<RADIX_BITS><<<blocks, threads>>>(d_block_hist, d_block_offsets, num_blocks);
        }
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        
        // Step 7: Scatter elements to correct positions using hierarchical offsets
        scatter_by_digit_hierarchical<RADIX_BITS><<<num_blocks, BLOCK_SIZE>>>(
            current, next, n, d_digits, d_bucket_positions, d_block_offsets);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        
        // Swap pointers
        uint32_t* swap = current;
        current = next;
        next = swap;
    }
    
    // After NUM_ITERATIONS, copy back to d_data if needed
    if (current != d_data) {
        CUDA_CHECK(cudaMemcpy(d_data, current, n * sizeof(uint32_t), cudaMemcpyDeviceToDevice));
    }
    
    CUDA_CHECK(cudaFree(d_temp));
    CUDA_CHECK(cudaFree(d_digits));
    CUDA_CHECK(cudaFree(d_histogram));
    CUDA_CHECK(cudaFree(d_bucket_positions));
    CUDA_CHECK(cudaFree(d_block_hist));
    CUDA_CHECK(cudaFree(d_block_offsets));
}

template<int RADIX_BITS>
void radix_sort_host(const uint32_t* h_input, uint32_t* h_output, size_t n) {
    static_assert(32 % RADIX_BITS == 0, "RADIX_BITS must divide 32 evenly");
    
    if (n == 0) return;
    
    // Allocate device memory
    uint32_t *d_input, *d_output;
    CUDA_CHECK(cudaMalloc(&d_input, n * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_output, n * sizeof(uint32_t)));
    
    // Copy input to device
    CUDA_CHECK(cudaMemcpy(d_input, h_input, n * sizeof(uint32_t), cudaMemcpyHostToDevice));
    
    // Sort
    radix_sort<RADIX_BITS>(d_input, d_output, n);
    
    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(h_output, d_output, n * sizeof(uint32_t), cudaMemcpyDeviceToHost));
    
    // Free device memory
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));
}

// Explicit instantiations for common radix bit sizes
template void radix_sort<1>(const uint32_t*, uint32_t*, size_t);
template void radix_sort<2>(const uint32_t*, uint32_t*, size_t);
template void radix_sort<4>(const uint32_t*, uint32_t*, size_t);
template void radix_sort<8>(const uint32_t*, uint32_t*, size_t);

template void radix_sort_inplace<1>(uint32_t*, size_t);
template void radix_sort_inplace<2>(uint32_t*, size_t);
template void radix_sort_inplace<4>(uint32_t*, size_t);
template void radix_sort_inplace<8>(uint32_t*, size_t);

template void radix_sort_host<1>(const uint32_t*, uint32_t*, size_t);
template void radix_sort_host<2>(const uint32_t*, uint32_t*, size_t);
template void radix_sort_host<4>(const uint32_t*, uint32_t*, size_t);
template void radix_sort_host<8>(const uint32_t*, uint32_t*, size_t);

} // namespace radix_sort
} // namespace pmpp
