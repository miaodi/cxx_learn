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
 * @brief Extract N-bit digit values for all elements with coarsening
 * Each thread processes multiple elements to reduce kernel launch overhead
 * 
 * @tparam RADIX_BITS Number of bits to extract
 */
template<int RADIX_BITS>
__global__ void extract_digits(const uint32_t* input, uint32_t* digits, size_t n, int start_bit) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    
    constexpr uint32_t MASK = (1u << RADIX_BITS) - 1;
    
    // Coarsening: each thread processes multiple elements
    for (size_t idx = tid; idx < n; idx += stride) {
        digits[idx] = (input[idx] >> start_bit) & MASK;
    }
}

/**
 * @brief Compute histogram of digit values using shared memory (for small RADIX_BITS)
 * 
 * @tparam RADIX_BITS Number of bits in the digit
 */
template<int RADIX_BITS>
__global__ void compute_histogram_shared(const uint32_t* digits, size_t n, uint32_t* histogram) {
    constexpr int NUM_BUCKETS = 1 << RADIX_BITS;
    
    __shared__ uint32_t local_hist[NUM_BUCKETS];
    
    // Initialize shared histogram
    for (int i = threadIdx.x; i < NUM_BUCKETS; i += blockDim.x) {
        local_hist[i] = 0;
    }
    __syncthreads();
    
    // Compute local histogram
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    
    for (size_t idx = tid; idx < n; idx += stride) {
        atomicAdd(&local_hist[digits[idx]], 1);
    }
    __syncthreads();
    
    // Write to global histogram
    for (int i = threadIdx.x; i < NUM_BUCKETS; i += blockDim.x) {
        atomicAdd(&histogram[i], local_hist[i]);
    }
}

/**
 * @brief Compute histogram using global atomics (for large RADIX_BITS)
 * 
 * @tparam RADIX_BITS Number of bits in the digit
 */
template<int RADIX_BITS>
__global__ void compute_histogram_global(const uint32_t* digits, size_t n, uint32_t* histogram) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    
    // Directly use global atomics for large histograms
    for (size_t idx = tid; idx < n; idx += stride) {
        atomicAdd(&histogram[digits[idx]], 1);
    }
}

/**
 * @brief Dispatcher that selects appropriate histogram kernel
 */
template<int RADIX_BITS>
void compute_histogram(const uint32_t* digits, size_t n, uint32_t* histogram, int num_blocks) {
    constexpr int NUM_BUCKETS = 1 << RADIX_BITS;
    constexpr size_t SHARED_MEM_SIZE = NUM_BUCKETS * sizeof(uint32_t);
    constexpr size_t MAX_SHARED_MEM = 48 * 1024; // 48KB typical limit
    
    if constexpr (SHARED_MEM_SIZE <= MAX_SHARED_MEM) {
        // Use shared memory for small histograms
        compute_histogram_shared<RADIX_BITS><<<num_blocks, BLOCK_SIZE>>>(digits, n, histogram);
    } else {
        // Use global atomics for large histograms
        compute_histogram_global<RADIX_BITS><<<num_blocks, BLOCK_SIZE>>>(digits, n, histogram);
    }
}

/**
 * @brief Scatter elements based on digit values and bucket positions with coarsening
 * Each thread processes multiple elements to improve throughput
 * 
 * @tparam RADIX_BITS Number of bits in the digit
 * @param input Input array
 * @param output Output array
 * @param n Number of elements
 * @param digits Digit values for each element
 * @param bucket_positions Starting position for each bucket (prefix sum of histogram)
 */
template<int RADIX_BITS>
__global__ void scatter_by_digit(const uint32_t* input, uint32_t* output, size_t n,
                                 const uint32_t* digits, uint32_t* bucket_positions) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    
    // Coarsening: each thread processes multiple elements
    for (size_t idx = tid; idx < n; idx += stride) {
        uint32_t value = input[idx];
        uint32_t digit = digits[idx];
        
        // Atomically get position within bucket and increment
        uint32_t out_pos = atomicAdd(&bucket_positions[digit], 1);
        output[out_pos] = value;
    }
}

template<int RADIX_BITS>
void radix_sort(const uint32_t* d_input, uint32_t* d_output, size_t n) {
    static_assert(32 % RADIX_BITS == 0, "RADIX_BITS must divide 32 evenly");
    
    if (n == 0) return;
    
    constexpr int NUM_BUCKETS = 1 << RADIX_BITS;
    constexpr int NUM_ITERATIONS = 32 / RADIX_BITS;
    
    // Allocate temporary buffers
    uint32_t *d_temp, *d_digits, *d_histogram, *d_bucket_positions;
    CUDA_CHECK(cudaMalloc(&d_temp, n * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_digits, n * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_histogram, NUM_BUCKETS * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_bucket_positions, NUM_BUCKETS * sizeof(uint32_t)));
    
    // Copy input to output initially
    CUDA_CHECK(cudaMemcpy(d_output, d_input, n * sizeof(uint32_t), cudaMemcpyDeviceToDevice));
    
    // Calculate optimal number of blocks based on hardware
    int num_blocks = get_optimal_num_blocks(n);
    
    // Process RADIX_BITS bits per iteration from LSB to MSB
    for (int iter = 0; iter < NUM_ITERATIONS; iter++) {
        int start_bit = iter * RADIX_BITS;
        
        // Step 1: Extract digit values
        extract_digits<RADIX_BITS><<<num_blocks, BLOCK_SIZE>>>(d_output, d_digits, n, start_bit);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        
        // Step 2: Clear histogram
        CUDA_CHECK(cudaMemset(d_histogram, 0, NUM_BUCKETS * sizeof(uint32_t)));
        
        // Step 3: Compute histogram
        compute_histogram<RADIX_BITS>(d_digits, n, d_histogram, num_blocks);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        
        // Step 4: Compute exclusive prefix sum of histogram to get bucket start positions
        exclusive_scan(d_histogram, d_bucket_positions, NUM_BUCKETS);
        CUDA_CHECK(cudaDeviceSynchronize());
        
        // Step 5: Scatter elements to correct positions
        scatter_by_digit<RADIX_BITS><<<num_blocks, BLOCK_SIZE>>>(d_output, d_temp, n, d_digits, 
                                                                   d_bucket_positions);
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
}

template<int RADIX_BITS>
void radix_sort_inplace(uint32_t* d_data, size_t n) {
    static_assert(32 % RADIX_BITS == 0, "RADIX_BITS must divide 32 evenly");
    
    if (n == 0) return;
    
    constexpr int NUM_BUCKETS = 1 << RADIX_BITS;
    constexpr int NUM_ITERATIONS = 32 / RADIX_BITS;
    
    // Allocate temporary buffers
    uint32_t *d_temp, *d_digits, *d_histogram, *d_bucket_positions;
    CUDA_CHECK(cudaMalloc(&d_temp, n * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_digits, n * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_histogram, NUM_BUCKETS * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_bucket_positions, NUM_BUCKETS * sizeof(uint32_t)));
    
    uint32_t* current = d_data;
    uint32_t* next = d_temp;
    
    // Calculate optimal number of blocks based on hardware
    int num_blocks = get_optimal_num_blocks(n);
    
    // Process RADIX_BITS bits per iteration from LSB to MSB
    for (int iter = 0; iter < NUM_ITERATIONS; iter++) {
        int start_bit = iter * RADIX_BITS;
        
        // Step 1: Extract digit values
        extract_digits<RADIX_BITS><<<num_blocks, BLOCK_SIZE>>>(current, d_digits, n, start_bit);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        
        // Step 2: Clear histogram
        CUDA_CHECK(cudaMemset(d_histogram, 0, NUM_BUCKETS * sizeof(uint32_t)));
        
        // Step 3: Compute histogram
        compute_histogram<RADIX_BITS>(d_digits, n, d_histogram, num_blocks);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        
        // Step 4: Compute exclusive prefix sum of histogram to get bucket start positions
        exclusive_scan(d_histogram, d_bucket_positions, NUM_BUCKETS);
        CUDA_CHECK(cudaDeviceSynchronize());
        
        // Step 5: Scatter elements to correct positions
        scatter_by_digit<RADIX_BITS><<<num_blocks, BLOCK_SIZE>>>(current, next, n, d_digits,
                                                                   d_bucket_positions);
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
