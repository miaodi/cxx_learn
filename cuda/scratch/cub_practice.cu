#include <iostream>
#include <cub/cub.cuh>
#include <cuda_runtime.h>

// Helper macro for CUDA error checking
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA Error: " << cudaGetErrorString(err) << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// Example 1: Device-wide reduction
void example_device_reduce() {
    std::cout << "=== Device-wide Reduction ===" << std::endl;
    
    const int num_items = 10000;
    int* d_in = nullptr;
    int* d_out = nullptr;
    
    CUDA_CHECK(cudaMalloc(&d_in, num_items * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_out, sizeof(int)));
    
    // Initialize data on device
    int* h_in = new int[num_items];
    for (int i = 0; i < num_items; ++i) {
        h_in[i] = 1; // Each element is 1, so sum should be num_items
    }
    CUDA_CHECK(cudaMemcpy(d_in, h_in, num_items * sizeof(int), cudaMemcpyHostToDevice));
    
    // Determine temporary device storage requirements
    void* d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;
    cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, d_in, d_out, num_items);
    
    // Allocate temporary storage
    CUDA_CHECK(cudaMalloc(&d_temp_storage, temp_storage_bytes));
    
    // Run sum-reduction
    cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, d_in, d_out, num_items);
    
    // Copy result back to host
    int h_out = 0;
    CUDA_CHECK(cudaMemcpy(&h_out, d_out, sizeof(int), cudaMemcpyDeviceToHost));
    
    std::cout << "Sum of " << num_items << " ones = " << h_out << std::endl;
    
    // Cleanup
    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));
    CUDA_CHECK(cudaFree(d_temp_storage));
    delete[] h_in;
}

// Example 2: Device-wide sort
void example_device_sort() {
    std::cout << "\n=== Device-wide Sort ===" << std::endl;
    
    const int num_items = 100;
    int* d_keys_in = nullptr;
    int* d_keys_out = nullptr;
    
    CUDA_CHECK(cudaMalloc(&d_keys_in, num_items * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_keys_out, num_items * sizeof(int)));
    
    // Initialize with reverse sorted data
    int* h_keys = new int[num_items];
    for (int i = 0; i < num_items; ++i) {
        h_keys[i] = num_items - i;
    }
    CUDA_CHECK(cudaMemcpy(d_keys_in, h_keys, num_items * sizeof(int), cudaMemcpyHostToDevice));
    
    // Determine temporary device storage requirements
    void* d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;
    cub::DeviceRadixSort::SortKeys(d_temp_storage, temp_storage_bytes, d_keys_in, d_keys_out, num_items);
    
    // Allocate temporary storage
    CUDA_CHECK(cudaMalloc(&d_temp_storage, temp_storage_bytes));
    
    // Run sorting operation
    cub::DeviceRadixSort::SortKeys(d_temp_storage, temp_storage_bytes, d_keys_in, d_keys_out, num_items);
    
    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(h_keys, d_keys_out, num_items * sizeof(int), cudaMemcpyDeviceToHost));
    
    std::cout << "First 10 sorted elements: ";
    for (int i = 0; i < 10; ++i) {
        std::cout << h_keys[i] << " ";
    }
    std::cout << std::endl;
    
    // Cleanup
    CUDA_CHECK(cudaFree(d_keys_in));
    CUDA_CHECK(cudaFree(d_keys_out));
    CUDA_CHECK(cudaFree(d_temp_storage));
    delete[] h_keys;
}

// Example 2b: Segmented sort - sort independent segments within an array
void example_segmented_sort() {
    std::cout << "\n=== Segmented Sort ===" << std::endl;
    
    const int num_items = 10;
    const int num_segments = 3;
    
    int* d_keys_in = nullptr;
    int* d_keys_out = nullptr;
    int* d_offsets = nullptr;  // Segment boundaries
    
    CUDA_CHECK(cudaMalloc(&d_keys_in, num_items * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_keys_out, num_items * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_offsets, (num_segments + 1) * sizeof(int)));
    
    // Initialize with reverse sorted data
    int h_keys[10] = {9, 8, 7, 6, 5, 4, 3, 2, 1, 0};
    
    // Define segment boundaries: [0,2), [2,5), [5,10)
    int h_offsets[4] = {0, 2, 5, 10};
    
    CUDA_CHECK(cudaMemcpy(d_keys_in, h_keys, num_items * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_offsets, h_offsets, (num_segments + 1) * sizeof(int), cudaMemcpyHostToDevice));
    
    std::cout << "Input: ";
    for (int i = 0; i < num_items; ++i) {
        std::cout << h_keys[i] << " ";
    }
    std::cout << std::endl;
    
    // Determine temporary device storage requirements
    void* d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;
    cub::DeviceSegmentedRadixSort::SortKeys(
        d_temp_storage, temp_storage_bytes, 
        d_keys_in, d_keys_out, 
        num_items, num_segments, 
        d_offsets, d_offsets + 1);
    
    // Allocate temporary storage
    CUDA_CHECK(cudaMalloc(&d_temp_storage, temp_storage_bytes));
    
    // Run segmented sorting operation
    cub::DeviceSegmentedRadixSort::SortKeys(
        d_temp_storage, temp_storage_bytes,
        d_keys_in, d_keys_out,
        num_items, num_segments,
        d_offsets, d_offsets + 1);
    
    // Copy result back to host
    int h_keys_out[10];
    CUDA_CHECK(cudaMemcpy(h_keys_out, d_keys_out, num_items * sizeof(int), cudaMemcpyDeviceToHost));
    
    std::cout << "Output: ";
    for (int i = 0; i < num_items; ++i) {
        std::cout << h_keys_out[i] << " ";
        if (i == 1 || i == 4) std::cout << "| ";  // Visual separator
    }
    std::cout << std::endl;
    std::cout << "Segments: [0,2), [2,5), [5,10) sorted independently" << std::endl;
    
    // Cleanup
    CUDA_CHECK(cudaFree(d_keys_in));
    CUDA_CHECK(cudaFree(d_keys_out));
    CUDA_CHECK(cudaFree(d_offsets));
    CUDA_CHECK(cudaFree(d_temp_storage));
}

// Example 3: Block-level reduction kernel
__global__ void block_reduce_kernel(int* input, int* output, int num_items) {
    // Specialize BlockReduce for a 1D block of 256 threads of type int
    typedef cub::BlockReduce<int, 256> BlockReduce;
    
    // Allocate shared memory for BlockReduce
    __shared__ typename BlockReduce::TempStorage temp_storage;
    
    // Obtain a segment of consecutive items that are blocked across threads
    int thread_data = 0;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_items) {
        thread_data = input[idx];
    }
    
    // Compute the block-wide sum for thread0
    int aggregate = BlockReduce(temp_storage).Sum(thread_data);
    
    // Thread 0 writes the result
    if (threadIdx.x == 0) {
        output[blockIdx.x] = aggregate;
    }
}

void example_block_reduce() {
    std::cout << "\n=== Block-level Reduction ===" << std::endl;
    
    const int num_items = 1024;
    const int block_size = 256;
    const int num_blocks = (num_items + block_size - 1) / block_size;
    
    int* d_in = nullptr;
    int* d_out = nullptr;
    
    CUDA_CHECK(cudaMalloc(&d_in, num_items * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_out, num_blocks * sizeof(int)));
    
    // Initialize data
    int* h_in = new int[num_items];
    for (int i = 0; i < num_items; ++i) {
        h_in[i] = 1;
    }
    CUDA_CHECK(cudaMemcpy(d_in, h_in, num_items * sizeof(int), cudaMemcpyHostToDevice));
    
    // Launch kernel
    block_reduce_kernel<<<num_blocks, block_size>>>(d_in, d_out, num_items);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Copy results back
    int* h_out = new int[num_blocks];
    CUDA_CHECK(cudaMemcpy(h_out, d_out, num_blocks * sizeof(int), cudaMemcpyDeviceToHost));
    
    // Sum the partial sums
    int total = 0;
    for (int i = 0; i < num_blocks; ++i) {
        total += h_out[i];
    }
    
    std::cout << "Sum of " << num_items << " ones = " << total << std::endl;
    
    // Cleanup
    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));
    delete[] h_in;
    delete[] h_out;
}

// Example 4: Block-level scan (prefix sum)
__global__ void block_scan_kernel(int* input, int* output, int num_items) {
    // Specialize BlockScan for a 1D block of 256 threads of type int
    typedef cub::BlockScan<int, 256> BlockScan;
    
    // Allocate shared memory for BlockScan
    __shared__ typename BlockScan::TempStorage temp_storage;
    
    // Obtain input item for each thread
    int thread_data = 0;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_items) {
        thread_data = input[idx];
    }
    
    // Collectively compute the block-wide exclusive prefix sum
    int block_aggregate;
    BlockScan(temp_storage).ExclusiveSum(thread_data, thread_data, block_aggregate);
    
    // Write output
    if (idx < num_items) {
        output[idx] = thread_data;
    }
}

void example_block_scan() {
    std::cout << "\n=== Block-level Scan (Prefix Sum) ===" << std::endl;
    
    const int num_items = 256;
    const int block_size = 256;
    
    int* d_in = nullptr;
    int* d_out = nullptr;
    
    CUDA_CHECK(cudaMalloc(&d_in, num_items * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_out, num_items * sizeof(int)));
    
    // Initialize data with all 1s
    int* h_in = new int[num_items];
    for (int i = 0; i < num_items; ++i) {
        h_in[i] = 1;
    }
    CUDA_CHECK(cudaMemcpy(d_in, h_in, num_items * sizeof(int), cudaMemcpyHostToDevice));
    
    // Launch kernel (single block)
    block_scan_kernel<<<1, block_size>>>(d_in, d_out, num_items);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Copy results back
    int* h_out = new int[num_items];
    CUDA_CHECK(cudaMemcpy(h_out, d_out, num_items * sizeof(int), cudaMemcpyDeviceToHost));
    
    std::cout << "First 10 prefix sum results: ";
    for (int i = 0; i < 10; ++i) {
        std::cout << h_out[i] << " ";
    }
    std::cout << "\nLast 5 prefix sum results: ";
    for (int i = num_items - 5; i < num_items; ++i) {
        std::cout << h_out[i] << " ";
    }
    std::cout << std::endl;
    
    // Cleanup
    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));
    delete[] h_in;
    delete[] h_out;
}

// Example 5: Device-wide select (compaction)
void example_device_select() {
    std::cout << "\n=== Device-wide Select (Compaction) ===" << std::endl;
    
    const int num_items = 100;
    int* d_in = nullptr;
    int* d_out = nullptr;
    int* d_num_selected = nullptr;
    
    CUDA_CHECK(cudaMalloc(&d_in, num_items * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_out, num_items * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_num_selected, sizeof(int)));
    
    // Initialize data (even numbers only)
    int* h_in = new int[num_items];
    for (int i = 0; i < num_items; ++i) {
        h_in[i] = i;
    }
    CUDA_CHECK(cudaMemcpy(d_in, h_in, num_items * sizeof(int), cudaMemcpyHostToDevice));
    
    // Create a predicate for selecting even numbers
    auto select_op = [] __device__ (const int& a) { return (a % 2) == 0; };
    
    // Determine temporary device storage requirements
    void* d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;
    cub::DeviceSelect::If(d_temp_storage, temp_storage_bytes, d_in, d_out, d_num_selected, num_items, select_op);
    
    // Allocate temporary storage
    CUDA_CHECK(cudaMalloc(&d_temp_storage, temp_storage_bytes));
    
    // Run selection
    cub::DeviceSelect::If(d_temp_storage, temp_storage_bytes, d_in, d_out, d_num_selected, num_items, select_op);
    
    // Copy results back
    int h_num_selected = 0;
    CUDA_CHECK(cudaMemcpy(&h_num_selected, d_num_selected, sizeof(int), cudaMemcpyDeviceToHost));
    
    int* h_out = new int[h_num_selected];
    CUDA_CHECK(cudaMemcpy(h_out, d_out, h_num_selected * sizeof(int), cudaMemcpyDeviceToHost));
    
    std::cout << "Selected " << h_num_selected << " even numbers from " << num_items << " items" << std::endl;
    std::cout << "First 10 selected items: ";
    for (int i = 0; i < std::min(10, h_num_selected); ++i) {
        std::cout << h_out[i] << " ";
    }
    std::cout << std::endl;
    
    // Cleanup
    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));
    CUDA_CHECK(cudaFree(d_num_selected));
    CUDA_CHECK(cudaFree(d_temp_storage));
    delete[] h_in;
    delete[] h_out;
}

// Example 6: Warp-level reduction
__global__ void warp_reduce_kernel(int* input, int* output, int num_items) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int warp_id = idx / 32;
    
    // Allocate shared memory for WarpReduce
    typedef cub::WarpReduce<int> WarpReduce;
    __shared__ typename WarpReduce::TempStorage temp_storage[8]; // 8 warps per block (256/32)
    
    // Load data
    int thread_data = 0;
    if (idx < num_items) {
        thread_data = input[idx];
    }
    
    // Warp-level reduction
    int lane_id = threadIdx.x % 32;
    int warp_id_in_block = threadIdx.x / 32;
    int aggregate = WarpReduce(temp_storage[warp_id_in_block]).Sum(thread_data);
    
    // First thread in each warp writes the result
    if (lane_id == 0 && idx < num_items) {
        output[warp_id] = aggregate;
    }
}

void example_warp_reduce() {
    std::cout << "\n=== Warp-level Reduction ===" << std::endl;
    
    const int num_items = 256;
    const int block_size = 256;
    const int num_warps = (num_items + 31) / 32;
    
    int* d_in = nullptr;
    int* d_out = nullptr;
    
    CUDA_CHECK(cudaMalloc(&d_in, num_items * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_out, num_warps * sizeof(int)));
    
    // Initialize data
    int* h_in = new int[num_items];
    for (int i = 0; i < num_items; ++i) {
        h_in[i] = 1;
    }
    CUDA_CHECK(cudaMemcpy(d_in, h_in, num_items * sizeof(int), cudaMemcpyHostToDevice));
    
    // Launch kernel
    warp_reduce_kernel<<<1, block_size>>>(d_in, d_out, num_items);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Copy results back
    int* h_out = new int[num_warps];
    CUDA_CHECK(cudaMemcpy(h_out, d_out, num_warps * sizeof(int), cudaMemcpyDeviceToHost));
    
    std::cout << "Warp-level reduction results (per warp): ";
    for (int i = 0; i < num_warps; ++i) {
        std::cout << h_out[i] << " ";
    }
    std::cout << std::endl;
    
    // Cleanup
    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));
    delete[] h_in;
    delete[] h_out;
}

int main() {
    std::cout << "CUB (CUDA UnBound) Practice Examples\n" << std::endl;
    
    // Run all examples
    example_device_reduce();
    example_device_sort();
    example_segmented_sort();
    example_block_reduce();
    example_block_scan();
    example_device_select();
    example_warp_reduce();
    
    std::cout << "\nAll CUB examples completed successfully!" << std::endl;
    
    return 0;
}
