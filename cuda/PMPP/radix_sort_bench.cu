#include "radix_sort.cuh"

#include <benchmark/benchmark.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <random>
#include <vector>
#include <thrust/sort.h>
#include <thrust/device_vector.h>

using namespace pmpp::radix_sort;

// Benchmark CUDA radix sort with 1-bit radix
static void BM_RadixSort_1bit(benchmark::State& state) {
    const size_t n = state.range(0);
    
    // Generate random data
    std::mt19937 rng(42);
    std::uniform_int_distribution<uint32_t> dist(0, UINT32_MAX);
    std::vector<uint32_t> h_input(n);
    for (size_t i = 0; i < n; i++) {
        h_input[i] = dist(rng);
    }
    
    // Allocate device memory
    uint32_t *d_input, *d_output;
    cudaMalloc(&d_input, n * sizeof(uint32_t));
    cudaMalloc(&d_output, n * sizeof(uint32_t));
    cudaMemcpy(d_input, h_input.data(), n * sizeof(uint32_t), cudaMemcpyHostToDevice);
    
    for (auto _ : state) {
        radix_sort<1>(d_input, d_output, n);
        cudaDeviceSynchronize();
    }
    
    cudaFree(d_input);
    cudaFree(d_output);
    
    state.SetItemsProcessed(state.iterations() * n);
    state.SetBytesProcessed(state.iterations() * n * sizeof(uint32_t));
}

// Benchmark CUDA radix sort with 2-bit radix
static void BM_RadixSort_2bit(benchmark::State& state) {
    const size_t n = state.range(0);
    
    // Generate random data
    std::mt19937 rng(42);
    std::uniform_int_distribution<uint32_t> dist(0, UINT32_MAX);
    std::vector<uint32_t> h_input(n);
    for (size_t i = 0; i < n; i++) {
        h_input[i] = dist(rng);
    }
    
    // Allocate device memory
    uint32_t *d_input, *d_output;
    cudaMalloc(&d_input, n * sizeof(uint32_t));
    cudaMalloc(&d_output, n * sizeof(uint32_t));
    cudaMemcpy(d_input, h_input.data(), n * sizeof(uint32_t), cudaMemcpyHostToDevice);
    
    for (auto _ : state) {
        radix_sort<2>(d_input, d_output, n);
        cudaDeviceSynchronize();
    }
    
    cudaFree(d_input);
    cudaFree(d_output);
    
    state.SetItemsProcessed(state.iterations() * n);
    state.SetBytesProcessed(state.iterations() * n * sizeof(uint32_t));
}

// Benchmark CUDA radix sort with 4-bit radix
static void BM_RadixSort_4bit(benchmark::State& state) {
    const size_t n = state.range(0);
    
    // Generate random data
    std::mt19937 rng(42);
    std::uniform_int_distribution<uint32_t> dist(0, UINT32_MAX);
    std::vector<uint32_t> h_input(n);
    for (size_t i = 0; i < n; i++) {
        h_input[i] = dist(rng);
    }
    
    // Allocate device memory
    uint32_t *d_input, *d_output;
    cudaMalloc(&d_input, n * sizeof(uint32_t));
    cudaMalloc(&d_output, n * sizeof(uint32_t));
    cudaMemcpy(d_input, h_input.data(), n * sizeof(uint32_t), cudaMemcpyHostToDevice);
    
    for (auto _ : state) {
        radix_sort<4>(d_input, d_output, n);
        cudaDeviceSynchronize();
    }
    
    cudaFree(d_input);
    cudaFree(d_output);
    
    state.SetItemsProcessed(state.iterations() * n);
    state.SetBytesProcessed(state.iterations() * n * sizeof(uint32_t));
}

// Benchmark CUDA radix sort with 8-bit radix
static void BM_RadixSort_8bit(benchmark::State& state) {
    const size_t n = state.range(0);
    
    // Generate random data
    std::mt19937 rng(42);
    std::uniform_int_distribution<uint32_t> dist(0, UINT32_MAX);
    std::vector<uint32_t> h_input(n);
    for (size_t i = 0; i < n; i++) {
        h_input[i] = dist(rng);
    }
    
    // Allocate device memory
    uint32_t *d_input, *d_output;
    cudaMalloc(&d_input, n * sizeof(uint32_t));
    cudaMalloc(&d_output, n * sizeof(uint32_t));
    cudaMemcpy(d_input, h_input.data(), n * sizeof(uint32_t), cudaMemcpyHostToDevice);
    
    for (auto _ : state) {
        radix_sort<8>(d_input, d_output, n);
        cudaDeviceSynchronize();
    }
    
    cudaFree(d_input);
    cudaFree(d_output);
    
    state.SetItemsProcessed(state.iterations() * n);
    state.SetBytesProcessed(state.iterations() * n * sizeof(uint32_t));
}

// Benchmark in-place version
static void BM_RadixSort_Inplace(benchmark::State& state) {
    const size_t n = state.range(0);
    
    std::mt19937 rng(42);
    std::uniform_int_distribution<uint32_t> dist(0, UINT32_MAX);
    std::vector<uint32_t> h_input(n);
    for (size_t i = 0; i < n; i++) {
        h_input[i] = dist(rng);
    }
    
    uint32_t *d_data;
    cudaMalloc(&d_data, n * sizeof(uint32_t));
    
    for (auto _ : state) {
        state.PauseTiming();
        cudaMemcpy(d_data, h_input.data(), n * sizeof(uint32_t), cudaMemcpyHostToDevice);
        state.ResumeTiming();
        
        radix_sort_inplace<4>(d_data, n);
        cudaDeviceSynchronize();
    }
    
    cudaFree(d_data);
    
    state.SetItemsProcessed(state.iterations() * n);
    state.SetBytesProcessed(state.iterations() * n * sizeof(uint32_t));
}

// Benchmark host wrapper (includes transfer overhead)
static void BM_RadixSort_Host(benchmark::State& state) {
    const size_t n = state.range(0);
    
    std::mt19937 rng(42);
    std::uniform_int_distribution<uint32_t> dist(0, UINT32_MAX);
    std::vector<uint32_t> h_input(n);
    std::vector<uint32_t> h_output(n);
    
    for (size_t i = 0; i < n; i++) {
        h_input[i] = dist(rng);
    }
    
    for (auto _ : state) {
        radix_sort_host<4>(h_input.data(), h_output.data(), n);
    }
    
    state.SetItemsProcessed(state.iterations() * n);
    state.SetBytesProcessed(state.iterations() * n * sizeof(uint32_t));
}

// Benchmark Thrust sort for comparison
static void BM_ThrustSort(benchmark::State& state) {
    const size_t n = state.range(0);
    
    std::mt19937 rng(42);
    std::uniform_int_distribution<uint32_t> dist(0, UINT32_MAX);
    std::vector<uint32_t> h_input(n);
    for (size_t i = 0; i < n; i++) {
        h_input[i] = dist(rng);
    }
    
    thrust::device_vector<uint32_t> d_data(h_input.begin(), h_input.end());
    
    for (auto _ : state) {
        state.PauseTiming();
        thrust::copy(h_input.begin(), h_input.end(), d_data.begin());
        state.ResumeTiming();
        
        thrust::sort(d_data.begin(), d_data.end());
        cudaDeviceSynchronize();
    }
    
    state.SetItemsProcessed(state.iterations() * n);
    state.SetBytesProcessed(state.iterations() * n * sizeof(uint32_t));
}

// Benchmark std::sort for CPU baseline
static void BM_StdSort(benchmark::State& state) {
    const size_t n = state.range(0);
    
    std::mt19937 rng(42);
    std::uniform_int_distribution<uint32_t> dist(0, UINT32_MAX);
    std::vector<uint32_t> input(n);
    for (size_t i = 0; i < n; i++) {
        input[i] = dist(rng);
    }
    
    std::vector<uint32_t> data = input;
    
    for (auto _ : state) {
        state.PauseTiming();
        data = input;
        state.ResumeTiming();
        
        std::sort(data.begin(), data.end());
    }
    
    state.SetItemsProcessed(state.iterations() * n);
    state.SetBytesProcessed(state.iterations() * n * sizeof(uint32_t));
}

// Register benchmarks with different sizes
BENCHMARK(BM_RadixSort_1bit)->Range(1<<10, 1<<28)->Unit(benchmark::kMillisecond);
BENCHMARK(BM_RadixSort_2bit)->Range(1<<10, 1<<28)->Unit(benchmark::kMillisecond);
BENCHMARK(BM_RadixSort_4bit)->Range(1<<10, 1<<28)->Unit(benchmark::kMillisecond);
BENCHMARK(BM_RadixSort_8bit)->Range(1<<10, 1<<28)->Unit(benchmark::kMillisecond);
BENCHMARK(BM_RadixSort_Inplace)->Range(1<<10, 1<<28)->Unit(benchmark::kMillisecond);
BENCHMARK(BM_RadixSort_Host)->Range(1<<10, 1<<28)->Unit(benchmark::kMillisecond);
BENCHMARK(BM_ThrustSort)->Range(1<<10, 1<<28)->Unit(benchmark::kMillisecond);
BENCHMARK(BM_StdSort)->Range(1<<10, 1<<28)->Unit(benchmark::kMillisecond);

BENCHMARK_MAIN();
