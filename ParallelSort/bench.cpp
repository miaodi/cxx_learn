#include "parallel_sort.h"
#include <benchmark/benchmark.h>
#include <vector>
#include <random>
#include <algorithm>
#include <omp.h>

// Generate random data for sorting
template<typename T>
std::vector<T> generate_random_data(size_t size, unsigned seed = 42) {
    std::vector<T> data(size);
    std::mt19937 gen(seed);
    
    if constexpr (std::is_integral_v<T>) {
        std::uniform_int_distribution<T> dist(std::numeric_limits<T>::min(), 
                                               std::numeric_limits<T>::max());
        for (auto& val : data) {
            val = dist(gen);
        }
    } else if constexpr (std::is_floating_point_v<T>) {
        std::uniform_real_distribution<T> dist(0.0, 1.0);
        for (auto& val : data) {
            val = dist(gen);
        }
    }
    
    return data;
}

// Benchmark std::sort
static void BM_StdSort_Int(benchmark::State& state) {
    const size_t size = state.range(0);
    auto original_data = generate_random_data<int>(size);
    
    for (auto _ : state) {
        state.PauseTiming();
        auto data = original_data; // Copy data
        state.ResumeTiming();
        
        std::sort(data.begin(), data.end());
        
        benchmark::DoNotOptimize(data.data());
        benchmark::ClobberMemory();
    }
    
    state.SetComplexityN(size);
    state.SetItemsProcessed(state.iterations() * size);
}

// Benchmark parallel sort with different thread counts
static void BM_ParallelSort_Int(benchmark::State& state) {
    const size_t size = state.range(0);
    const int nthreads = state.range(1);
    auto original_data = generate_random_data<int>(size);
    
    for (auto _ : state) {
        state.PauseTiming();
        auto data = original_data; // Copy data
        state.ResumeTiming();
        
        parallel::sort(data.begin(), data.end(), nthreads);
        
        benchmark::DoNotOptimize(data.data());
        benchmark::ClobberMemory();
    }
    
    state.SetComplexityN(size);
    state.SetItemsProcessed(state.iterations() * size);
    state.SetLabel("threads=" + std::to_string(nthreads));
}

// Benchmark std::sort for doubles
static void BM_StdSort_Double(benchmark::State& state) {
    const size_t size = state.range(0);
    auto original_data = generate_random_data<double>(size);
    
    for (auto _ : state) {
        state.PauseTiming();
        auto data = original_data;
        state.ResumeTiming();
        
        std::sort(data.begin(), data.end());
        
        benchmark::DoNotOptimize(data.data());
        benchmark::ClobberMemory();
    }
    
    state.SetComplexityN(size);
    state.SetItemsProcessed(state.iterations() * size);
}

// Benchmark parallel sort for doubles
static void BM_ParallelSort_Double(benchmark::State& state) {
    const size_t size = state.range(0);
    const int nthreads = state.range(1);
    auto original_data = generate_random_data<double>(size);
    
    for (auto _ : state) {
        state.PauseTiming();
        auto data = original_data;
        state.ResumeTiming();
        
        parallel::sort(data.begin(), data.end(), nthreads);
        
        benchmark::DoNotOptimize(data.data());
        benchmark::ClobberMemory();
    }
    
    state.SetComplexityN(size);
    state.SetItemsProcessed(state.iterations() * size);
    state.SetLabel("threads=" + std::to_string(nthreads));
}

// Benchmark with already sorted data (best case)
static void BM_StdSort_Sorted(benchmark::State& state) {
    const size_t size = state.range(0);
    std::vector<int> data(size);
    std::iota(data.begin(), data.end(), 0);
    
    for (auto _ : state) {
        auto copy = data;
        std::sort(copy.begin(), copy.end());
        benchmark::DoNotOptimize(copy.data());
    }
    
    state.SetComplexityN(size);
}

static void BM_ParallelSort_Sorted(benchmark::State& state) {
    const size_t size = state.range(0);
    const int nthreads = state.range(1);
    std::vector<int> data(size);
    std::iota(data.begin(), data.end(), 0);
    
    for (auto _ : state) {
        auto copy = data;
        parallel::sort(copy.begin(), copy.end(), nthreads);
        benchmark::DoNotOptimize(copy.data());
    }
    
    state.SetComplexityN(size);
    state.SetLabel("threads=" + std::to_string(nthreads));
}

// Benchmark with reverse sorted data (worst case for some algorithms)
static void BM_StdSort_Reversed(benchmark::State& state) {
    const size_t size = state.range(0);
    std::vector<int> data(size);
    std::iota(data.rbegin(), data.rend(), 0);
    
    for (auto _ : state) {
        auto copy = data;
        std::sort(copy.begin(), copy.end());
        benchmark::DoNotOptimize(copy.data());
    }
    
    state.SetComplexityN(size);
}

static void BM_ParallelSort_Reversed(benchmark::State& state) {
    const size_t size = state.range(0);
    const int nthreads = state.range(1);
    std::vector<int> data(size);
    std::iota(data.rbegin(), data.rend(), 0);
    
    for (auto _ : state) {
        auto copy = data;
        parallel::sort(copy.begin(), copy.end(), nthreads);
        benchmark::DoNotOptimize(copy.data());
    }
    
    state.SetComplexityN(size);
    state.SetLabel("threads=" + std::to_string(nthreads));
}

// Register benchmarks with various sizes
// Organized by size, then thread count: (size|threads)
const int max_threads = omp_get_max_threads();

// Small integer sorting benchmarks - 16 elements
BENCHMARK(BM_StdSort_Int)->Arg(16);
BENCHMARK(BM_ParallelSort_Int)->Args({16, 2});
BENCHMARK(BM_ParallelSort_Int)->Args({16, 4});
BENCHMARK(BM_ParallelSort_Int)->Args({16, 8});

// Small integer sorting benchmarks - 64 elements
BENCHMARK(BM_StdSort_Int)->Arg(64);
BENCHMARK(BM_ParallelSort_Int)->Args({64, 2});
BENCHMARK(BM_ParallelSort_Int)->Args({64, 4});
BENCHMARK(BM_ParallelSort_Int)->Args({64, 8});

// Small integer sorting benchmarks - 256 elements
BENCHMARK(BM_StdSort_Int)->Arg(256);
BENCHMARK(BM_ParallelSort_Int)->Args({256, 2});
BENCHMARK(BM_ParallelSort_Int)->Args({256, 4});
BENCHMARK(BM_ParallelSort_Int)->Args({256, 8});

// Small integer sorting benchmarks - 1024 elements
BENCHMARK(BM_StdSort_Int)->Arg(1024);
BENCHMARK(BM_ParallelSort_Int)->Args({1024, 2});
BENCHMARK(BM_ParallelSort_Int)->Args({1024, 4});
BENCHMARK(BM_ParallelSort_Int)->Args({1024, 8});

// Small integer sorting benchmarks - 2048 elements
BENCHMARK(BM_StdSort_Int)->Arg(2048);
BENCHMARK(BM_ParallelSort_Int)->Args({2048, 2});
BENCHMARK(BM_ParallelSort_Int)->Args({2048, 4});
BENCHMARK(BM_ParallelSort_Int)->Args({2048, 8});
// Small integer sorting benchmarks - 4096 elements
BENCHMARK(BM_StdSort_Int)->Arg(4096);
BENCHMARK(BM_ParallelSort_Int)->Args({4096, 2});
BENCHMARK(BM_ParallelSort_Int)->Args({4096, 4});
BENCHMARK(BM_ParallelSort_Int)->Args({4096, 8});

// Integer sorting benchmarks - 8K elements
BENCHMARK(BM_StdSort_Int)->Arg(1<<13);
BENCHMARK(BM_ParallelSort_Int)->Args({1<<13, 2});
BENCHMARK(BM_ParallelSort_Int)->Args({1<<13, 4});
BENCHMARK(BM_ParallelSort_Int)->Args({1<<13, 8});

// Integer sorting benchmarks - 64K elements
BENCHMARK(BM_StdSort_Int)->Arg(1<<16);
BENCHMARK(BM_ParallelSort_Int)->Args({1<<16, 2});
BENCHMARK(BM_ParallelSort_Int)->Args({1<<16, 4});
BENCHMARK(BM_ParallelSort_Int)->Args({1<<16, 8});

// Integer sorting benchmarks - 1M elements
BENCHMARK(BM_StdSort_Int)->Arg(1<<20);
BENCHMARK(BM_ParallelSort_Int)->Args({1<<20, 2});
BENCHMARK(BM_ParallelSort_Int)->Args({1<<20, 4});
BENCHMARK(BM_ParallelSort_Int)->Args({1<<20, 8});

// Integer sorting benchmarks - 8M elements
BENCHMARK(BM_StdSort_Int)->Arg(1<<23);
BENCHMARK(BM_ParallelSort_Int)->Args({1<<23, 2});
BENCHMARK(BM_ParallelSort_Int)->Args({1<<23, 4});
BENCHMARK(BM_ParallelSort_Int)->Args({1<<23, 8});

// Double sorting benchmarks - 64K elements
BENCHMARK(BM_StdSort_Double)->Arg(1<<16);
BENCHMARK(BM_ParallelSort_Double)->Args({1<<16, 2});
BENCHMARK(BM_ParallelSort_Double)->Args({1<<16, 4});
BENCHMARK(BM_ParallelSort_Double)->Args({1<<16, 8});

// Double sorting benchmarks - 1M elements
BENCHMARK(BM_StdSort_Double)->Arg(1<<20);
BENCHMARK(BM_ParallelSort_Double)->Args({1<<20, 2});
BENCHMARK(BM_ParallelSort_Double)->Args({1<<20, 4});
BENCHMARK(BM_ParallelSort_Double)->Args({1<<20, 8});

// Already sorted data - 1M elements
BENCHMARK(BM_StdSort_Sorted)->Arg(1<<20);
BENCHMARK(BM_ParallelSort_Sorted)->Args({1<<20, 2});
BENCHMARK(BM_ParallelSort_Sorted)->Args({1<<20, 4});
BENCHMARK(BM_ParallelSort_Sorted)->Args({1<<20, 8});

// Reverse sorted data - 1M elements
BENCHMARK(BM_StdSort_Reversed)->Arg(1<<20);
BENCHMARK(BM_ParallelSort_Reversed)->Args({1<<20, 2});
BENCHMARK(BM_ParallelSort_Reversed)->Args({1<<20, 4});
BENCHMARK(BM_ParallelSort_Reversed)->Args({1<<20, 8});

BENCHMARK_MAIN();
