#include <benchmark/benchmark.h>
#include <vector>
#include <random>
#include <algorithm>
#include <cstdlib>

// Cache sizes (typical x86-64):
// L1d: 32KB, L2: 256KB, L3: 8MB
constexpr size_t L1_SIZE = 32 * 1024;      // 32KB
constexpr size_t L2_SIZE = 256 * 1024;     // 256KB  
constexpr size_t L3_SIZE = 8 * 1024 * 1024; // 8MB
constexpr size_t RAM_SIZE = 20 * 1024 * 1024; // 20MB

// =============================================================================
// 1. MEMORY LATENCY BENCHMARK - Pointer Chasing
// This is the gold standard for measuring pure memory access latency
// =============================================================================

static void BM_MemoryLatency(benchmark::State& state) {
    size_t size = state.range(0);
    size_t num_elements = size / sizeof(size_t);
    
    // Allocate memory and create random pointer chain
    std::vector<size_t> data(num_elements);
    
    // Initialize indices sequentially first
    for (size_t i = 0; i < num_elements - 1; ++i) {
        data[i] = i + 1;
    }
    data[num_elements - 1] = 0; // Loop back to start
    
    // Randomize the chain to prevent predictive prefetching
    std::random_device rd;
    std::mt19937 gen(rd());
    std::shuffle(data.begin(), data.end() - 1, gen);
    
    // Ensure the chain is still valid by rebuilding it
    std::vector<size_t> indices(num_elements);
    std::iota(indices.begin(), indices.end(), 0);
    std::shuffle(indices.begin(), indices.end(), gen);
    
    for (size_t i = 0; i < num_elements - 1; ++i) {
        data[indices[i]] = indices[i + 1];
    }
    data[indices[num_elements - 1]] = indices[0];
    
    // Benchmark: Follow the pointer chain
    size_t index = 0;
    for (auto _ : state) {
        // Follow pointers for a fixed number of steps
        for (int steps = 0; steps < 1000; ++steps) {
            index = data[index];
            benchmark::DoNotOptimize(index);
        }
    }
    
    // Report memory size being tested
    state.SetLabel(std::to_string(size / 1024) + "KB");
}

// Test different memory sizes to hit different cache levels
BENCHMARK(BM_MemoryLatency)
    ->Arg(4 * 1024)        // 4KB - fits in L1
    ->Arg(16 * 1024)       // 16KB - still L1
    ->Arg(128 * 1024)      // 128KB - fits in L2
    ->Arg(1024 * 1024)     // 1MB - fits in L3
    ->Arg(16 * 1024 * 1024) // 16MB - exceeds L3, hits RAM
    ->Arg(64 * 1024 * 1024) // 64MB - definitely RAM
    ->Unit(benchmark::kNanosecond);

// =============================================================================
// 2. SEQUENTIAL ACCESS BENCHMARK - Shows cache line effects
// =============================================================================

static void BM_SequentialAccess(benchmark::State& state) {
    size_t size = state.range(0);
    std::vector<int> data(size / sizeof(int), 1);
    
    for (auto _ : state) {
        long sum = 0;
        for (size_t i = 0; i < data.size(); ++i) {
            sum += data[i];
        }
        benchmark::DoNotOptimize(sum);
    }
    
    state.SetBytesProcessed(state.iterations() * size);
    state.SetLabel(std::to_string(size / 1024) + "KB");
}

BENCHMARK(BM_SequentialAccess)
    ->Arg(4 * 1024)        // L1
    ->Arg(32 * 1024)       // L1 boundary
    ->Arg(256 * 1024)      // L2
    ->Arg(2 * 1024 * 1024) // L3
    ->Arg(32 * 1024 * 1024); // RAM

// =============================================================================
// 3. RANDOM ACCESS BENCHMARK - Shows cache miss penalty
// =============================================================================

static void BM_RandomAccess(benchmark::State& state) {
    size_t size = state.range(0);
    size_t num_elements = size / sizeof(int);
    std::vector<int> data(num_elements, 1);
    
    // Generate random indices
    std::vector<size_t> indices(num_elements);
    std::iota(indices.begin(), indices.end(), 0);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::shuffle(indices.begin(), indices.end(), gen);
    
    for (auto _ : state) {
        long sum = 0;
        for (size_t i = 0; i < indices.size(); ++i) {
            sum += data[indices[i]];
        }
        benchmark::DoNotOptimize(sum);
    }
    
    state.SetBytesProcessed(state.iterations() * size);
    state.SetLabel(std::to_string(size / 1024) + "KB");
}

BENCHMARK(BM_RandomAccess)
    ->Arg(4 * 1024)        // L1
    ->Arg(32 * 1024)       // L1 boundary  
    ->Arg(256 * 1024)      // L2
    ->Arg(2 * 1024 * 1024) // L3
    ->Arg(32 * 1024 * 1024); // RAM

// =============================================================================
// 4. STRIDE ACCESS BENCHMARK - Shows cache line size effects
// =============================================================================

template<int STRIDE>
static void BM_StrideAccess(benchmark::State& state) {
    size_t size = state.range(0);
    size_t num_elements = size / sizeof(int);
    std::vector<int> data(num_elements, 1);
    
    for (auto _ : state) {
        long sum = 0;
        for (size_t i = 0; i < num_elements; i += STRIDE) {
            sum += data[i];
        }
        benchmark::DoNotOptimize(sum);
    }
    
    state.SetBytesProcessed(state.iterations() * (num_elements / STRIDE) * sizeof(int));
    state.SetLabel("stride=" + std::to_string(STRIDE) + ", " + std::to_string(size / 1024) + "KB");
}

// Test different strides - cache lines are typically 64 bytes (16 ints)
BENCHMARK_TEMPLATE(BM_StrideAccess, 1)->Arg(1024 * 1024);   // Sequential
BENCHMARK_TEMPLATE(BM_StrideAccess, 2)->Arg(1024 * 1024);   // Every other
BENCHMARK_TEMPLATE(BM_StrideAccess, 4)->Arg(1024 * 1024);   // Every 4th
BENCHMARK_TEMPLATE(BM_StrideAccess, 8)->Arg(1024 * 1024);   // Every 8th
BENCHMARK_TEMPLATE(BM_StrideAccess, 16)->Arg(1024 * 1024);  // Every cache line
BENCHMARK_TEMPLATE(BM_StrideAccess, 32)->Arg(1024 * 1024);  // Every other cache line
BENCHMARK_TEMPLATE(BM_StrideAccess, 64)->Arg(1024 * 1024);  // Every 4th cache line

// =============================================================================
// 5. MATRIX TRAVERSAL - Cache-friendly vs Cache-unfriendly
// =============================================================================

static void BM_MatrixRowMajor(benchmark::State& state) {
    size_t n = state.range(0);
    std::vector<std::vector<int>> matrix(n, std::vector<int>(n, 1));
    
    for (auto _ : state) {
        long sum = 0;
        // Cache-friendly: row-major access
        for (size_t i = 0; i < n; ++i) {
            for (size_t j = 0; j < n; ++j) {
                sum += matrix[i][j];
            }
        }
        benchmark::DoNotOptimize(sum);
    }
    
    state.SetBytesProcessed(state.iterations() * n * n * sizeof(int));
    state.SetLabel(std::to_string(n) + "x" + std::to_string(n));
}

static void BM_MatrixColumnMajor(benchmark::State& state) {
    size_t n = state.range(0);
    std::vector<std::vector<int>> matrix(n, std::vector<int>(n, 1));
    
    for (auto _ : state) {
        long sum = 0;
        // Cache-unfriendly: column-major access
        for (size_t j = 0; j < n; ++j) {
            for (size_t i = 0; i < n; ++i) {
                sum += matrix[i][j];
            }
        }
        benchmark::DoNotOptimize(sum);
    }
    
    state.SetBytesProcessed(state.iterations() * n * n * sizeof(int));
    state.SetLabel(std::to_string(n) + "x" + std::to_string(n));
}

// Test matrices that exceed different cache levels
BENCHMARK(BM_MatrixRowMajor)->RangeMultiplier(2)->Range(64, 1024);
BENCHMARK(BM_MatrixColumnMajor)->RangeMultiplier(2)->Range(64, 1024);

// =============================================================================
// 6. PREFETCH DEMONSTRATION
// =============================================================================

static void BM_NoPrefetch(benchmark::State& state) {
    size_t size = state.range(0);
    std::vector<int> data(size, 1);
    
    for (auto _ : state) {
        long sum = 0;
        for (size_t i = 0; i < size; i += 16) { // Jump by cache line
            sum += data[i];
        }
        benchmark::DoNotOptimize(sum);
    }
}

static void BM_WithPrefetch(benchmark::State& state) {
    size_t size = state.range(0);
    std::vector<int> data(size, 1);
    
    for (auto _ : state) {
        long sum = 0;
        for (size_t i = 0; i < size; i += 16) { // Jump by cache line
            // Prefetch next cache line
            if (i + 32 < size) {
                __builtin_prefetch(&data[i + 32], 0, 3);
            }
            sum += data[i];
        }
        benchmark::DoNotOptimize(sum);
    }
}

BENCHMARK(BM_NoPrefetch)->Arg(16 * 1024 * 1024);
BENCHMARK(BM_WithPrefetch)->Arg(16 * 1024 * 1024);

BENCHMARK_MAIN();