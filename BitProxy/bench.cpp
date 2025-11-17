#include <benchmark/benchmark.h>
#include <vector>
#include <random>
#include <algorithm>

// Benchmark random reads from vector<bool>
static void BM_VectorBool_RandomRead(benchmark::State& state) {
    const size_t size = state.range(0);
    std::vector<bool> vec(size);
    
    // Initialize with random values
    std::mt19937 gen(42);
    std::uniform_int_distribution<int> dist(0, 1);
    for (size_t i = 0; i < size; ++i) {
        vec[i] = dist(gen);
    }
    
    // Generate random indices
    std::vector<size_t> indices(size);
    for (size_t i = 0; i < size; ++i) {
        indices[i] = i;
    }
    std::shuffle(indices.begin(), indices.end(), gen);
    
    size_t idx = 0;
    bool result = false;
    for (auto _ : state) {
        result ^= vec[indices[idx % size]];
        idx++;
        benchmark::DoNotOptimize(result);
    }
    
    state.SetItemsProcessed(state.iterations());
    state.SetBytesProcessed(state.iterations() * sizeof(bool));
}

// Benchmark random reads from vector<char>
static void BM_VectorChar_RandomRead(benchmark::State& state) {
    const size_t size = state.range(0);
    std::vector<char> vec(size);
    
    // Initialize with random values
    std::mt19937 gen(42);
    std::uniform_int_distribution<int> dist(0, 1);
    for (size_t i = 0; i < size; ++i) {
        vec[i] = dist(gen);
    }
    
    // Generate random indices
    std::vector<size_t> indices(size);
    for (size_t i = 0; i < size; ++i) {
        indices[i] = i;
    }
    std::shuffle(indices.begin(), indices.end(), gen);
    
    size_t idx = 0;
    char result = 0;
    for (auto _ : state) {
        result ^= vec[indices[idx % size]];
        idx++;
        benchmark::DoNotOptimize(result);
    }
    
    state.SetItemsProcessed(state.iterations());
    state.SetBytesProcessed(state.iterations() * sizeof(char));
}

// Benchmark random writes to vector<bool>
static void BM_VectorBool_RandomWrite(benchmark::State& state) {
    const size_t size = state.range(0);
    std::vector<bool> vec(size);
    
    // Generate random indices
    std::mt19937 gen(42);
    std::vector<size_t> indices(size);
    for (size_t i = 0; i < size; ++i) {
        indices[i] = i;
    }
    std::shuffle(indices.begin(), indices.end(), gen);
    
    // Generate random values
    std::uniform_int_distribution<int> dist(0, 1);
    std::vector<bool> values(size);
    for (size_t i = 0; i < size; ++i) {
        values[i] = dist(gen);
    }
    
    size_t idx = 0;
    for (auto _ : state) {
        vec[indices[idx % size]] = values[idx % size];
        idx++;
        benchmark::ClobberMemory();
    }
    
    state.SetItemsProcessed(state.iterations());
    state.SetBytesProcessed(state.iterations() * sizeof(bool));
}

// Benchmark random writes to vector<char>
static void BM_VectorChar_RandomWrite(benchmark::State& state) {
    const size_t size = state.range(0);
    std::vector<char> vec(size);
    
    // Generate random indices
    std::mt19937 gen(42);
    std::vector<size_t> indices(size);
    for (size_t i = 0; i < size; ++i) {
        indices[i] = i;
    }
    std::shuffle(indices.begin(), indices.end(), gen);
    
    // Generate random values
    std::uniform_int_distribution<int> dist(0, 1);
    std::vector<char> values(size);
    for (size_t i = 0; i < size; ++i) {
        values[i] = dist(gen);
    }
    
    size_t idx = 0;
    for (auto _ : state) {
        vec[indices[idx % size]] = values[idx % size];
        idx++;
        benchmark::DoNotOptimize(vec.data());
    }
    
    state.SetItemsProcessed(state.iterations());
    state.SetBytesProcessed(state.iterations() * sizeof(char));
}

// Benchmark random read-modify-write for vector<bool>
static void BM_VectorBool_RandomReadModifyWrite(benchmark::State& state) {
    const size_t size = state.range(0);
    std::vector<bool> vec(size);
    
    // Initialize with random values
    std::mt19937 gen(42);
    std::uniform_int_distribution<int> dist(0, 1);
    for (size_t i = 0; i < size; ++i) {
        vec[i] = dist(gen);
    }
    
    // Generate random indices
    std::vector<size_t> indices(size);
    for (size_t i = 0; i < size; ++i) {
        indices[i] = i;
    }
    std::shuffle(indices.begin(), indices.end(), gen);
    
    size_t idx = 0;
    for (auto _ : state) {
        vec[indices[idx % size]] = !vec[indices[idx % size]];
        idx++;
        benchmark::ClobberMemory();
    }
    
    state.SetItemsProcessed(state.iterations());
    state.SetBytesProcessed(state.iterations() * sizeof(bool));
}

// Benchmark random read-modify-write for vector<char>
static void BM_VectorChar_RandomReadModifyWrite(benchmark::State& state) {
    const size_t size = state.range(0);
    std::vector<char> vec(size);
    
    // Initialize with random values
    std::mt19937 gen(42);
    std::uniform_int_distribution<int> dist(0, 1);
    for (size_t i = 0; i < size; ++i) {
        vec[i] = dist(gen);
    }
    
    // Generate random indices
    std::vector<size_t> indices(size);
    for (size_t i = 0; i < size; ++i) {
        indices[i] = i;
    }
    std::shuffle(indices.begin(), indices.end(), gen);
    
    size_t idx = 0;
    for (auto _ : state) {
        vec[indices[idx % size]] = !vec[indices[idx % size]];
        idx++;
        benchmark::DoNotOptimize(vec.data());
    }
    
    state.SetItemsProcessed(state.iterations());
    state.SetBytesProcessed(state.iterations() * sizeof(char));
}

// Register benchmarks with various sizes
BENCHMARK(BM_VectorBool_RandomRead)->Range(1<<10, 1<<20);
BENCHMARK(BM_VectorChar_RandomRead)->Range(1<<10, 1<<20);

BENCHMARK(BM_VectorBool_RandomWrite)->Range(1<<10, 1<<20);
BENCHMARK(BM_VectorChar_RandomWrite)->Range(1<<10, 1<<20);

BENCHMARK(BM_VectorBool_RandomReadModifyWrite)->Range(1<<10, 1<<20);
BENCHMARK(BM_VectorChar_RandomReadModifyWrite)->Range(1<<10, 1<<20);

BENCHMARK_MAIN();
