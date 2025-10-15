#include <benchmark/benchmark.h>
#include <vector>
#include <random>
#include "vector_add.h"
#include "gemm.h"

class VectorAddBenchmark : public benchmark::Fixture {
public:
    void SetUp(const ::benchmark::State& state) override {
        size_t N = state.range(0);
        A.resize(N);
        B.resize(N);
        C.resize(N);
        
        // Initialize with random data
        std::mt19937 rng(42);
        std::uniform_real_distribution<float> dist(-1000.0f, 1000.0f);
        
        for (size_t i = 0; i < N; ++i) {
            A[i] = dist(rng);
            B[i] = dist(rng);
        }
    }

    void TearDown(const ::benchmark::State& state) override {
        // Cleanup if needed
    }

protected:
    std::vector<float> A, B, C;
};

BENCHMARK_DEFINE_F(VectorAddBenchmark, CPU)(benchmark::State& state) {
    size_t N = state.range(0);
    
    for (auto _ : state) {
        cpu_vector_add(A.data(), B.data(), C.data(), N);
        benchmark::DoNotOptimize(C.data());
    }
    
    // Calculate throughput
    state.SetBytesProcessed(static_cast<int64_t>(state.iterations()) * N * 3 * sizeof(float));
    state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) * N);
}

BENCHMARK_DEFINE_F(VectorAddBenchmark, GPU)(benchmark::State& state) {
    size_t N = state.range(0);
    
    for (auto _ : state) {
        gpu_vector_add(A.data(), B.data(), C.data(), N);
        benchmark::DoNotOptimize(C.data());
    }
    
    // Calculate throughput
    state.SetBytesProcessed(static_cast<int64_t>(state.iterations()) * N * 3 * sizeof(float));
    state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) * N);
}

// Register benchmarks for different vector sizes
BENCHMARK_REGISTER_F(VectorAddBenchmark, CPU)
    ->Range(1024, 1024*1024*16)  // 1K to 16M elements
    ->Unit(benchmark::kMicrosecond)
    ->UseRealTime();

BENCHMARK_REGISTER_F(VectorAddBenchmark, GPU)
    ->Range(1024, 1024*1024*16)  // 1K to 16M elements
    ->Unit(benchmark::kMicrosecond)
    ->UseRealTime();

// Memory bandwidth focused benchmarks
static void BM_CPU_VectorAdd_Bandwidth(benchmark::State& state) {
    size_t N = state.range(0);
    std::vector<float> A(N), B(N), C(N);
    
    // Initialize data
    std::fill(A.begin(), A.end(), 1.0f);
    std::fill(B.begin(), B.end(), 2.0f);
    
    for (auto _ : state) {
        cpu_vector_add(A.data(), B.data(), C.data(), N);
        benchmark::DoNotOptimize(C.data());
    }
    
    double bytes_per_iteration = N * 3 * sizeof(float);  // Read A, B; Write C
    state.SetBytesProcessed(static_cast<int64_t>(state.iterations() * bytes_per_iteration));
}

static void BM_GPU_VectorAdd_Bandwidth(benchmark::State& state) {
    size_t N = state.range(0);
    std::vector<float> A(N), B(N), C(N);
    
    // Initialize data
    std::fill(A.begin(), A.end(), 1.0f);
    std::fill(B.begin(), B.end(), 2.0f);
    
    for (auto _ : state) {
        gpu_vector_add(A.data(), B.data(), C.data(), N);
        benchmark::DoNotOptimize(C.data());
    }
    
    double bytes_per_iteration = N * 3 * sizeof(float);  // Read A, B; Write C
    state.SetBytesProcessed(static_cast<int64_t>(state.iterations() * bytes_per_iteration));
}

BENCHMARK(BM_CPU_VectorAdd_Bandwidth)
    ->Range(1024*1024, 1024*1024*64)  // 1M to 64M elements
    ->Unit(benchmark::kMillisecond)
    ->UseRealTime();

BENCHMARK(BM_GPU_VectorAdd_Bandwidth)
    ->Range(1024*1024, 1024*1024*64)  // 1M to 64M elements  
    ->Unit(benchmark::kMillisecond)
    ->UseRealTime();

// Cache-focused benchmarks with smaller sizes
static void BM_CPU_VectorAdd_Cache(benchmark::State& state) {
    size_t N = state.range(0);
    std::vector<float> A(N, 1.0f), B(N, 2.0f), C(N);
    
    for (auto _ : state) {
        cpu_vector_add(A.data(), B.data(), C.data(), N);
        benchmark::DoNotOptimize(C.data());
        benchmark::ClobberMemory();
    }
    
    state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) * N);
}

static void BM_GPU_VectorAdd_Cache(benchmark::State& state) {
    size_t N = state.range(0);
    std::vector<float> A(N, 1.0f), B(N, 2.0f), C(N);
    
    for (auto _ : state) {
        gpu_vector_add(A.data(), B.data(), C.data(), N);
        benchmark::DoNotOptimize(C.data());
        benchmark::ClobberMemory();
    }
    
    state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) * N);
}

BENCHMARK(BM_CPU_VectorAdd_Cache)
    ->Range(8, 8192)  // 8 to 8K elements (fits in various cache levels)
    ->Unit(benchmark::kNanosecond);

BENCHMARK(BM_GPU_VectorAdd_Cache)
    ->Range(8, 8192)  // 8 to 8K elements
    ->Unit(benchmark::kNanosecond);

// GEMM Benchmarks
class GEMMBenchmark : public benchmark::Fixture {
public:
    void SetUp(const ::benchmark::State& state) override {
        // For square matrices, all dimensions are the same
        if (state.range_size() == 1) {
            M = N = K = state.range(0);
        } else {
            // For non-square matrices, use separate dimensions
            M = state.range(0);
            N = state.range(1);
            K = state.range(2);
        }
        
        A.resize(M * K);
        B.resize(K * N);
        C.resize(M * N);
        
        // Initialize with random data
        std::mt19937 rng(42);
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
        
        for (size_t i = 0; i < A.size(); ++i) {
            A[i] = dist(rng);
        }
        for (size_t i = 0; i < B.size(); ++i) {
            B[i] = dist(rng);
        }
    }

    void TearDown(const ::benchmark::State& state) override {
        // Cleanup if needed
    }

protected:
    std::vector<float> A, B, C;
    int M, N, K;
};

BENCHMARK_DEFINE_F(GEMMBenchmark, GPU_Square)(benchmark::State& state) {
    for (auto _ : state) {
        gpu_gemm(A.data(), B.data(), C.data(), M, N, K);
        benchmark::DoNotOptimize(C.data());
    }
    
    // Calculate FLOPS (2*M*N*K for GEMM)
    int64_t flops_per_iteration = static_cast<int64_t>(2) * M * N * K;
    state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) * flops_per_iteration);
    
    // Calculate bytes processed (read A, B; write C)
    int64_t bytes_per_iteration = static_cast<int64_t>(M * K + K * N + M * N) * sizeof(float);
    state.SetBytesProcessed(static_cast<int64_t>(state.iterations()) * bytes_per_iteration);
}

// Register square matrix benchmarks
BENCHMARK_REGISTER_F(GEMMBenchmark, GPU_Square)
    ->Range(16, 512)  // 16x16 to 512x512 matrices
    ->Unit(benchmark::kMillisecond)
    ->UseRealTime();

// Simple GEMM benchmarks for different matrix sizes
static void BM_GPU_GEMM_Small(benchmark::State& state) {
    int N = state.range(0);
    std::vector<float> A(N * N), B(N * N), C(N * N);
    
    // Initialize data
    std::fill(A.begin(), A.end(), 1.0f);
    std::fill(B.begin(), B.end(), 2.0f);
    
    for (auto _ : state) {
        gpu_gemm(A.data(), B.data(), C.data(), N, N, N);
        benchmark::DoNotOptimize(C.data());
    }
    
    // FLOPS calculation
    int64_t flops = static_cast<int64_t>(2) * N * N * N;
    state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) * flops);
}

static void BM_GPU_GEMM_Medium(benchmark::State& state) {
    int N = state.range(0);
    std::vector<float> A(N * N), B(N * N), C(N * N);
    
    // Initialize with random data for more realistic benchmark
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    
    for (int i = 0; i < N * N; ++i) {
        A[i] = dist(rng);
        B[i] = dist(rng);
    }
    
    for (auto _ : state) {
        gpu_gemm(A.data(), B.data(), C.data(), N, N, N);
        benchmark::DoNotOptimize(C.data());
    }
    
    int64_t flops = static_cast<int64_t>(2) * N * N * N;
    state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) * flops);
    
    // Memory bandwidth calculation
    int64_t bytes = static_cast<int64_t>(3) * N * N * sizeof(float);  // Read A, B; Write C
    state.SetBytesProcessed(static_cast<int64_t>(state.iterations()) * bytes);
}

// Rectangular matrix benchmarks
static void BM_GPU_GEMM_Rectangular(benchmark::State& state) {
    int M = state.range(0);
    int N = state.range(1); 
    int K = state.range(2);
    
    std::vector<float> A(M * K), B(K * N), C(M * N);
    
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    
    for (size_t i = 0; i < A.size(); ++i) A[i] = dist(rng);
    for (size_t i = 0; i < B.size(); ++i) B[i] = dist(rng);
    
    for (auto _ : state) {
        gpu_gemm(A.data(), B.data(), C.data(), M, N, K);
        benchmark::DoNotOptimize(C.data());
    }
    
    int64_t flops = static_cast<int64_t>(2) * M * N * K;
    state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) * flops);
}

// Threading boundary tests
static void BM_GPU_GEMM_ThreadBoundaries(benchmark::State& state) {
    int N = state.range(0);
    std::vector<float> A(N * N, 1.0f), B(N * N, 2.0f), C(N * N);
    
    for (auto _ : state) {
        gpu_gemm(A.data(), B.data(), C.data(), N, N, N);
        benchmark::DoNotOptimize(C.data());
        benchmark::ClobberMemory();
    }
    
    int64_t flops = static_cast<int64_t>(2) * N * N * N;
    state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) * flops);
}

// Register benchmarks with different ranges
BENCHMARK(BM_GPU_GEMM_Small)
    ->Range(8, 64)  // Small matrices 8x8 to 64x64
    ->Unit(benchmark::kMicrosecond);

BENCHMARK(BM_GPU_GEMM_Medium)
    ->Range(128, 1024)  // Medium matrices 128x128 to 1024x1024
    ->Unit(benchmark::kMillisecond)
    ->UseRealTime();

// Rectangular matrix benchmarks
BENCHMARK(BM_GPU_GEMM_Rectangular)
    ->Args({64, 32, 128})   // Tall-skinny: 64x32x128
    ->Args({32, 128, 64})   // Short-wide: 32x128x64
    ->Args({128, 64, 32})   // Medium rectangular
    ->Args({256, 128, 64})  // Larger rectangular
    ->Unit(benchmark::kMillisecond);

// Thread boundary specific sizes (multiples and near-multiples of common block sizes)
BENCHMARK(BM_GPU_GEMM_ThreadBoundaries)
    ->DenseRange(15, 17, 1)    // Around 16
    ->DenseRange(31, 33, 1)    // Around 32
    ->DenseRange(63, 65, 1)    // Around 64
    ->DenseRange(127, 129, 1)  // Around 128
    ->DenseRange(255, 257, 1)  // Around 256
    ->Unit(benchmark::kMicrosecond);

BENCHMARK_MAIN();