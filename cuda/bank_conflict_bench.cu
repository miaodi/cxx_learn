// Bank Conflict Benchmark: Demonstrates Shared Memory Bank Conflicts
//
// This benchmark measures the impact of shared memory bank conflicts on GPU performance.
// Shared memory is divided into banks (typically 32 on modern GPUs). When multiple threads
// in a warp access the same bank simultaneously, the accesses are serialized.
//
// Key concepts:
// - No conflict: Each thread accesses a different bank (stride 1)
// - 2-way conflict: 2 threads access each bank (stride 2)
// - N-way conflict: N threads access each bank (stride N)
// - Broadcast: All threads read the same address (special case, no conflict)
//
// Educational value: Shows why proper shared memory access patterns are critical
// for GPU performance and how stride affects bank conflicts.

#include <benchmark/benchmark.h>
#include <cuda_runtime.h>

#include <stdexcept>
#include <string>

#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t err__ = (call);                                                \
    if (err__ != cudaSuccess) {                                                \
      throw std::runtime_error(std::string("CUDA error: ") +                   \
                               cudaGetErrorString(err__));                     \
    }                                                                          \
  } while (0)

namespace {

// constexpr int kWarpSize = 32;
constexpr int kBlockSize = 256;
constexpr int kGridSize = 128;
constexpr int kSharedMemSize = 1024; // Number of floats in shared memory

// Kernel with configurable stride for shared memory access
// Each thread reads from shared memory with the given stride pattern
__global__ void bank_conflict_kernel(float *output, int stride,
                                     int iterations) {
  __shared__ float shared_data[kSharedMemSize];

  const int tid = threadIdx.x;
  const int global_tid = blockIdx.x * blockDim.x + threadIdx.x;

  // Initialize shared memory
  if (tid < kSharedMemSize) {
    shared_data[tid] = static_cast<float>(tid);
  }
  __syncthreads();

  float accum = 0.0f;

  // Perform multiple reads with the given stride
  // The stride determines the bank conflict pattern:
  // - stride 1: No conflicts (each thread accesses different bank)
  // - stride 2: 2-way conflicts (every 2 threads hit same bank)
  // - stride 16: 16-way conflicts
  // - stride 32: 32-way conflicts (worst case for 32-bank system)
  for (int i = 0; i < iterations; ++i) {
    int index = (tid * stride + i) % kSharedMemSize;
    accum += shared_data[index];
  }

  // Write result to prevent compiler optimization
  output[global_tid] = accum;
}

// Broadcast access pattern - all threads read same location
// This is a special case that doesn't cause conflicts
__global__ void broadcast_kernel(float *output, int iterations) {
  __shared__ float shared_data[kSharedMemSize];

  const int tid = threadIdx.x;
  const int global_tid = blockIdx.x * blockDim.x + threadIdx.x;

  // Initialize shared memory
  if (tid < kSharedMemSize) {
    shared_data[tid] = static_cast<float>(tid);
  }
  __syncthreads();

  float accum = 0.0f;

  // All threads read from the same location (broadcast)
  // This should NOT cause conflicts due to hardware optimization
  for (int i = 0; i < iterations; ++i) {
    accum += shared_data[0]; // All threads read same address
  }

  output[global_tid] = accum;
}

} // namespace

// Benchmark bank conflicts with different stride patterns
static void BM_BankConflict(benchmark::State &state) {
  const int stride = state.range(0);

  float *device_output = nullptr;
  const size_t output_bytes =
      static_cast<size_t>(kGridSize) * kBlockSize * sizeof(float);

  CUDA_CHECK(cudaMalloc(&device_output, output_bytes));

  // More iterations per thread to make timing more stable
  const int iterations = 1000;

  for (auto _ : state) {
    bank_conflict_kernel<<<kGridSize, kBlockSize>>>(device_output, stride,
                                                     iterations);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
  }

  CUDA_CHECK(cudaFree(device_output));

  // Calculate effective bandwidth
  // Each thread does 'iterations' reads from shared memory
  const int64_t total_threads =
      static_cast<int64_t>(kGridSize) * kBlockSize;
  const int64_t total_reads = total_threads * iterations;
  const int64_t bytes_read = total_reads * sizeof(float);
  state.SetBytesProcessed(state.iterations() * bytes_read);
  state.SetLabel("stride=" + std::to_string(stride));
}

// Benchmark broadcast access pattern
static void BM_BankConflict_Broadcast(benchmark::State &state) {
  float *device_output = nullptr;
  const size_t output_bytes =
      static_cast<size_t>(kGridSize) * kBlockSize * sizeof(float);

  CUDA_CHECK(cudaMalloc(&device_output, output_bytes));

  const int iterations = 1000;

  for (auto _ : state) {
    broadcast_kernel<<<kGridSize, kBlockSize>>>(device_output, iterations);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
  }

  CUDA_CHECK(cudaFree(device_output));

  const int64_t total_threads =
      static_cast<int64_t>(kGridSize) * kBlockSize;
  const int64_t total_reads = total_threads * iterations;
  const int64_t bytes_read = total_reads * sizeof(float);
  state.SetBytesProcessed(state.iterations() * bytes_read);
  state.SetLabel("broadcast");
}

// Test stride values 1 to 32 to show the full conflict spectrum
BENCHMARK(BM_BankConflict)
    ->DenseRange(1, 32, 1)
    ->Unit(benchmark::kMillisecond)
    ->UseRealTime();

// Power-of-2 strides to emphasize the conflict pattern
// These show clear degradation at each power of 2
BENCHMARK(BM_BankConflict)
    ->RangeMultiplier(2)
    ->Range(1, 32)
    ->Unit(benchmark::kMillisecond)
    ->UseRealTime();

// Test broadcast pattern (special case with no conflicts)
BENCHMARK(BM_BankConflict_Broadcast)->Unit(benchmark::kMillisecond)->UseRealTime();

BENCHMARK_MAIN();
