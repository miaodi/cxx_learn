// Memory Stride Benchmark: Demonstrates GPU Memory Coalescing
//
// This benchmark measures the impact of memory access patterns on GPU performance.
// It shows how different stride values affect memory bandwidth due to:
// 1. Memory coalescing: Adjacent threads accessing adjacent memory locations
// 2. Bank conflicts: Power-of-2 strides causing serialized memory accesses
//
// Key observations:
// - Stride 1: Optimal coalescing, maximum bandwidth (~2 TiB/s)
// - Stride 2-31: Reduced coalescing but no bank conflicts (~900-1200 GiB/s)
// - Stride 32: Severe bank conflicts, lowest bandwidth (~400-500 GiB/s)
//   (32-way bank conflict on typical GPUs with 32 memory banks)
//
// Educational value: Clearly demonstrates why sequential memory access patterns
// are critical for GPU performance.

#include <benchmark/benchmark.h>
#include <cuda_runtime.h>

#include <stdexcept>
#include <string>
#include <vector>

#define CUDA_CHECK(call)                                                     \
  do {                                                                       \
    cudaError_t err__ = (call);                                              \
    if (err__ != cudaSuccess) {                                              \
      throw std::runtime_error(std::string("CUDA error: ") +                 \
                               cudaGetErrorString(err__));                   \
    }                                                                        \
  } while (0)

namespace {

// Constants for benchmark configuration
constexpr size_t kDataElements = 1 << 24;      // 16M floats (~64 MiB)
constexpr size_t kTotalReadsPerLaunch = 1 << 26; // Target reads per kernel launch
constexpr int kBlockSize = 256;                 // Threads per block
constexpr int kGridSize = 256;                  // Number of blocks

// Kernel that reads global memory with a configurable stride
// Each thread reads from indices: tid*stride, tid*stride+stride, ...
// Performance depends heavily on the stride value due to coalescing
__global__ void stride_read_kernel(const float *data, float *sink,
                                   size_t elements, int stride,
                                   int iterations_per_thread) {
  const size_t tid = blockIdx.x * blockDim.x + threadIdx.x;

  // Start at position based on thread ID and stride
  size_t index = (tid * static_cast<size_t>(stride)) % elements;
  float accum = 0.0f;

  // Perform multiple reads to increase sample size
  for (int i = 0; i < iterations_per_thread; ++i) {
    accum += data[index];
    index += stride;
    if (index >= elements) {
      index -= elements;
    }
  }

  // Store result to prevent compiler optimization
  sink[tid] = accum;
}

} // namespace

// Benchmark function: Measures global memory read performance with different strides
// 
// How it works:
// - Allocates 64 MiB of device memory
// - Launches kernel where each thread reads with a specific stride
// - Measures bandwidth achieved for different stride values
//
// Expected results:
// - Stride 1: Best performance (coalesced access, ~2 TiB/s)
// - Stride 2-31: Moderate performance (partial coalescing, ~1 TiB/s)
// - Stride 32: Worst performance (32-way bank conflict, ~400 GiB/s)
static void BM_GlobalMemoryStride(benchmark::State &state) {
  const int stride = state.range(0);
  if (stride <= 0) {
    state.SkipWithError("Stride must be positive");
    return;
  }

  // Initialize host data with sequential pattern
  std::vector<float> host_data(kDataElements);
  for (size_t i = 0; i < host_data.size(); ++i) {
    host_data[i] = static_cast<float>(i % 1024);
  }

  float *device_data = nullptr;
  float *device_sink = nullptr;

  const size_t data_bytes = host_data.size() * sizeof(float);
  const size_t sink_elements = static_cast<size_t>(kGridSize) * kBlockSize;
  const size_t sink_bytes = sink_elements * sizeof(float);

  // Allocate and copy to device
  CUDA_CHECK(cudaMalloc(&device_data, data_bytes));
  CUDA_CHECK(cudaMalloc(&device_sink, sink_bytes));
  CUDA_CHECK(
      cudaMemcpy(device_data, host_data.data(), data_bytes, cudaMemcpyHostToDevice));

  // Calculate iterations to achieve target read count
  const size_t total_threads = sink_elements;
  const size_t reads_per_launch = kTotalReadsPerLaunch;
  const size_t min_iterations = reads_per_launch / total_threads;
  const int iterations_per_thread =
      static_cast<int>(min_iterations > 0 ? min_iterations : 1);
  const size_t actual_reads_per_launch =
      static_cast<size_t>(iterations_per_thread) * total_threads;

  // Benchmark loop
  for (auto _ : state) {
    CUDA_CHECK(cudaMemset(device_sink, 0, sink_bytes));
    stride_read_kernel<<<kGridSize, kBlockSize>>>(
        device_data, device_sink, kDataElements, stride, iterations_per_thread);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
  }

  CUDA_CHECK(cudaFree(device_data));
  CUDA_CHECK(cudaFree(device_sink));

  // Report bandwidth (bytes read / time)
  const int64_t bytes_per_iteration =
      static_cast<int64_t>(actual_reads_per_launch * sizeof(float));
  state.SetBytesProcessed(state.iterations() * bytes_per_iteration);
  state.SetLabel("stride=" + std::to_string(stride));
}

// Run benchmark for stride values 1 to 32
// Stride 32 is particularly important as it triggers worst-case bank conflicts
BENCHMARK(BM_GlobalMemoryStride)
    ->DenseRange(1, 32, 1)
    ->Unit(benchmark::kMillisecond)
    ->UseRealTime();

// Power-of-2 stride benchmark: Demonstrates bank conflict severity
// 
// Power-of-2 strides cause bank conflicts because GPU memory is organized
// into 32 banks (on typical GPUs). When stride is a power of 2:
// - Stride 1: All 32 threads in a warp access different banks (optimal)
// - Stride 2, 4, 8, 16: Multiple threads access the same bank (conflicts)
// - Stride 32: All 32 threads access the SAME bank (32-way conflict!)
//
// Expected pattern:
// - Stride 1: ~2 TiB/s (perfect coalescing)
// - Stride 2-16: Decreasing performance as conflicts increase
// - Stride 32+: ~400-500 GiB/s (maximum serialization)
static void BM_GlobalMemoryStride_PowerOf2(benchmark::State &state) {
  BM_GlobalMemoryStride(state);
}

BENCHMARK(BM_GlobalMemoryStride_PowerOf2)
    ->RangeMultiplier(2)
    ->Range(1, 256)
    ->Unit(benchmark::kMillisecond)
    ->UseRealTime();

BENCHMARK_MAIN();
