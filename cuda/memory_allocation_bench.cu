#include <benchmark/benchmark.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

namespace {

bool check_cuda(cudaError_t status, benchmark::State &state, const char *step) {
  if (status == cudaSuccess) {
    return true;
  }
  std::ostringstream oss;
  oss << step << " failed: " << cudaGetErrorString(status);
  state.SkipWithError(oss.str().c_str());
  return false;
}

std::string format_bytes(std::size_t bytes) {
  constexpr const char *kUnits[] = {"B", "KiB", "MiB", "GiB"};
  double value = static_cast<double>(bytes);
  int unit = 0;
  while (value >= 1024.0 && unit < 3) {
    value /= 1024.0;
    ++unit;
  }

  std::ostringstream oss;
  if (unit == 0) {
    oss << bytes << kUnits[unit];
  } else {
    oss.setf(std::ios::fixed);
    oss.precision(value >= 10.0 ? 0 : 1);
    oss << value << kUnits[unit];
  }
  return oss.str();
}

void set_common_metrics(benchmark::State &state, std::size_t bytes_per_alloc,
                        int alloc_count) {
  const auto total_bytes =
      static_cast<int64_t>(bytes_per_alloc) * static_cast<int64_t>(alloc_count);
  state.SetItemsProcessed(state.iterations() * alloc_count);
  state.SetBytesProcessed(state.iterations() * total_bytes);
  state.counters["allocs/s"] = benchmark::Counter(
      static_cast<double>(alloc_count), benchmark::Counter::kIsIterationInvariantRate);
  state.counters["bytes/s"] = benchmark::Counter(
      static_cast<double>(total_bytes),
      benchmark::Counter::kIsIterationInvariantRate,
      benchmark::Counter::OneK::kIs1024);
  state.SetLabel(format_bytes(bytes_per_alloc) + " x " + std::to_string(alloc_count));
}

void apply_allocation_args(benchmark::internal::Benchmark *bench) {
  const std::vector<std::pair<int64_t, int64_t>> configs = {
      {4LL * 1024, 1},
      {4LL * 1024, 256},
      {64LL * 1024, 1},
      {64LL * 1024, 256},
      {1LL << 20, 1},
      {1LL << 20, 64},
      {8LL << 20, 1},
      {8LL << 20, 64},
      {8LL << 24, 64},
  };

  for (const auto &[bytes, allocs] : configs) {
    bench->Args({bytes, allocs});
  }

  bench->Unit(benchmark::kMicrosecond)->UseRealTime();
}

static void BM_CudaMallocFree(benchmark::State &state) {
  const std::size_t bytes_per_alloc = static_cast<std::size_t>(state.range(0));
  const int alloc_count = static_cast<int>(state.range(1));

  int device_count = 0;
  const cudaError_t device_status = cudaGetDeviceCount(&device_count);
  if (device_status == cudaErrorNoDevice) {
    state.SkipWithError("No CUDA-capable device detected");
    return;
  }
  if (!check_cuda(device_status, state, "cudaGetDeviceCount")) {
    return;
  }

  std::vector<void *> ptrs(static_cast<std::size_t>(alloc_count), nullptr);

  for (auto _ : state) {
    for (int i = 0; i < alloc_count; ++i) {
      if (!check_cuda(cudaMalloc(&ptrs[static_cast<std::size_t>(i)], bytes_per_alloc),
                      state, "cudaMalloc")) {
        return;
      }
    }

    for (int i = alloc_count - 1; i >= 0; --i) {
      if (!check_cuda(cudaFree(ptrs[static_cast<std::size_t>(i)]), state,
                      "cudaFree")) {
        return;
      }
      ptrs[static_cast<std::size_t>(i)] = nullptr;
    }
  }

  set_common_metrics(state, bytes_per_alloc, alloc_count);
}

#if CUDART_VERSION >= 11020
struct AsyncPoolContext {
  benchmark::State &state;
  int device = 0;
  cudaStream_t stream = nullptr;
  cudaMemPool_t original_pool = nullptr;
  cudaMemPool_t benchmark_pool = nullptr;
  bool ready = false;

  explicit AsyncPoolContext(benchmark::State &s) : state(s) {
    int device_count = 0;
    const cudaError_t device_status = cudaGetDeviceCount(&device_count);
    if (device_status == cudaErrorNoDevice) {
      state.SkipWithError("No CUDA-capable device detected");
      return;
    }
    if (!check_cuda(device_status, state, "cudaGetDeviceCount")) {
      return;
    }
    if (!check_cuda(cudaGetDevice(&device), state, "cudaGetDevice")) {
      return;
    }

    int pools_supported = 0;
    if (!check_cuda(cudaDeviceGetAttribute(&pools_supported,
                                           cudaDevAttrMemoryPoolsSupported, device),
                    state, "cudaDeviceGetAttribute(memoryPoolsSupported)")) {
      return;
    }
    if (pools_supported == 0) {
      state.SkipWithError("Device does not support CUDA memory pools");
      return;
    }

    if (!check_cuda(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), state,
                    "cudaStreamCreateWithFlags")) {
      return;
    }
    if (!check_cuda(cudaDeviceGetMemPool(&original_pool, device), state,
                    "cudaDeviceGetMemPool")) {
      return;
    }

    cudaMemPoolProps props{};
    props.allocType = cudaMemAllocationTypePinned;
    props.handleTypes = cudaMemHandleTypeNone;
    props.location.type = cudaMemLocationTypeDevice;
    props.location.id = device;

    if (!check_cuda(cudaMemPoolCreate(&benchmark_pool, &props), state,
                    "cudaMemPoolCreate")) {
      return;
    }

    std::uint64_t release_threshold = std::numeric_limits<std::uint64_t>::max();
    if (!check_cuda(cudaMemPoolSetAttribute(benchmark_pool,
                                            cudaMemPoolAttrReleaseThreshold,
                                            &release_threshold),
                    state, "cudaMemPoolSetAttribute(releaseThreshold)")) {
      return;
    }

    if (!check_cuda(cudaDeviceSetMemPool(device, benchmark_pool), state,
                    "cudaDeviceSetMemPool")) {
      return;
    }

    ready = true;
  }

  ~AsyncPoolContext() {
    if (!ready) {
      if (benchmark_pool != nullptr) {
        cudaMemPoolDestroy(benchmark_pool);
      }
      if (stream != nullptr) {
        cudaStreamDestroy(stream);
      }
      return;
    }

    cudaStreamSynchronize(stream);
    cudaDeviceSetMemPool(device, original_pool);
    cudaMemPoolTrimTo(benchmark_pool, 0);
    cudaMemPoolDestroy(benchmark_pool);
    cudaStreamDestroy(stream);
  }
};

static void BM_CudaMallocAsyncPool(benchmark::State &state) {
  const std::size_t bytes_per_alloc = static_cast<std::size_t>(state.range(0));
  const int alloc_count = static_cast<int>(state.range(1));

  AsyncPoolContext context(state);
  if (!context.ready) {
    return;
  }

  std::vector<void *> ptrs(static_cast<std::size_t>(alloc_count), nullptr);

  void *warmup_ptr = nullptr;
  if (!check_cuda(cudaMallocAsync(&warmup_ptr, bytes_per_alloc, context.stream), state,
                  "cudaMallocAsync warmup")) {
    return;
  }
  if (!check_cuda(cudaFreeAsync(warmup_ptr, context.stream), state,
                  "cudaFreeAsync warmup")) {
    return;
  }
  if (!check_cuda(cudaStreamSynchronize(context.stream), state,
                  "cudaStreamSynchronize warmup")) {
    return;
  }

  for (auto _ : state) {
    for (int i = 0; i < alloc_count; ++i) {
      if (!check_cuda(cudaMallocAsync(&ptrs[static_cast<std::size_t>(i)],
                                      bytes_per_alloc, context.stream),
                      state, "cudaMallocAsync")) {
        return;
      }
    }

    for (int i = alloc_count - 1; i >= 0; --i) {
      if (!check_cuda(cudaFreeAsync(ptrs[static_cast<std::size_t>(i)], context.stream),
                      state, "cudaFreeAsync")) {
        return;
      }
      ptrs[static_cast<std::size_t>(i)] = nullptr;
    }

    if (!check_cuda(cudaStreamSynchronize(context.stream), state,
                    "cudaStreamSynchronize")) {
      return;
    }
  }

  set_common_metrics(state, bytes_per_alloc, alloc_count);
}
#else
static void BM_CudaMallocAsyncPool(benchmark::State &state) {
  state.SkipWithError("cudaMallocAsync requires CUDA 11.2 or newer");
}
#endif

} // namespace

BENCHMARK(BM_CudaMallocFree)->Apply(apply_allocation_args);
BENCHMARK(BM_CudaMallocAsyncPool)->Apply(apply_allocation_args);

BENCHMARK_MAIN();
