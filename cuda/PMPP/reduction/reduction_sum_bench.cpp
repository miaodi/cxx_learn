#include "reduction_sum.h"

#include <benchmark/benchmark.h>
#include <cuda_runtime.h>

#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void check_cuda(cudaError_t status, const char *operation) {
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string(operation) + ": " +
                             cudaGetErrorString(status));
  }
}

class SimpleStrideSumBenchmark : public benchmark::Fixture {
public:
  void SetUp(const benchmark::State &state) override {
    values.resize(static_cast<std::size_t>(state.range(0)));

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    for (float &value : values) {
      value = dist(rng);
    }

    check_cuda(cudaMalloc(&device_values, values.size() * sizeof(float)),
               "cudaMalloc device_values");
    check_cuda(cudaMemcpy(device_values, values.data(),
                          values.size() * sizeof(float),
                          cudaMemcpyHostToDevice),
               "cudaMemcpy device_values");
    pmpp::reduction::allocate_device_reduction_buffers(buffers, values.size());
  }

  void TearDown(const benchmark::State &) override {
    pmpp::reduction::free_device_reduction_buffers(buffers);
    cudaFree(device_values);
    device_values = nullptr;
  }

protected:
  std::vector<float> values;
  float *device_values = nullptr;
  pmpp::reduction::DeviceReductionBuffers buffers;
};

BENCHMARK_DEFINE_F(SimpleStrideSumBenchmark, CPU)(benchmark::State &state) {
  for (auto _ : state) {
    float result = std::accumulate(values.begin(), values.end(), 0.0f);
    benchmark::DoNotOptimize(result);
  }

  state.SetItemsProcessed(state.iterations() * state.range(0));
  state.SetBytesProcessed(state.iterations() * state.range(0) *
                          static_cast<int64_t>(sizeof(float)));
}

BENCHMARK_DEFINE_F(SimpleStrideSumBenchmark, GPU_Stride)(benchmark::State &state) {
  for (auto _ : state) {
    float result =
        pmpp::reduction::simple_stride_sum_device(device_values, values.size(),
                                                  buffers);
    benchmark::DoNotOptimize(result);
  }

  state.SetItemsProcessed(state.iterations() * state.range(0));
  state.SetBytesProcessed(state.iterations() * state.range(0) *
                          static_cast<int64_t>(sizeof(float)));
}

BENCHMARK_DEFINE_F(SimpleStrideSumBenchmark, GPU_SequentialAddressing)(
    benchmark::State &state) {
  for (auto _ : state) {
    float result = pmpp::reduction::sequential_addressing_sum_device(
        device_values, values.size(), buffers);
    benchmark::DoNotOptimize(result);
  }

  state.SetItemsProcessed(state.iterations() * state.range(0));
  state.SetBytesProcessed(state.iterations() * state.range(0) *
                          static_cast<int64_t>(sizeof(float)));
}

BENCHMARK_DEFINE_F(SimpleStrideSumBenchmark, GPU_CoarsenedSharedMemory)(
    benchmark::State &state) {
  for (auto _ : state) {
    float result = pmpp::reduction::coarsened_shared_memory_sum_device(
        device_values, values.size(), buffers);
    benchmark::DoNotOptimize(result);
  }

  state.SetItemsProcessed(state.iterations() * state.range(0));
  state.SetBytesProcessed(state.iterations() * state.range(0) *
                          static_cast<int64_t>(sizeof(float)));
}

BENCHMARK_DEFINE_F(SimpleStrideSumBenchmark,
                   GPU_CoarsenedSharedMemoryOptimized)(
    benchmark::State &state) {
  for (auto _ : state) {
    float result = pmpp::reduction::coarsened_shared_memory_sum_optimized_device(
        device_values, values.size(), buffers);
    benchmark::DoNotOptimize(result);
  }

  state.SetItemsProcessed(state.iterations() * state.range(0));
  state.SetBytesProcessed(state.iterations() * state.range(0) *
                          static_cast<int64_t>(sizeof(float)));
}

BENCHMARK_DEFINE_F(SimpleStrideSumBenchmark, GPU_ThrustReference)(
    benchmark::State &state) {
  for (auto _ : state) {
    float result =
        pmpp::reduction::thrust_reference_sum_device(device_values,
                                                     values.size());
    benchmark::DoNotOptimize(result);
  }

  state.SetItemsProcessed(state.iterations() * state.range(0));
  state.SetBytesProcessed(state.iterations() * state.range(0) *
                          static_cast<int64_t>(sizeof(float)));
}

BENCHMARK_REGISTER_F(SimpleStrideSumBenchmark, CPU)
    ->Range(1024, 1024 * 1024 * 16)
    ->Unit(benchmark::kMicrosecond)
    ->UseRealTime();

BENCHMARK_REGISTER_F(SimpleStrideSumBenchmark, GPU_Stride)
    ->Range(1024, 1024 * 1024 * 16)
    ->Unit(benchmark::kMicrosecond)
    ->UseRealTime();

BENCHMARK_REGISTER_F(SimpleStrideSumBenchmark, GPU_SequentialAddressing)
    ->Range(1024, 1024 * 1024 * 16)
    ->Unit(benchmark::kMicrosecond)
    ->UseRealTime();

BENCHMARK_REGISTER_F(SimpleStrideSumBenchmark, GPU_CoarsenedSharedMemory)
    ->Range(1024, 1024 * 1024 * 16)
    ->Unit(benchmark::kMicrosecond)
    ->UseRealTime();

BENCHMARK_REGISTER_F(SimpleStrideSumBenchmark,
                     GPU_CoarsenedSharedMemoryOptimized)
    ->Range(1024, 1024 * 1024 * 16)
    ->Unit(benchmark::kMicrosecond)
    ->UseRealTime();

BENCHMARK_REGISTER_F(SimpleStrideSumBenchmark, GPU_ThrustReference)
    ->Range(1024, 1024 * 1024 * 16)
    ->Unit(benchmark::kMicrosecond)
    ->UseRealTime();

} // namespace

BENCHMARK_MAIN();
