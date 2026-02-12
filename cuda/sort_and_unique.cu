#include <benchmark/benchmark.h>
#include <cuda_runtime.h>

#include <cub/cub.cuh>

#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/sort.h>
#include <thrust/unique.h>

#include <cstdint>
#include <exception>
#include <random>
#include <string>
#include <vector>

namespace {

using KeyT = uint64_t;

bool check_cuda(cudaError_t status, benchmark::State& state, const char* step) {
  if (status == cudaSuccess) {
    return true;
  }
  std::string msg = std::string(step) + ": " + cudaGetErrorString(status);
  state.SkipWithError(msg.c_str());
  return false;
}

std::vector<KeyT> make_random_input(size_t n) {
  // Keep value range narrower than n to ensure duplicates are common.
  const KeyT max_value = static_cast<KeyT>(n > 1 ? (n / 4) : 1);
  std::mt19937 rng(static_cast<uint32_t>(0xC0FFEEu + n));
  std::uniform_int_distribution<KeyT> dist(0, max_value);

  std::vector<KeyT> out(n);
  for (size_t i = 0; i < n; ++i) {
    out[i] = dist(rng);
  }
  return out;
}

void set_metrics(benchmark::State& state, size_t n) {
  const int64_t bytes_per_iter = static_cast<int64_t>(n * sizeof(KeyT));
  state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) *
                          static_cast<int64_t>(n));
  state.SetBytesProcessed(static_cast<int64_t>(state.iterations()) *
                          bytes_per_iter);
}

static void BM_SortUnique_CUB(benchmark::State& state) {
  const size_t n = static_cast<size_t>(state.range(0));
  const std::vector<KeyT> h_input = make_random_input(n);

  KeyT* d_input = nullptr;
  KeyT* d_work = nullptr;
  KeyT* d_sorted = nullptr;
  KeyT* d_unique_out = nullptr;
  int* d_unique_count = nullptr;

  if (!check_cuda(cudaMalloc(&d_input, n * sizeof(KeyT)), state, "cudaMalloc d_input")) {
    goto cleanup;
  }
  if (!check_cuda(cudaMalloc(&d_work, n * sizeof(KeyT)), state, "cudaMalloc d_work")) {
    goto cleanup;
  }
  if (!check_cuda(cudaMalloc(&d_sorted, n * sizeof(KeyT)), state, "cudaMalloc d_sorted")) {
    goto cleanup;
  }
  if (!check_cuda(cudaMalloc(&d_unique_out, n * sizeof(KeyT)), state,
                  "cudaMalloc d_unique_out")) {
    goto cleanup;
  }
  if (!check_cuda(cudaMalloc(&d_unique_count, sizeof(int)), state,
                  "cudaMalloc d_unique_count")) {
    goto cleanup;
  }

  if (!check_cuda(cudaMemcpy(d_input, h_input.data(), n * sizeof(KeyT), cudaMemcpyHostToDevice),
                  state, "cudaMemcpy H2D")) {
    goto cleanup;
  }

  for (auto _ : state) {
    void* d_temp_storage = nullptr;
    size_t sort_storage_bytes = 0;
    size_t unique_storage_bytes = 0;

    if (!check_cuda(cub::DeviceRadixSort::SortKeys(nullptr, sort_storage_bytes,
                                                    d_work, d_sorted, n),
                    state, "CUB SortKeys temp query")) {
      break;
    }
    if (!check_cuda(cub::DeviceSelect::Unique(nullptr, unique_storage_bytes,
                                               d_sorted, d_unique_out,
                                               d_unique_count, n),
                    state, "CUB Unique temp query")) {
      break;
    }
    const size_t temp_storage_bytes =
        sort_storage_bytes > unique_storage_bytes ? sort_storage_bytes
                                                  : unique_storage_bytes;
    if (!check_cuda(cudaMalloc(&d_temp_storage, temp_storage_bytes), state,
                    "cudaMalloc shared temp")) {
      break;
    }

    if (!check_cuda(cudaMemcpy(d_work, d_input, n * sizeof(KeyT), cudaMemcpyDeviceToDevice),
                    state, "cudaMemcpy D2D reset")) {
      cudaFree(d_temp_storage);
      break;
    }

    if (!check_cuda(cub::DeviceRadixSort::SortKeys(d_temp_storage, sort_storage_bytes,
                                                    d_work, d_sorted, n),
                    state, "CUB SortKeys")) {
      cudaFree(d_temp_storage);
      break;
    }

    if (!check_cuda(cub::DeviceSelect::Unique(d_temp_storage, unique_storage_bytes,
                                               d_sorted, d_unique_out,
                                               d_unique_count, n),
                    state, "CUB Unique")) {
      cudaFree(d_temp_storage);
      break;
    }

    if (!check_cuda(cudaDeviceSynchronize(), state, "cudaDeviceSynchronize")) {
      cudaFree(d_temp_storage);
      break;
    }

    cudaFree(d_temp_storage);
  }

  set_metrics(state, n);

cleanup:
  cudaFree(d_input);
  cudaFree(d_work);
  cudaFree(d_sorted);
  cudaFree(d_unique_out);
  cudaFree(d_unique_count);
}

static void BM_SortUnique_Thrust(benchmark::State& state) {
  const size_t n = static_cast<size_t>(state.range(0));
  try {
    const std::vector<KeyT> h_input = make_random_input(n);
    thrust::device_vector<KeyT> d_input(h_input.begin(), h_input.end());
    thrust::device_vector<KeyT> d_work(n);

    for (auto _ : state) {
      thrust::copy(thrust::device, d_input.begin(), d_input.end(), d_work.begin());
      thrust::sort(thrust::device, d_work.begin(), d_work.end());
      auto unique_end = thrust::unique(thrust::device, d_work.begin(), d_work.end());

      if (!check_cuda(cudaDeviceSynchronize(), state, "cudaDeviceSynchronize")) {
        break;
      }

      benchmark::DoNotOptimize(unique_end);
    }
  } catch (const std::exception& e) {
    state.SkipWithError(e.what());
    return;
  }

  set_metrics(state, n);
}

void configure_sizes(benchmark::internal::Benchmark* b) {
  for (int exp = 10; exp <= 26; exp += 4) {
    b->Arg(1 << exp);
  }
  b->UseRealTime()->Unit(benchmark::kMillisecond);
}

}  // namespace

BENCHMARK(BM_SortUnique_CUB)->Apply(configure_sizes);
BENCHMARK(BM_SortUnique_Thrust)->Apply(configure_sizes);

BENCHMARK_MAIN();
