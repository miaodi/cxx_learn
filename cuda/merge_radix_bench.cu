/**
 * Benchmark: Merge sort (CUB DeviceMergeSort) vs Radix sort (CUB DeviceRadixSort)
 * across different array lengths and key widths (8, 16, 32, 64 bit).
 *
 * Requires CUB 2.2+ (e.g. CUDA Toolkit 12.3+) for DeviceMergeSort.
 */

#include <benchmark/benchmark.h>
#include <cuda_runtime.h>
#include <cub/cub.cuh>

#include <cstdint>
#include <random>
#include <stdexcept>
#include <vector>

// Device-callable less-than comparator for CUB DeviceMergeSort
struct DeviceLess {
  template <typename T>
  __device__ __host__ constexpr bool operator()(const T& a, const T& b) const {
    return a < b;
  }
};

namespace {

enum KeyBits : int { Bits8 = 0, Bits16 = 1, Bits32 = 2, Bits64 = 3 };

const char* key_bits_name(int bits) {
  switch (bits) {
    case Bits8: return "8bit";
    case Bits16: return "16bit";
    case Bits32: return "32bit";
    case Bits64: return "64bit";
    default: return "?";
  }
}

bool check_cuda(cudaError_t err, benchmark::State& state, const char* msg) {
  if (err != cudaSuccess) {
    state.SkipWithError(std::string(msg) + ": " + cudaGetErrorString(err));
    return false;
  }
  return true;
}

// ---- 8-bit (uint8_t) ----
static void BM_MergeSort_8bit(benchmark::State& state) {
  const size_t n = static_cast<size_t>(state.range(0));
  std::mt19937 rng(42);
  std::uniform_int_distribution<int> dist(0, 255);
  std::vector<uint8_t> h_data(n);
  for (size_t i = 0; i < n; ++i) h_data[i] = static_cast<uint8_t>(dist(rng));

  uint8_t* d_in = nullptr;
  uint8_t* d_out = nullptr;
  if (!check_cuda(cudaMalloc(&d_in, n * sizeof(uint8_t)), state, "malloc in")) return;
  if (!check_cuda(cudaMalloc(&d_out, n * sizeof(uint8_t)), state, "malloc out")) {
    cudaFree(d_in);
    return;
  }
  if (!check_cuda(cudaMemcpy(d_in, h_data.data(), n * sizeof(uint8_t), cudaMemcpyHostToDevice),
                  state, "H2D")) {
    cudaFree(d_in);
    cudaFree(d_out);
    return;
  }

  size_t temp_bytes = 0;
  cub::DeviceMergeSort::SortKeysCopy(nullptr, temp_bytes, d_in, d_out, n, DeviceLess{});
  void* d_temp = nullptr;
  if (!check_cuda(cudaMalloc(&d_temp, temp_bytes), state, "malloc temp")) {
    cudaFree(d_in);
    cudaFree(d_out);
    return;
  }

  for (auto _ : state) {
    cub::DeviceMergeSort::SortKeysCopy(d_temp, temp_bytes, d_in, d_out, n, DeviceLess{});
    if (!check_cuda(cudaDeviceSynchronize(), state, "sync")) break;
  }

  cudaFree(d_temp);
  cudaFree(d_in);
  cudaFree(d_out);
  state.SetItemsProcessed(state.iterations() * static_cast<int64_t>(n));
  state.SetBytesProcessed(state.iterations() * static_cast<int64_t>(n * sizeof(uint8_t)));
  state.SetLabel(key_bits_name(Bits8));
}

static void BM_RadixSort_8bit(benchmark::State& state) {
  const size_t n = static_cast<size_t>(state.range(0));
  std::mt19937 rng(42);
  std::uniform_int_distribution<int> dist(0, 255);
  std::vector<uint8_t> h_data(n);
  for (size_t i = 0; i < n; ++i) h_data[i] = static_cast<uint8_t>(dist(rng));

  uint8_t* d_in = nullptr;
  uint8_t* d_out = nullptr;
  if (!check_cuda(cudaMalloc(&d_in, n * sizeof(uint8_t)), state, "malloc in")) return;
  if (!check_cuda(cudaMalloc(&d_out, n * sizeof(uint8_t)), state, "malloc out")) {
    cudaFree(d_in);
    return;
  }
  if (!check_cuda(cudaMemcpy(d_in, h_data.data(), n * sizeof(uint8_t), cudaMemcpyHostToDevice),
                  state, "H2D")) {
    cudaFree(d_in);
    cudaFree(d_out);
    return;
  }

  size_t temp_bytes = 0;
  cub::DeviceRadixSort::SortKeys(nullptr, temp_bytes, d_in, d_out, n, 0, 8);
  void* d_temp = nullptr;
  if (!check_cuda(cudaMalloc(&d_temp, temp_bytes), state, "malloc temp")) {
    cudaFree(d_in);
    cudaFree(d_out);
    return;
  }

  for (auto _ : state) {
    cub::DeviceRadixSort::SortKeys(d_temp, temp_bytes, d_in, d_out, n, 0, 8);
    if (!check_cuda(cudaDeviceSynchronize(), state, "sync")) break;
  }

  cudaFree(d_temp);
  cudaFree(d_in);
  cudaFree(d_out);
  state.SetItemsProcessed(state.iterations() * static_cast<int64_t>(n));
  state.SetBytesProcessed(state.iterations() * static_cast<int64_t>(n * sizeof(uint8_t)));
  state.SetLabel(key_bits_name(Bits8));
}

// ---- 16-bit (uint16_t) ----
static void BM_MergeSort_16bit(benchmark::State& state) {
  const size_t n = static_cast<size_t>(state.range(0));
  std::mt19937 rng(42);
  std::uniform_int_distribution<int> dist(0, 65535);
  std::vector<uint16_t> h_data(n);
  for (size_t i = 0; i < n; ++i) h_data[i] = static_cast<uint16_t>(dist(rng));

  uint16_t* d_in = nullptr;
  uint16_t* d_out = nullptr;
  if (!check_cuda(cudaMalloc(&d_in, n * sizeof(uint16_t)), state, "malloc in")) return;
  if (!check_cuda(cudaMalloc(&d_out, n * sizeof(uint16_t)), state, "malloc out")) {
    cudaFree(d_in);
    return;
  }
  if (!check_cuda(cudaMemcpy(d_in, h_data.data(), n * sizeof(uint16_t), cudaMemcpyHostToDevice),
                  state, "H2D")) {
    cudaFree(d_in);
    cudaFree(d_out);
    return;
  }

  size_t temp_bytes = 0;
  cub::DeviceMergeSort::SortKeysCopy(nullptr, temp_bytes, d_in, d_out, n, DeviceLess{});
  void* d_temp = nullptr;
  if (!check_cuda(cudaMalloc(&d_temp, temp_bytes), state, "malloc temp")) {
    cudaFree(d_in);
    cudaFree(d_out);
    return;
  }

  for (auto _ : state) {
    cub::DeviceMergeSort::SortKeysCopy(d_temp, temp_bytes, d_in, d_out, n, DeviceLess{});
    if (!check_cuda(cudaDeviceSynchronize(), state, "sync")) break;
  }

  cudaFree(d_temp);
  cudaFree(d_in);
  cudaFree(d_out);
  state.SetItemsProcessed(state.iterations() * static_cast<int64_t>(n));
  state.SetBytesProcessed(state.iterations() * static_cast<int64_t>(n * sizeof(uint16_t)));
  state.SetLabel(key_bits_name(Bits16));
}

static void BM_RadixSort_16bit(benchmark::State& state) {
  const size_t n = static_cast<size_t>(state.range(0));
  std::mt19937 rng(42);
  std::uniform_int_distribution<int> dist(0, 65535);
  std::vector<uint16_t> h_data(n);
  for (size_t i = 0; i < n; ++i) h_data[i] = static_cast<uint16_t>(dist(rng));

  uint16_t* d_in = nullptr;
  uint16_t* d_out = nullptr;
  if (!check_cuda(cudaMalloc(&d_in, n * sizeof(uint16_t)), state, "malloc in")) return;
  if (!check_cuda(cudaMalloc(&d_out, n * sizeof(uint16_t)), state, "malloc out")) {
    cudaFree(d_in);
    return;
  }
  if (!check_cuda(cudaMemcpy(d_in, h_data.data(), n * sizeof(uint16_t), cudaMemcpyHostToDevice),
                  state, "H2D")) {
    cudaFree(d_in);
    cudaFree(d_out);
    return;
  }

  size_t temp_bytes = 0;
  cub::DeviceRadixSort::SortKeys(nullptr, temp_bytes, d_in, d_out, n, 0, 16);
  void* d_temp = nullptr;
  if (!check_cuda(cudaMalloc(&d_temp, temp_bytes), state, "malloc temp")) {
    cudaFree(d_in);
    cudaFree(d_out);
    return;
  }

  for (auto _ : state) {
    cub::DeviceRadixSort::SortKeys(d_temp, temp_bytes, d_in, d_out, n, 0, 16);
    if (!check_cuda(cudaDeviceSynchronize(), state, "sync")) break;
  }

  cudaFree(d_temp);
  cudaFree(d_in);
  cudaFree(d_out);
  state.SetItemsProcessed(state.iterations() * static_cast<int64_t>(n));
  state.SetBytesProcessed(state.iterations() * static_cast<int64_t>(n * sizeof(uint16_t)));
  state.SetLabel(key_bits_name(Bits16));
}

// ---- 32-bit (uint32_t) ----
static void BM_MergeSort_32bit(benchmark::State& state) {
  const size_t n = static_cast<size_t>(state.range(0));
  std::mt19937 rng(42);
  std::uniform_int_distribution<uint32_t> dist(0, UINT32_MAX);
  std::vector<uint32_t> h_data(n);
  for (size_t i = 0; i < n; ++i) h_data[i] = dist(rng);

  uint32_t* d_in = nullptr;
  uint32_t* d_out = nullptr;
  if (!check_cuda(cudaMalloc(&d_in, n * sizeof(uint32_t)), state, "malloc in")) return;
  if (!check_cuda(cudaMalloc(&d_out, n * sizeof(uint32_t)), state, "malloc out")) {
    cudaFree(d_in);
    return;
  }
  if (!check_cuda(cudaMemcpy(d_in, h_data.data(), n * sizeof(uint32_t), cudaMemcpyHostToDevice),
                  state, "H2D")) {
    cudaFree(d_in);
    cudaFree(d_out);
    return;
  }

  size_t temp_bytes = 0;
  cub::DeviceMergeSort::SortKeysCopy(nullptr, temp_bytes, d_in, d_out, n, DeviceLess{});
  void* d_temp = nullptr;
  if (!check_cuda(cudaMalloc(&d_temp, temp_bytes), state, "malloc temp")) {
    cudaFree(d_in);
    cudaFree(d_out);
    return;
  }

  for (auto _ : state) {
    cub::DeviceMergeSort::SortKeysCopy(d_temp, temp_bytes, d_in, d_out, n, DeviceLess{});
    if (!check_cuda(cudaDeviceSynchronize(), state, "sync")) break;
  }

  cudaFree(d_temp);
  cudaFree(d_in);
  cudaFree(d_out);
  state.SetItemsProcessed(state.iterations() * static_cast<int64_t>(n));
  state.SetBytesProcessed(state.iterations() * static_cast<int64_t>(n * sizeof(uint32_t)));
  state.SetLabel(key_bits_name(Bits32));
}

static void BM_RadixSort_32bit(benchmark::State& state) {
  const size_t n = static_cast<size_t>(state.range(0));
  std::mt19937 rng(42);
  std::uniform_int_distribution<uint32_t> dist(0, UINT32_MAX);
  std::vector<uint32_t> h_data(n);
  for (size_t i = 0; i < n; ++i) h_data[i] = dist(rng);

  uint32_t* d_in = nullptr;
  uint32_t* d_out = nullptr;
  if (!check_cuda(cudaMalloc(&d_in, n * sizeof(uint32_t)), state, "malloc in")) return;
  if (!check_cuda(cudaMalloc(&d_out, n * sizeof(uint32_t)), state, "malloc out")) {
    cudaFree(d_in);
    return;
  }
  if (!check_cuda(cudaMemcpy(d_in, h_data.data(), n * sizeof(uint32_t), cudaMemcpyHostToDevice),
                  state, "H2D")) {
    cudaFree(d_in);
    cudaFree(d_out);
    return;
  }

  size_t temp_bytes = 0;
  cub::DeviceRadixSort::SortKeys(nullptr, temp_bytes, d_in, d_out, n);
  void* d_temp = nullptr;
  if (!check_cuda(cudaMalloc(&d_temp, temp_bytes), state, "malloc temp")) {
    cudaFree(d_in);
    cudaFree(d_out);
    return;
  }

  for (auto _ : state) {
    cub::DeviceRadixSort::SortKeys(d_temp, temp_bytes, d_in, d_out, n);
    if (!check_cuda(cudaDeviceSynchronize(), state, "sync")) break;
  }

  cudaFree(d_temp);
  cudaFree(d_in);
  cudaFree(d_out);
  state.SetItemsProcessed(state.iterations() * static_cast<int64_t>(n));
  state.SetBytesProcessed(state.iterations() * static_cast<int64_t>(n * sizeof(uint32_t)));
  state.SetLabel(key_bits_name(Bits32));
}

// ---- 64-bit (uint64_t) ----
static void BM_MergeSort_64bit(benchmark::State& state) {
  const size_t n = static_cast<size_t>(state.range(0));
  std::mt19937 rng(42);
  std::uniform_int_distribution<uint64_t> dist(0, UINT64_MAX);
  std::vector<uint64_t> h_data(n);
  for (size_t i = 0; i < n; ++i) h_data[i] = dist(rng);

  uint64_t* d_in = nullptr;
  uint64_t* d_out = nullptr;
  if (!check_cuda(cudaMalloc(&d_in, n * sizeof(uint64_t)), state, "malloc in")) return;
  if (!check_cuda(cudaMalloc(&d_out, n * sizeof(uint64_t)), state, "malloc out")) {
    cudaFree(d_in);
    return;
  }
  if (!check_cuda(cudaMemcpy(d_in, h_data.data(), n * sizeof(uint64_t), cudaMemcpyHostToDevice),
                  state, "H2D")) {
    cudaFree(d_in);
    cudaFree(d_out);
    return;
  }

  size_t temp_bytes = 0;
  cub::DeviceMergeSort::SortKeysCopy(nullptr, temp_bytes, d_in, d_out, n, DeviceLess{});
  void* d_temp = nullptr;
  if (!check_cuda(cudaMalloc(&d_temp, temp_bytes), state, "malloc temp")) {
    cudaFree(d_in);
    cudaFree(d_out);
    return;
  }

  for (auto _ : state) {
    cub::DeviceMergeSort::SortKeysCopy(d_temp, temp_bytes, d_in, d_out, n, DeviceLess{});
    if (!check_cuda(cudaDeviceSynchronize(), state, "sync")) break;
  }

  cudaFree(d_temp);
  cudaFree(d_in);
  cudaFree(d_out);
  state.SetItemsProcessed(state.iterations() * static_cast<int64_t>(n));
  state.SetBytesProcessed(state.iterations() * static_cast<int64_t>(n * sizeof(uint64_t)));
  state.SetLabel(key_bits_name(Bits64));
}

static void BM_RadixSort_64bit(benchmark::State& state) {
  const size_t n = static_cast<size_t>(state.range(0));
  std::mt19937 rng(42);
  std::uniform_int_distribution<uint64_t> dist(0, UINT64_MAX);
  std::vector<uint64_t> h_data(n);
  for (size_t i = 0; i < n; ++i) h_data[i] = dist(rng);

  uint64_t* d_in = nullptr;
  uint64_t* d_out = nullptr;
  if (!check_cuda(cudaMalloc(&d_in, n * sizeof(uint64_t)), state, "malloc in")) return;
  if (!check_cuda(cudaMalloc(&d_out, n * sizeof(uint64_t)), state, "malloc out")) {
    cudaFree(d_in);
    return;
  }
  if (!check_cuda(cudaMemcpy(d_in, h_data.data(), n * sizeof(uint64_t), cudaMemcpyHostToDevice),
                  state, "H2D")) {
    cudaFree(d_in);
    cudaFree(d_out);
    return;
  }

  size_t temp_bytes = 0;
  cub::DeviceRadixSort::SortKeys(nullptr, temp_bytes, d_in, d_out, n);
  void* d_temp = nullptr;
  if (!check_cuda(cudaMalloc(&d_temp, temp_bytes), state, "malloc temp")) {
    cudaFree(d_in);
    cudaFree(d_out);
    return;
  }

  for (auto _ : state) {
    cub::DeviceRadixSort::SortKeys(d_temp, temp_bytes, d_in, d_out, n);
    if (!check_cuda(cudaDeviceSynchronize(), state, "sync")) break;
  }

  cudaFree(d_temp);
  cudaFree(d_in);
  cudaFree(d_out);
  state.SetItemsProcessed(state.iterations() * static_cast<int64_t>(n));
  state.SetBytesProcessed(state.iterations() * static_cast<int64_t>(n * sizeof(uint64_t)));
  state.SetLabel(key_bits_name(Bits64));
}

void apply_sizes(benchmark::internal::Benchmark* b) {
  for (int exp = 10; exp <= 26; exp += 2)
    b->Arg(1 << exp);
  b->Unit(benchmark::kMillisecond)->UseRealTime();
}

}  // namespace

// Merge sort = CUB DeviceMergeSort::SortKeysCopy
BENCHMARK(BM_MergeSort_8bit)->Apply(apply_sizes)->ArgName("n");
BENCHMARK(BM_MergeSort_16bit)->Apply(apply_sizes)->ArgName("n");
BENCHMARK(BM_MergeSort_32bit)->Apply(apply_sizes)->ArgName("n");
BENCHMARK(BM_MergeSort_64bit)->Apply(apply_sizes)->ArgName("n");

// Radix sort = CUB DeviceRadixSort::SortKeys
BENCHMARK(BM_RadixSort_8bit)->Apply(apply_sizes)->ArgName("n");
BENCHMARK(BM_RadixSort_16bit)->Apply(apply_sizes)->ArgName("n");
BENCHMARK(BM_RadixSort_32bit)->Apply(apply_sizes)->ArgName("n");
BENCHMARK(BM_RadixSort_64bit)->Apply(apply_sizes)->ArgName("n");

BENCHMARK_MAIN();
