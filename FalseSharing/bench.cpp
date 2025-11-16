// False Sharing Benchmark Suite
//
// This benchmark demonstrates the performance impact of false sharing and various solutions.
// False sharing occurs when threads on different cores modify variables that reside on the
// same cache line, causing expensive cache coherency traffic.
//
// Typical cache line size: 64 bytes (16 ints)
//
// Benchmarks ordered from WORST to BEST performance:
// 1. OMP_ATOMIC_Reduce      - All threads contend on single atomic (extreme contention)
// 2. OMP_ARRAY_Reduce       - FALSE SHARING: Adjacent array elements on same cache line
// 3. OMP_ARRAY_OPT_Reduce   - Same false sharing issue as #2
// 4. OMP_ARRAY_ALIGN_Reduce - FIXED: Each thread's data on separate cache line (64-byte stride)
// 5. OMP_LOCAL_VAL_Reduce   - BEST: Each thread uses local variable (no sharing)

#include <algorithm>
#include <atomic>
#include <benchmark/benchmark.h>
#include <iostream>
#include <iterator>
#include <new>
#include <numeric>
#include <omp.h>
#include <random>
#include <vector>

static const int threads = 8;
static constexpr unsigned int random_seed = 12345; // Fixed seed for reproducible benchmarks

static void Serial_Reduce(benchmark::State &state) {
  // First create an instance of an engine.
  std::random_device rnd_device;
  // Specify the engine and distribution.
  std::mt19937 mersenne_engine{rnd_device()}; // Generates random integers
  std::uniform_int_distribution<int> dist{0, std::numeric_limits<int>::max()};

  auto gen = [&]() { return dist(mersenne_engine); };
  const size_t size = state.range(0);

  std::vector<int> vec(size);
  std::generate(vec.begin(), vec.end(), gen);
  int min_val{std::numeric_limits<int>::min()};

  for (auto _ : state) {
    for (size_t i = 0; i < vec.size(); i++)
      benchmark::DoNotOptimize(min_val = std::max(min_val, vec[i]));
  }
  state.SetBytesProcessed(int64_t(state.iterations()) *
                          int64_t(state.range(0)) * 4);
}

BENCHMARK(Serial_Reduce)->RangeMultiplier(8)->Range(1 << 10, 1 << 24);

template <typename T>
void update_maximum(std::atomic<T> &maximum_value, T const &value) noexcept {
  T prev_value = maximum_value;
  while (prev_value < value &&
         !maximum_value.compare_exchange_weak(prev_value, value)) {
  }
}

// WORST: Atomic operations - all threads contend on a single memory location
// This is worse than false sharing because of explicit synchronization overhead
static void OMP_ATOMIC_Reduce(benchmark::State &state) {
  omp_set_num_threads(threads);
  // First create an instance of an engine.
  std::random_device rnd_device;
  // Specify the engine and distribution.
  std::mt19937 mersenne_engine{rnd_device()}; // Generates random integers
  std::uniform_int_distribution<int> dist{0, std::numeric_limits<int>::max()};

  auto gen = [&]() { return dist(mersenne_engine); };
  const size_t size = state.range(0);

  std::vector<int> vec(size);
  std::generate(vec.begin(), vec.end(), gen);
  std::atomic<int> min_val{std::numeric_limits<int>::min()};
  for (auto _ : state) {
#pragma omp parallel for
    for (auto i : vec)
      update_maximum(min_val, i);
  }
  state.SetBytesProcessed(int64_t(state.iterations()) *
                          int64_t(state.range(0)) * 4);
}

BENCHMARK(OMP_ATOMIC_Reduce)->RangeMultiplier(8)->Range(1 << 10, 1 << 24);

// BAD: False sharing - thread_min_vals[0], [1], [2]... are adjacent in memory
// On a system with 64-byte cache lines, threads 0-15 share the same cache line!
// Each write by one thread invalidates the cache line for other threads
static void OMP_ARRAY_Reduce(benchmark::State &state) {
  omp_set_num_threads(threads);
  // First create an instance of an engine.
  std::random_device rnd_device;
  // Specify the engine and distribution.
  std::mt19937 mersenne_engine{rnd_device()}; // Generates random integers
  std::uniform_int_distribution<int> dist{0, std::numeric_limits<int>::max()};

  auto gen = [&]() { return dist(mersenne_engine); };
  const size_t size = state.range(0);

  std::vector<int> vec(size);
  std::generate(vec.begin(), vec.end(), gen);
  int min_val{std::numeric_limits<int>::min()};
  for (auto _ : state) {
    std::vector<int> thread_min_vals(threads, std::numeric_limits<int>::min());
#pragma omp parallel
    {
      const int tid = omp_get_thread_num();
#pragma omp for
      for (auto i : vec)
        benchmark::DoNotOptimize(thread_min_vals[tid] =
                                     std::max(thread_min_vals[tid], i));
    }
    for (auto i : thread_min_vals)
      benchmark::DoNotOptimize(min_val = std::max(min_val, i));
  }
  state.SetBytesProcessed(int64_t(state.iterations()) *
                          int64_t(state.range(0)) * 4);
}

BENCHMARK(OMP_ARRAY_Reduce)->RangeMultiplier(8)->Range(1 << 10, 1 << 24);

static void OMP_ARRAY_OPT_Reduce(benchmark::State &state) {
  omp_set_num_threads(threads);
  // First create an instance of an engine.
  std::random_device rnd_device;
  // Specify the engine and distribution.
  std::mt19937 mersenne_engine{rnd_device()}; // Generates random integers
  std::uniform_int_distribution<int> dist{0, std::numeric_limits<int>::max()};

  auto gen = [&]() { return dist(mersenne_engine); };
  const size_t size = state.range(0);

  std::vector<int> vec(size);
  std::generate(vec.begin(), vec.end(), gen);
  int min_val{std::numeric_limits<int>::min()};
  for (auto _ : state) {
    std::vector<int> thread_min_vals(threads, std::numeric_limits<int>::min());
#pragma omp parallel
    {
      const int tid = omp_get_thread_num();
#pragma omp for
      for (auto i : vec)
        thread_min_vals[tid] = std::max(thread_min_vals[tid], i);
    }
    for (auto i : thread_min_vals)
      min_val = std::max(min_val, i);
  }
  state.SetBytesProcessed(int64_t(state.iterations()) *
                          int64_t(state.range(0)) * 4);
}

BENCHMARK(OMP_ARRAY_OPT_Reduce)->RangeMultiplier(8)->Range(1 << 10, 1 << 24);

// GOOD: Padding to prevent false sharing
// Each thread writes to thread_min_vals[tid * 16], spacing data by 64 bytes
// This ensures each thread's data is on a separate cache line
// Expected: Significantly faster than OMP_ARRAY_Reduce
static void OMP_ARRAY_ALIGN_Reduce(benchmark::State &state) {
  omp_set_num_threads(threads);
  // First create an instance of an engine.
  std::random_device rnd_device;
  // Specify the engine and distribution.
  std::mt19937 mersenne_engine{rnd_device()}; // Generates random integers
  std::uniform_int_distribution<int> dist{0, std::numeric_limits<int>::max()};

  auto gen = [&]() { return dist(mersenne_engine); };
  const size_t size = state.range(0);

  std::vector<int> vec(size);
  std::generate(vec.begin(), vec.end(), gen);
  int min_val{std::numeric_limits<int>::min()};
  for (auto _ : state) {
    alignas(64) int thread_min_vals[threads * 16];
    for (int i = 0; i < threads; i++)
      thread_min_vals[i * 16] = std::numeric_limits<int>::min();
#pragma omp parallel
    {
      const int tid = omp_get_thread_num();
#pragma omp for
      for (auto i : vec)
        benchmark::DoNotOptimize(thread_min_vals[tid * 16] =
                                     std::max(thread_min_vals[tid * 16], i));
    }
    for (int i = 0; i < threads; i++)
      benchmark::DoNotOptimize(min_val =
                                   std::max(min_val, thread_min_vals[i * 16]));
  }
  state.SetBytesProcessed(int64_t(state.iterations()) *
                          int64_t(state.range(0)) * 4);
}

BENCHMARK(OMP_ARRAY_ALIGN_Reduce)->RangeMultiplier(8)->Range(1 << 10, 1 << 24);

// BEST: Local variable per thread
// Each thread uses a stack-local variable during computation
// Only writes to shared array once after the loop completes
// Expected: Best performance - no cache line sharing during hot loop
static void OMP_LOCAL_VAL_Reduce(benchmark::State &state) {
  omp_set_num_threads(threads);
  // First create an instance of an engine.
  std::random_device rnd_device;
  // Specify the engine and distribution.
  std::mt19937 mersenne_engine{rnd_device()}; // Generates random integers
  std::uniform_int_distribution<int> dist{0, std::numeric_limits<int>::max()};

  auto gen = [&]() { return dist(mersenne_engine); };
  const size_t size = state.range(0);

  std::vector<int> vec(size);
  std::generate(vec.begin(), vec.end(), gen);
  int min_val{std::numeric_limits<int>::min()};
  for (auto _ : state) {
    std::vector<int> thread_min_vals(threads);
#pragma omp parallel
    {
      int local_min_val{std::numeric_limits<int>::min()};
#pragma omp for
      for (auto i : vec)
        local_min_val = std::max(local_min_val, i);
      benchmark::DoNotOptimize(thread_min_vals[omp_get_thread_num()] =
                                   local_min_val);
    }
    for (auto i : thread_min_vals)
      benchmark::DoNotOptimize(min_val = std::max(min_val, i));
  }
  state.SetBytesProcessed(int64_t(state.iterations()) *
                          int64_t(state.range(0)) * 4);
}

BENCHMARK(OMP_LOCAL_VAL_Reduce)->RangeMultiplier(8)->Range(1 << 10, 1 << 24);

// ============================================================================
// Int Array vs Bool Array False Sharing Benchmark
// ============================================================================
//
// This benchmark compares false sharing behavior between int and bool arrays.
// Multiple threads randomly flip values (0->1 or 1->0) at positions specified
// by a position array. This demonstrates:
// 1. Bool arrays pack 8 bits per byte, causing extreme false sharing
// 2. Int arrays use 4 bytes each, reducing but not eliminating false sharing
// 3. Aligned int arrays (with padding) avoid false sharing completely

// Random flip benchmark using int array (no alignment)
// FALSE SHARING: Adjacent ints may share cache lines (16 ints per 64-byte line)
static void BM_RandomFlip_Int(benchmark::State &state) {
  omp_set_num_threads(threads);
  const size_t array_size = state.range(0);
  const size_t num_flips = 1000000; // Number of flips per iteration

  // Initialize arrays
  std::vector<int> data(array_size, 0);
  std::vector<size_t> positions(num_flips);

  // Generate random positions with fixed seed for reproducibility
  std::mt19937 gen(12345); // Fixed seed for fair comparison
  std::uniform_int_distribution<size_t> dis(0, array_size - 1);
  for (auto &pos : positions) {
    pos = dis(gen);
  }

  for (auto _ : state) {
#pragma omp parallel for schedule(static)
    for (size_t i = 0; i < num_flips; ++i) {
      size_t pos = positions[i];
      // Atomic flip: 0->1 or 1->0
      #pragma omp atomic
      data[pos] ^= 1;
    }
  }

  state.SetItemsProcessed(state.iterations() * num_flips);
  state.SetLabel("int_array");
}

BENCHMARK(BM_RandomFlip_Int)->Arg(64)->Arg(256)->Arg(1024)->Arg(4096);

// Random flip benchmark using bool array
// SEVERE FALSE SHARING: Bools pack tightly, many per cache line
// Up to 512 bools (64 bytes) share the same cache line!
static void BM_RandomFlip_Bool(benchmark::State &state) {
  omp_set_num_threads(threads);
  const size_t array_size = state.range(0);
  const size_t num_flips = 1000000;

  std::vector<bool> data(array_size, false);
  std::vector<size_t> positions(num_flips);

  // Generate random positions with fixed seed for reproducibility
  std::mt19937 gen(random_seed);
  std::uniform_int_distribution<size_t> dis(0, array_size - 1);
  for (auto &pos : positions) {
    pos = dis(gen);
  }

  for (auto _ : state) {
#pragma omp parallel for schedule(static)
    for (size_t i = 0; i < num_flips; ++i) {
      size_t pos = positions[i];
      // Bool flip - causes severe false sharing
      // Note: std::vector<bool> doesn't support atomic operations well
      #pragma omp critical
      data[pos] = !data[pos];
    }
  }

  state.SetItemsProcessed(state.iterations() * num_flips);
  state.SetLabel("bool_array");
}

BENCHMARK(BM_RandomFlip_Bool)->Arg(64)->Arg(256)->Arg(1024)->Arg(4096);

// Random flip benchmark using aligned int array (with padding)
// NO FALSE SHARING: Each int is padded to cache line size (64 bytes)
// This should be significantly faster than the non-aligned version
static void BM_RandomFlip_Int_Aligned(benchmark::State &state) {
  omp_set_num_threads(threads);
  const size_t array_size = state.range(0);
  const size_t num_flips = 1000000;

  // Align each int to 64 bytes (16 ints spacing) to avoid false sharing
  constexpr size_t cache_line_ints = 64 / sizeof(int); // 16 ints per cache line
  std::vector<int> data(array_size * cache_line_ints, 0);
  std::vector<size_t> positions(num_flips);

  // Generate random positions with fixed seed for reproducibility
  std::mt19937 gen(random_seed);
  std::uniform_int_distribution<size_t> dis(0, array_size - 1);
  for (auto &pos : positions) {
    pos = dis(gen);
  }

  for (auto _ : state) {
#pragma omp parallel for schedule(static)
    for (size_t i = 0; i < num_flips; ++i) {
      size_t pos = positions[i];
      // Access with stride to ensure different cache lines
      size_t aligned_pos = pos * cache_line_ints;
      #pragma omp atomic
      data[aligned_pos] ^= 1;
    }
  }

  state.SetItemsProcessed(state.iterations() * num_flips);
  state.SetLabel("int_aligned");
}

BENCHMARK(BM_RandomFlip_Int_Aligned)->Arg(64)->Arg(256)->Arg(1024)->Arg(4096);

// ============================================================================
// Boolean Array Benchmarks with Different Alignments
// ============================================================================
//
// Compare different byte alignments for boolean values:
// - 1 bit (std::vector<bool>): Extreme false sharing, 512 bools per cache line
// - 1 byte (uint8_t): 64 values per cache line
// - 4 bytes (uint32_t): 16 values per cache line
// - 8 bytes (uint64_t): 8 values per cache line
// - 16 bytes: 4 values per cache line
// - 32 bytes: 2 values per cache line
// - 64 bytes: 1 value per cache line (no false sharing)

// Helper template for aligned boolean storage
// Uses uint64_t for consistent atomic performance across all alignments
template<size_t Alignment>
struct alignas(Alignment) AlignedBool {
  uint64_t value;
  AlignedBool() : value(0) {}
};

// 1 byte alignment (uint8_t as bool)
static void BM_RandomFlip_Bool_1Byte(benchmark::State &state) {
  omp_set_num_threads(threads);
  const size_t array_size = state.range(0);
  const size_t num_flips = 1000000;

  std::vector<uint8_t> data(array_size, 0);
  std::vector<size_t> positions(num_flips);

  std::mt19937 gen(random_seed);
  std::uniform_int_distribution<size_t> dis(0, array_size - 1);
  for (auto &pos : positions) {
    pos = dis(gen);
  }

  for (auto _ : state) {
#pragma omp parallel for schedule(static)
    for (size_t i = 0; i < num_flips; ++i) {
      size_t pos = positions[i];
      #pragma omp atomic
      data[pos] ^= 1;
    }
  }

  state.SetItemsProcessed(state.iterations() * num_flips);
  state.SetLabel("1byte_align");
}

BENCHMARK(BM_RandomFlip_Bool_1Byte)->Arg(64)->Arg(256)->Arg(1024)->Arg(4096);

// 4 byte alignment (uint32_t as bool)
static void BM_RandomFlip_Bool_4Byte(benchmark::State &state) {
  omp_set_num_threads(threads);
  const size_t array_size = state.range(0);
  const size_t num_flips = 1000000;

  std::vector<uint32_t> data(array_size, 0);
  std::vector<size_t> positions(num_flips);

  std::mt19937 gen(random_seed);
  std::uniform_int_distribution<size_t> dis(0, array_size - 1);
  for (auto &pos : positions) {
    pos = dis(gen);
  }

  for (auto _ : state) {
#pragma omp parallel for schedule(static)
    for (size_t i = 0; i < num_flips; ++i) {
      size_t pos = positions[i];
      #pragma omp atomic
      data[pos] ^= 1;
    }
  }

  state.SetItemsProcessed(state.iterations() * num_flips);
  state.SetLabel("4byte_align");
}

BENCHMARK(BM_RandomFlip_Bool_4Byte)->Arg(64)->Arg(256)->Arg(1024)->Arg(4096);

// 8 byte alignment (uint64_t as bool)
static void BM_RandomFlip_Bool_8Byte(benchmark::State &state) {
  omp_set_num_threads(threads);
  const size_t array_size = state.range(0);
  const size_t num_flips = 1000000;

  std::vector<uint64_t> data(array_size, 0);
  std::vector<size_t> positions(num_flips);

  std::mt19937 gen(random_seed);
  std::uniform_int_distribution<size_t> dis(0, array_size - 1);
  for (auto &pos : positions) {
    pos = dis(gen);
  }

  for (auto _ : state) {
#pragma omp parallel for schedule(static)
    for (size_t i = 0; i < num_flips; ++i) {
      size_t pos = positions[i];
      #pragma omp atomic
      data[pos] ^= 1;
    }
  }

  state.SetItemsProcessed(state.iterations() * num_flips);
  state.SetLabel("8byte_align");
}

BENCHMARK(BM_RandomFlip_Bool_8Byte)->Arg(64)->Arg(256)->Arg(1024)->Arg(4096);

// 16 byte alignment
static void BM_RandomFlip_Bool_16Byte(benchmark::State &state) {
  omp_set_num_threads(threads);
  const size_t array_size = state.range(0);
  const size_t num_flips = 1000000;

  std::vector<AlignedBool<16>> data(array_size);
  std::vector<size_t> positions(num_flips);

  std::mt19937 gen(random_seed);
  std::uniform_int_distribution<size_t> dis(0, array_size - 1);
  for (auto &pos : positions) {
    pos = dis(gen);
  }

  for (auto _ : state) {
#pragma omp parallel for schedule(static)
    for (size_t i = 0; i < num_flips; ++i) {
      size_t pos = positions[i];
      #pragma omp atomic
      data[pos].value ^= 1;
    }
  }

  state.SetItemsProcessed(state.iterations() * num_flips);
  state.SetLabel("16byte_align");
}

BENCHMARK(BM_RandomFlip_Bool_16Byte)->Arg(64)->Arg(256)->Arg(1024)->Arg(4096);

// 32 byte alignment
static void BM_RandomFlip_Bool_32Byte(benchmark::State &state) {
  omp_set_num_threads(threads);
  const size_t array_size = state.range(0);
  const size_t num_flips = 1000000;

  std::vector<AlignedBool<32>> data(array_size);
  std::vector<size_t> positions(num_flips);

  std::mt19937 gen(random_seed);
  std::uniform_int_distribution<size_t> dis(0, array_size - 1);
  for (auto &pos : positions) {
    pos = dis(gen);
  }

  for (auto _ : state) {
#pragma omp parallel for schedule(static)
    for (size_t i = 0; i < num_flips; ++i) {
      size_t pos = positions[i];
      #pragma omp atomic
      data[pos].value ^= 1;
    }
  }

  state.SetItemsProcessed(state.iterations() * num_flips);
  state.SetLabel("32byte_align");
}

BENCHMARK(BM_RandomFlip_Bool_32Byte)->Arg(64)->Arg(256)->Arg(1024)->Arg(4096);

// 64 byte alignment (cache line aligned - NO FALSE SHARING)
static void BM_RandomFlip_Bool_64Byte(benchmark::State &state) {
  omp_set_num_threads(threads);
  const size_t array_size = state.range(0);
  const size_t num_flips = 1000000;

  std::vector<AlignedBool<64>> data(array_size);
  std::vector<size_t> positions(num_flips);

  std::mt19937 gen(random_seed);
  std::uniform_int_distribution<size_t> dis(0, array_size - 1);
  for (auto &pos : positions) {
    pos = dis(gen);
  }

  for (auto _ : state) {
#pragma omp parallel for schedule(static)
    for (size_t i = 0; i < num_flips; ++i) {
      size_t pos = positions[i];
      #pragma omp atomic
      data[pos].value ^= 1;
    }
  }

  state.SetItemsProcessed(state.iterations() * num_flips);
  state.SetLabel("64byte_align");
}

BENCHMARK(BM_RandomFlip_Bool_64Byte)->Arg(64)->Arg(256)->Arg(1024)->Arg(4096);

// 128 byte alignment (2x cache line aligned)
static void BM_RandomFlip_Bool_128Byte(benchmark::State &state) {
  omp_set_num_threads(threads);
  const size_t array_size = state.range(0);
  const size_t num_flips = 1000000;

  std::vector<AlignedBool<128>> data(array_size);
  std::vector<size_t> positions(num_flips);

  std::mt19937 gen(random_seed);
  std::uniform_int_distribution<size_t> dis(0, array_size - 1);
  for (auto &pos : positions) {
    pos = dis(gen);
  }

  for (auto _ : state) {
#pragma omp parallel for schedule(static)
    for (size_t i = 0; i < num_flips; ++i) {
      size_t pos = positions[i];
      #pragma omp atomic
      data[pos].value ^= 1;
    }
  }

  state.SetItemsProcessed(state.iterations() * num_flips);
  state.SetLabel("128byte_align");
}

BENCHMARK(BM_RandomFlip_Bool_128Byte)->Arg(64)->Arg(256)->Arg(1024)->Arg(4096);

// 256 byte alignment (4x cache line aligned)
static void BM_RandomFlip_Bool_256Byte(benchmark::State &state) {
  omp_set_num_threads(threads);
  const size_t array_size = state.range(0);
  const size_t num_flips = 1000000;

  std::vector<AlignedBool<256>> data(array_size);
  std::vector<size_t> positions(num_flips);

  std::mt19937 gen(random_seed);
  std::uniform_int_distribution<size_t> dis(0, array_size - 1);
  for (auto &pos : positions) {
    pos = dis(gen);
  }

  for (auto _ : state) {
#pragma omp parallel for schedule(static)
    for (size_t i = 0; i < num_flips; ++i) {
      size_t pos = positions[i];
      #pragma omp atomic
      data[pos].value ^= 1;
    }
  }

  state.SetItemsProcessed(state.iterations() * num_flips);
  state.SetLabel("256byte_align");
}

BENCHMARK(BM_RandomFlip_Bool_256Byte)->Arg(64)->Arg(256)->Arg(1024)->Arg(4096);

// 512 byte alignment (8x cache line aligned)
static void BM_RandomFlip_Bool_512Byte(benchmark::State &state) {
  omp_set_num_threads(threads);
  const size_t array_size = state.range(0);
  const size_t num_flips = 1000000;

  std::vector<AlignedBool<512>> data(array_size);
  std::vector<size_t> positions(num_flips);

  std::mt19937 gen(random_seed);
  std::uniform_int_distribution<size_t> dis(0, array_size - 1);
  for (auto &pos : positions) {
    pos = dis(gen);
  }

  for (auto _ : state) {
#pragma omp parallel for schedule(static)
    for (size_t i = 0; i < num_flips; ++i) {
      size_t pos = positions[i];
      #pragma omp atomic
      data[pos].value ^= 1;
    }
  }

  state.SetItemsProcessed(state.iterations() * num_flips);
  state.SetLabel("512byte_align");
}

BENCHMARK(BM_RandomFlip_Bool_512Byte)->Arg(64)->Arg(256)->Arg(1024)->Arg(4096);

BENCHMARK_MAIN();
