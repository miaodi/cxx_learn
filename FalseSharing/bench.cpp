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

BENCHMARK_MAIN();