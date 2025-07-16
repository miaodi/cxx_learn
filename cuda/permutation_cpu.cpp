#include <algorithm>
#include <atomic>
#include <benchmark/benchmark.h>
#include <iostream>
#include <random>
#include <permute.hpp>

static void Serial_Permute(benchmark::State &state)
{
  size_t size = state.range(0);
  std::vector<int> perm(size);
  permutation_generate(perm.data(), size);

  std::vector<double> from(size);
  std::vector<double> to(size);
  random_vector_generate(from.data(), size);

  for (auto _ : state)
  {
    permute_cpu<double, false>(from.data(), to.data(), perm.data(), size);
    swap(from, to);
  }
  state.SetBytesProcessed(int64_t(state.iterations()) *
                          int64_t(state.range(0)) * 8);
}

BENCHMARK(Serial_Permute)->RangeMultiplier(2)->Range(1 << 8, 1 << 28);

static const int threads = 8;

static void Parallel_Permute(benchmark::State &state)
{
  omp_set_num_threads(threads);
  size_t size = state.range(0);
  std::vector<int> perm(size);
  permutation_generate(perm.data(), size);

  std::vector<double> from(size);
  std::vector<double> to(size);
  random_vector_generate(from.data(), size);

  for (auto _ : state)
  {
    permute_cpu<double, false, threads>(from.data(), to.data(), perm.data(), size);
    swap(from, to);
  }
  state.SetBytesProcessed(int64_t(state.iterations()) *
                          int64_t(state.range(0)) * 8);
}

BENCHMARK(Parallel_Permute)->RangeMultiplier(2)->Range(1 << 8, 1 << 28);

BENCHMARK_MAIN();