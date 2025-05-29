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
#include <cmath>

static void Serial_Permute(benchmark::State &state)
{
  // First create an instance of an engine.
  std::random_device rnd_device;
  // Specify the engine and distribution.
  std::mt19937 mersenne_engine{rnd_device()}; // Generates random integers
  std::uniform_real_distribution<double> dist{0, 1};
  auto gen = [&]()
  { return dist(mersenne_engine); };
  const size_t size = state.range(0);
  std::vector<int> perm(size);
  for (size_t i = 0; i < size; ++i)
  {
    perm[i] = i;
  }
  std::shuffle(perm.begin(), perm.end(), mersenne_engine);

  std::vector<double> from(size);
  std::vector<double> to(size);
  std::generate(from.begin(), from.end(), gen);

  for (auto _ : state)
  {
    for (size_t i = 0; i < size; i++)
      to[perm[i]] = from[i];
    swap(from, to);
  }
  state.SetBytesProcessed(int64_t(state.iterations()) *
                          int64_t(state.range(0)) * 8);
}

BENCHMARK(Serial_Permute)->RangeMultiplier(4)->Range(1 << 4, 1 << 30);

static const int threads = 8;

static void Parallel_Permute(benchmark::State &state)
{
  omp_set_num_threads(threads);
  // First create an instance of an engine.
  std::random_device rnd_device;
  // Specify the engine and distribution.
  std::mt19937 mersenne_engine{rnd_device()}; // Generates random integers
  std::uniform_real_distribution<double> dist{0, 1};
  auto gen = [&]()
  { return dist(mersenne_engine); };
  const size_t size = state.range(0);
  std::vector<int> perm(size);
  for (size_t i = 0; i < size; ++i)
  {
    perm[i] = i;
  }
  std::shuffle(perm.begin(), perm.end(), mersenne_engine);

  std::vector<double> from(size);
  std::vector<double> to(size);
  std::generate(from.begin(), from.end(), gen);

  for (auto _ : state)
  {
#pragma omp parallel for
    for (size_t i = 0; i < size; i++)
      to[perm[i]] = from[i];
    swap(from, to);
  }
  state.SetBytesProcessed(int64_t(state.iterations()) *
                          int64_t(state.range(0)) * 8);
}

BENCHMARK(Parallel_Permute)->RangeMultiplier(4)->Range(1 << 4, 1 << 30);

BENCHMARK_MAIN();