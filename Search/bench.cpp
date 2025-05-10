#include <algorithm>
#include <benchmark/benchmark.h>
#include <random>
#include <vector>

static void LinearSearch_int(benchmark::State &state) {

  // First create an instance of an engine.
  std::random_device rnd_device;
  // Specify the engine and distribution.
  std::mt19937 mersenne_engine{rnd_device()}; // Generates random integers
  std::uniform_int_distribution<int> dist{std::numeric_limits<int>::min(),
                                          std::numeric_limits<int>::max()};
  auto gen = [&]() { return dist(mersenne_engine); };
  const auto size = state.range(0);
  std::vector<int> vec(size);
  std::generate(vec.begin(), vec.end(), gen);
  std::sort(vec.begin(), vec.end());
  const int last = vec[size - 1];

  size_t pos;

  for (auto _ : state) {
    for (size_t i = 0; i < vec.size(); i++) {
      if (vec[i] == last) {
        benchmark::DoNotOptimize(pos = i);
        break;
      }
    }
  }
}

BENCHMARK(LinearSearch_int)->RangeMultiplier(2)->Range(1 << 1, 1 << 8);

static void BinarySearch_int(benchmark::State &state) {

  // First create an instance of an engine.
  std::random_device rnd_device;
  // Specify the engine and distribution.
  std::mt19937 mersenne_engine{rnd_device()}; // Generates random integers
  std::uniform_int_distribution<int> dist{std::numeric_limits<int>::min(),
                                          std::numeric_limits<int>::max()};
  auto gen = [&]() { return dist(mersenne_engine); };
  const auto size = state.range(0);
  std::vector<int> vec(size);
  std::generate(vec.begin(), vec.end(), gen);
  std::sort(vec.begin(), vec.end());
  const int last = vec[size - 1];

  size_t pos;

  for (auto _ : state) {
    benchmark::DoNotOptimize(std::binary_search(vec.begin(), vec.end(), last));
  }
}

BENCHMARK(BinarySearch_int)->RangeMultiplier(2)->Range(1 << 1, 1 << 8);

BENCHMARK_MAIN();