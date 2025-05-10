#include <algorithm>
#include <benchmark/benchmark.h>
#include <vector>
#include <random>

static const int threads = 8;

__attribute__((optimize(0)))
static void SwapUnOptimized(benchmark::State &state) {
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
    for (size_t i = 0; i < vec.size() - 1; i++){
      // Swap the elements
      int tmp = vec[i];
      vec[i] = vec[i + 1];
      vec[i + 1] = tmp;
      // benchmark::DoNotOptimize(vec[i]);
    }
  }
  state.SetBytesProcessed(int64_t(state.iterations()) *
                          int64_t(state.range(0)) * 4);
}
BENCHMARK(SwapUnOptimized)->RangeMultiplier(8)->Range(1 << 10, 1 << 24);

__attribute__((optimize(2)))
static void SwapOptimized(benchmark::State &state) {
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
    for (size_t i = 0; i < vec.size() - 1; i++){
      // Swap the elements
      int tmp = vec[i];
      vec[i] = vec[i + 1];
      vec[i + 1] = tmp;
      // benchmark::DoNotOptimize(vec[i]);
    }
  }
  state.SetBytesProcessed(int64_t(state.iterations()) *
                          int64_t(state.range(0)) * 4);
}

BENCHMARK(SwapOptimized)->RangeMultiplier(8)->Range(1 << 10, 1 << 24);

BENCHMARK_MAIN();