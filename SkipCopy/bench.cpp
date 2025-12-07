#include <algorithm>
#include <benchmark/benchmark.h>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <numeric>
#include <string>
#include <vector>

namespace {

constexpr int kMinSize = 8;
constexpr int kMaxSize = 1 << 16;

// Materialize a sorted sequence of ints we can reuse between iterations.
std::vector<int> make_source(int size) {
  std::vector<int> src(size);
  std::iota(src.begin(), src.end(), 0);
  return src;
}

static void RunLinearSkip(benchmark::State &state, int size, int skip_index) {
  const std::vector<int> src = make_source(size);
  std::vector<int> dst(static_cast<std::size_t>(size - 1));
  const int skip_value = src[skip_index];

  for (auto _ : state) {
    int *out = dst.data();
    for (int value : src) {
      if (value != skip_value) {
        *out++ = value;
      }
    }
    benchmark::DoNotOptimize(dst);
  }

  state.SetLabel("skip=" + std::to_string(skip_index));
  state.SetBytesProcessed(static_cast<int64_t>(state.iterations()) *
                          static_cast<int64_t>((size - 1) *
                                               static_cast<int>(sizeof(int))));
}

static void RunBinarySkip(benchmark::State &state, int size, int skip_index) {
  const std::vector<int> src = make_source(size);
  std::vector<int> dst(static_cast<std::size_t>(size - 1));
  const int skip_value = src[skip_index];

  for (auto _ : state) {
    const auto it = std::lower_bound(src.begin(), src.end(), skip_value);
    const std::size_t prefix = static_cast<std::size_t>(it - src.begin());

    std::copy_n(src.data(), prefix, dst.data());
    std::copy_n(src.data() + prefix + 1, src.size() - prefix - 1,
                dst.data() + static_cast<std::ptrdiff_t>(prefix));

    benchmark::DoNotOptimize(dst);
    benchmark::ClobberMemory();
  }

  state.SetLabel("skip=" + std::to_string(skip_index));
  state.SetBytesProcessed(static_cast<int64_t>(state.iterations()) *
                          static_cast<int64_t>((size - 1) *
                                               static_cast<int>(sizeof(int))));
}

void RegisterBenchmarks() {
  constexpr double fractions[] = {0.0, 0.25, 0.5, 0.75, 1.0};
  constexpr const char *labels[] = {"first", "25pct", "50pct", "75pct", "last"};

  std::vector<int> sizes;
  for (int size = kMinSize; size <= kMaxSize; size <<= 1) {
    sizes.push_back(size);
  }
  sizes.push_back(24);
  std::sort(sizes.begin(), sizes.end());
  sizes.erase(std::unique(sizes.begin(), sizes.end()), sizes.end());

  for (int size : sizes) {
    for (std::size_t i = 0; i < std::size(labels); ++i) {
      const int skip_index =
          static_cast<int>(static_cast<double>(size - 1) * fractions[i]);
      const std::string suffix =
          "/size=" + std::to_string(size) + "/" + labels[i];

      benchmark::RegisterBenchmark(
          ("LinearSkip" + suffix).c_str(),
          [size, skip_index](benchmark::State &state) {
            RunLinearSkip(state, size, skip_index);
          })
          ->UseRealTime();

      benchmark::RegisterBenchmark(
          ("BinarySkip" + suffix).c_str(),
          [size, skip_index](benchmark::State &state) {
            RunBinarySkip(state, size, skip_index);
          })
          ->UseRealTime();
    }
  }
}

} // namespace

BENCHMARK_MAIN();

// Ensure benchmarks are registered before main executes.
static bool kRegistered = (RegisterBenchmarks(), true);
