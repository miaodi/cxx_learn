#include <benchmark/benchmark.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <numeric>
#include <random>
#include <string>
#include <utility>
#include <vector>

namespace {

using Value = std::uint64_t;

constexpr std::uint64_t kSeed = 0x9e3779b97f4a7c15ULL;

struct Input {
  std::vector<Value> values;
  std::vector<std::size_t> perm;
};

std::uint64_t splitmix64(std::uint64_t x) {
  x += 0x9e3779b97f4a7c15ULL;
  x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
  x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
  return x ^ (x >> 31);
}

std::string format_bytes(std::size_t bytes) {
  if (bytes >= 1024 * 1024) {
    return std::to_string(bytes / (1024 * 1024)) + "MiB";
  }
  if (bytes >= 1024) {
    return std::to_string(bytes / 1024) + "KiB";
  }
  return std::to_string(bytes) + "B";
}

Input make_input(std::size_t array_bytes) {
  const std::size_t n = std::max<std::size_t>(1, array_bytes / sizeof(Value));

  Input input;
  input.values.resize(n);
  for (std::size_t i = 0; i < n; ++i) {
    input.values[i] = splitmix64(i + kSeed);
  }

  input.perm.resize(n);
  std::iota(input.perm.begin(), input.perm.end(), std::size_t{0});

  // The seed depends only on n, not the accumulator count. Every benchmark
  // variant for the same array size uses the same random permutation.
  std::mt19937_64 gen(kSeed ^ static_cast<std::uint64_t>(n));
  std::shuffle(input.perm.begin(), input.perm.end(), gen);

  return input;
}

template <std::size_t... Is, std::size_t NumAcc>
inline void accumulate_group(const Value* values, const std::size_t* perm,
                             std::size_t i, std::array<Value, NumAcc>& sums,
                             std::index_sequence<Is...>) {
  ((sums[Is] += values[perm[i + Is]]), ...);
}

template <std::size_t NumAcc>
Value random_sum(const Value* values, const std::size_t* perm, std::size_t n) {
  static_assert(NumAcc > 0);

  std::array<Value, NumAcc> sums{};
  std::size_t i = 0;

  for (; i + NumAcc <= n; i += NumAcc) {
    accumulate_group(values, perm, i, sums, std::make_index_sequence<NumAcc>{});
  }

  Value total = 0;
  for (Value sum : sums) {
    total += sum;
  }

  for (; i < n; ++i) {
    total += values[perm[i]];
  }

  return total;
}

template <std::size_t NumAcc>
static void BM_RandomAccessSumAccumulators(benchmark::State& state) {
  const auto array_bytes = static_cast<std::size_t>(state.range(0));
  const Input input = make_input(array_bytes);
  const Value* values = input.values.data();
  const std::size_t* perm = input.perm.data();
  const std::size_t n = input.perm.size();

  for (auto _ : state) {
    benchmark::ClobberMemory();
    Value sum = random_sum<NumAcc>(values, perm, n);
    benchmark::DoNotOptimize(sum);
  }

  const auto iterations = static_cast<std::int64_t>(state.iterations());
  const auto elements = static_cast<std::int64_t>(n);
  const auto bytes_per_iteration =
      static_cast<std::int64_t>(n * (sizeof(Value) + sizeof(std::size_t)));

  state.SetItemsProcessed(iterations * elements);
  state.SetBytesProcessed(iterations * bytes_per_iteration);
  state.SetLabel("array=" + format_bytes(n * sizeof(Value)));
  state.counters["accumulators"] = static_cast<double>(NumAcc);
}

void array_sizes(benchmark::internal::Benchmark* b) {
  b->Arg(4 * 1024)
      ->Arg(32 * 1024)
      ->Arg(256 * 1024)
      ->Arg(2 * 1024 * 1024)
      ->Arg(16 * 1024 * 1024)
      ->Arg(64 * 1024 * 1024)
      ->ArgName("array_bytes")
      ->Unit(benchmark::kNanosecond);
}

BENCHMARK_TEMPLATE(BM_RandomAccessSumAccumulators, 1)->Apply(array_sizes);
BENCHMARK_TEMPLATE(BM_RandomAccessSumAccumulators, 2)->Apply(array_sizes);
BENCHMARK_TEMPLATE(BM_RandomAccessSumAccumulators, 4)->Apply(array_sizes);
BENCHMARK_TEMPLATE(BM_RandomAccessSumAccumulators, 8)->Apply(array_sizes);
BENCHMARK_TEMPLATE(BM_RandomAccessSumAccumulators, 16)->Apply(array_sizes);

}  // namespace

BENCHMARK_MAIN();
