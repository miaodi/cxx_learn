#include <benchmark/benchmark.h>

#include <algorithm>
#include <array>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <type_traits>
#include <vector>

namespace {
// Simple custom type that remains trivially copyable so memcpy is well-defined.
struct CustomPod {
  std::uint64_t id;
  double weight;
  bool active;
  int8_t priority;
  char label[24];
};

using SmallArray = std::array<int, 4>;
using LargeArray = std::array<std::uint64_t, 8>;

template <typename T>
T make_value(std::size_t i) {
  if constexpr (std::is_arithmetic_v<T>) {
    return static_cast<T>(i);
  } else {
    static_assert(std::is_same_v<T, void>,
                  "make_value needs a specialization for this type");
  }
}

template <>
SmallArray make_value<SmallArray>(std::size_t i) {
  return SmallArray{static_cast<int>(i), static_cast<int>(i + 1),
                    static_cast<int>(i + 2), static_cast<int>(i + 3)};
}

template <>
LargeArray make_value<LargeArray>(std::size_t i) {
  return LargeArray{static_cast<std::uint64_t>(i),
                    static_cast<std::uint64_t>(i + 1),
                    static_cast<std::uint64_t>(i + 2),
                    static_cast<std::uint64_t>(i + 3),
                    static_cast<std::uint64_t>(i + 4),
                    static_cast<std::uint64_t>(i + 5),
                    static_cast<std::uint64_t>(i + 6),
                    static_cast<std::uint64_t>(i + 7)};
}

template <>
CustomPod make_value<CustomPod>(std::size_t i) {
  CustomPod pod{};
  pod.id = static_cast<std::uint64_t>(i);
  pod.weight = 0.25 * static_cast<double>(i);
  std::snprintf(pod.label, sizeof(pod.label), "item-%zu", i);
  return pod;
}

template <typename T>
void fill_source(std::vector<T> &src) {
  for (std::size_t i = 0; i < src.size(); ++i) {
    src[i] = make_value<T>(i);
  }
}

template <typename T>
static void BM_StdCopy(benchmark::State &state) {
  const std::size_t count = static_cast<std::size_t>(state.range(0));
  std::vector<T> src(count);
  std::vector<T> dst(count);
  fill_source(src);

  for (auto _ : state) {
    std::copy(src.begin(), src.end(), dst.begin());
    benchmark::DoNotOptimize(dst);
  }

  state.SetBytesProcessed(static_cast<int64_t>(state.iterations()) *
                          static_cast<int64_t>(count * sizeof(T)));
}

template <typename T>
static void BM_StdMemcpy(benchmark::State &state) {
  static_assert(std::is_trivially_copyable_v<T>,
                "memcpy requires trivially copyable types");

  const std::size_t count = static_cast<std::size_t>(state.range(0));
  std::vector<T> src(count);
  std::vector<T> dst(count);
  fill_source(src);

  T *dst_ptr = dst.data();
  const T *src_ptr = src.data();
  const std::size_t bytes = count * sizeof(T);

  for (auto _ : state) {
    std::memcpy(dst_ptr, src_ptr, bytes);
    benchmark::DoNotOptimize(dst_ptr);
    benchmark::ClobberMemory();
  }

  state.SetBytesProcessed(static_cast<int64_t>(state.iterations()) *
                          static_cast<int64_t>(bytes));
}

constexpr int kMinElements = 1 << 6;
constexpr int kMaxElements = 1 << 16;

#define REGISTER_COPY_BENCHMARK(Type)                                         \
  BENCHMARK_TEMPLATE(BM_StdCopy, Type)                                        \
      ->RangeMultiplier(4)                                                    \
      ->Range(kMinElements, kMaxElements)                                     \
      ->UseRealTime();                                                        \
  BENCHMARK_TEMPLATE(BM_StdMemcpy, Type)                                      \
      ->RangeMultiplier(4)                                                    \
      ->Range(kMinElements, kMaxElements)                                     \
      ->UseRealTime()

REGISTER_COPY_BENCHMARK(int);
REGISTER_COPY_BENCHMARK(double);
REGISTER_COPY_BENCHMARK(SmallArray);
REGISTER_COPY_BENCHMARK(LargeArray);
REGISTER_COPY_BENCHMARK(CustomPod);

#undef REGISTER_COPY_BENCHMARK

} // namespace

BENCHMARK_MAIN();
