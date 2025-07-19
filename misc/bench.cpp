#include "func.h"
#include <benchmark/benchmark.h>
#include <functional>
#include <map>
#include <vector>

static void BM_TwiddleOrigin(benchmark::State &state) {
  std::uint64_t u = 123456789, v = 987654321;
  std::uint64_t result = 0;
  for (auto _ : state) {
    for (size_t i = 0; i < state.range(0); ++i) {
      result ^= MCRand::twiddle_origin(u, v);
      ++u;
      ++v;
      benchmark::DoNotOptimize(result);
    }
  }
}

static void BM_TwiddleNew(benchmark::State &state) {
  std::uint64_t u = 123456789, v = 987654321;
  std::uint64_t result = 0;
  for (auto _ : state) {
    for (size_t i = 0; i < state.range(0); ++i) {
      result ^= MCRand::twiddle_new(u, v);
      ++u;
      ++v;
      benchmark::DoNotOptimize(result);
    }
  }
}

BENCHMARK(BM_TwiddleOrigin)->RangeMultiplier(16)->Range(1 << 0, 1 << 20);
BENCHMARK(BM_TwiddleNew)->RangeMultiplier(16)->Range(1 << 0, 1 << 20);

// test dynamic cast vs static cast

class Base {
public:
  Base(const int val = 0) : _value{val} {}
  virtual ~Base() = default; // Ensure a virtual destructor for dynamic_cast

  int _value;
};

class Derived1 : public Base {
public:
  Derived1(const int val = 1) : Base(val) {}
};

class Derived2 : public Base {
public:
  Derived2(const int val = 2) : Base(val) {}
};

class Derived3 : public Base {
public:
  Derived3(const int val = 3) : Base(val) {}
};

static std::map<int,
                std::pair<std::vector<int>, std::vector<std::unique_ptr<Base>>>>
    cast_map;

std::pair<std::vector<int>, std::vector<std::unique_ptr<Base>>> &
getCast(const int size) {
  if (cast_map.find(size) == cast_map.end()) {
    std::vector<int> values(size);
    std::vector<std::unique_ptr<Base>> bases(size);
    for (int i = 0; i < size; ++i) {
      values[i] = rand() % 3; // Randomly select 0, 1, or 2
      switch (values[i]) {
      case 0:
        bases[i] = std::make_unique<Derived1>(i);
        break;
      case 1:
        bases[i] = std::make_unique<Derived2>(i);
        break;
      case 2:
        bases[i] = std::make_unique<Derived3>(i);
        break;
      }
    }
    cast_map[size] = {values, std::move(bases)};
  }
  return cast_map[size];
}

static std::vector<std::function<void(std::unique_ptr<Base> &)>>
    static_cast_functions = {
        [](std::unique_ptr<Base> &base) {
          benchmark::DoNotOptimize(static_cast<Derived1 *>(base.get())->_value);
        },
        [](std::unique_ptr<Base> &base) {
          benchmark::DoNotOptimize(static_cast<Derived2 *>(base.get())->_value);
        },
        [](std::unique_ptr<Base> &base) {
          benchmark::DoNotOptimize(static_cast<Derived3 *>(base.get())->_value);
        }};

static std::vector<std::function<void(std::unique_ptr<Base> &)>>
    dynamic_cast_functions = {
        [](std::unique_ptr<Base> &base) {
          benchmark::DoNotOptimize(
              dynamic_cast<Derived1 *>(base.get())->_value);
        },
        [](std::unique_ptr<Base> &base) {
          benchmark::DoNotOptimize(
              dynamic_cast<Derived2 *>(base.get())->_value);
        },
        [](std::unique_ptr<Base> &base) {
          benchmark::DoNotOptimize(
              dynamic_cast<Derived3 *>(base.get())->_value);
        }};

static void BM_CastStatic(benchmark::State &state) {
  auto &cast_data = getCast(state.range(0));
  auto &values = cast_data.first;
  auto &bases = cast_data.second;

  for (auto _ : state) {
    for (size_t i = 0; i < bases.size(); ++i) {
      static_cast_functions[values[i]](bases[i]);
    }
  }
}

static void BM_CastDynamic(benchmark::State &state) {
  auto &cast_data = getCast(state.range(0));
  auto &values = cast_data.first;
  auto &bases = cast_data.second;

  for (auto _ : state) {
    for (size_t i = 0; i < bases.size(); ++i) {
      dynamic_cast_functions[values[i]](bases[i]);
    }
  }
}

BENCHMARK(BM_CastStatic)->RangeMultiplier(16)->Range(1 << 0, 1 << 20);
BENCHMARK(BM_CastDynamic)->RangeMultiplier(16)->Range(1 << 0, 1 << 20);

BENCHMARK_MAIN();