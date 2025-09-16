#include <algorithm>
#include <atomic>
#include <benchmark/benchmark.h>
#include <functional>
#include <iostream>
#include <mutex>
#include <random>
#include <string>
#include <vector>

std::string f0() { return "x <= 1e-5\n"; }
std::string f1() { return "(1e-5, 1e-4]\n"; }
std::string f2() { return "(1e-4, 1e-3]\n"; }
std::string f3() { return "(1e-3, 1e-2]\n"; }
std::string f4() { return "(1e-2, 1e-1]\n"; }
std::string f5() { return "(1e-1, 1]\n"; }
std::string f6() { return "(1, 1e1]\n"; }
std::string f7() { return "(1e1, 1e2]\n"; }
std::string f8() { return "(1e2, 1e3]\n"; }
std::string f9() { return "x > 1e3\n"; }

std::string branch(const double val) {
  if (val > 1e3)
    return f9();
  if (val > 1e2)
    return f8();
  if (val > 1e1)
    return f7();
  if (val > 1)
    return f6();
  if (val > 1e-1)
    return f5();
  if (val > 1e-2)
    return f4();
  if (val > 1e-3)
    return f3();
  if (val > 1e-4)
    return f2();
  if (val > 1e-5)
    return f1();
  return f0();
}

inline int ilog10(double x) {
  union {
    double d;
    uint64_t u;
  } val = {x};
  int exp = ((val.u >> 52) & 0x7FF) - 1023;
  // log10(x) ≈ log2(x) * log10(2)
  return static_cast<int>(exp * 0.30103); // 0.30103 ≈ log10(2)
}
std::string branchless(const double val) {
  constexpr int NUM_RANGES = 10;
  constexpr double thresholds[NUM_RANGES - 1] = {1e-5, 1e-4, 1e-3, 1e-2, 1e-1,
                                                 1,    1e1,  1e2,  1e3};
  using Func = std::string (*)();
  static constexpr Func funcs[10] = {f0, f1, f2, f3, f4, f5, f6, f7, f8, f9};
  // Binary search without branches (can also be std::lower_bound)
  double logx = ilog10(val); // avoid -inf
  int idx = static_cast<int>(std::floor(logx)) + 5;

  // Clamp to [0, 9]
  idx = std::clamp(idx, 0, 9);
  return funcs[idx]();
}

const int vec_size = 10000;
std::vector<double> vec(vec_size);
std::once_flag vec_init_flag;

void initialize_vectors() {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<double> dis(-6, 4);
  std::generate(vec.begin(), vec.end(), [&]() {
    double rand_val = dis(gen);
    return std::pow(10,
                    rand_val); // Generate logarithmically uniform random number
  });
}

static void BM_Branch(benchmark::State &state) {
  std::call_once(vec_init_flag, initialize_vectors);
  size_t size = state.range(0);
  for (auto _ : state) {
    for (size_t i = 0; i < size; ++i) {
      benchmark::DoNotOptimize(branch(vec[i % vec_size]));
    }
  }
}

static void BM_Branchless(benchmark::State &state) {
  std::call_once(vec_init_flag, initialize_vectors);
  size_t size = state.range(0);
  for (auto _ : state) {
    for (size_t i = 0; i < size; ++i) {
      benchmark::DoNotOptimize(branchless(vec[i % vec_size]));
    }
  }
}

BENCHMARK(BM_Branch)->RangeMultiplier(16)->Range(1 << 0, 1 << 20);
BENCHMARK(BM_Branchless)->RangeMultiplier(16)->Range(1 << 0, 1 << 20);

BENCHMARK_MAIN();