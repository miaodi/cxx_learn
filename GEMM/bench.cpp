#include "gemm.hpp"
#include <algorithm>
#include <benchmark/benchmark.h>
#include <random>
#include <vector>

#define MIN 2
#define MAX 1024
static void BM_MatMatMul(benchmark::State &state) {
  int n = static_cast<int>(state.range(0));
  std::vector<float> A(n * n), B(n * n), C(n * n);

  // Initialize matrices with random values
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<float> dis(0.0f, 1.0f);
  std::generate(A.begin(), A.end(), [&]() { return dis(gen); });
  std::generate(B.begin(), B.end(), [&]() { return dis(gen); });

  for (auto _ : state) {
    gemm::MatMatMul(A.data(), B.data(), C.data(), n, n, n);
    benchmark::DoNotOptimize(C);
  }

  state.SetComplexityN(n);
}

BENCHMARK(BM_MatMatMul)->RangeMultiplier(2)->Range(MIN, MAX)->Complexity();

static void BM_MatMatTransMul(benchmark::State &state) {
  int n = static_cast<int>(state.range(0));
  std::vector<float> A(n * n), B(n * n), C(n * n);

  // Initialize matrices with random values
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<float> dis(0.0f, 1.0f);
  std::generate(A.begin(), A.end(), [&]() { return dis(gen); });
  std::generate(B.begin(), B.end(), [&]() { return dis(gen); });

  for (auto _ : state) {
    gemm::MatMatTransMul(A.data(), B.data(), C.data(), n, n, n);
    benchmark::DoNotOptimize(C);
  }

  state.SetComplexityN(n);
}

BENCHMARK(BM_MatMatTransMul)->RangeMultiplier(2)->Range(MIN, MAX)->Complexity();

static void BM_TiledMatMatMul(benchmark::State &state) {
  int n = static_cast<int>(state.range(0));
  std::vector<float> A(n * n), B(n * n), C(n * n);

  // Initialize matrices with random values
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<float> dis(0.0f, 1.0f);
  std::generate(A.begin(), A.end(), [&]() { return dis(gen); });
  std::generate(B.begin(), B.end(), [&]() { return dis(gen); });

  for (auto _ : state) {
    gemm::TiledMatMatMul(A.data(), B.data(), C.data(), n, n, n);
    benchmark::DoNotOptimize(C);
  }

  state.SetComplexityN(n);
}

BENCHMARK(BM_TiledMatMatMul)->RangeMultiplier(2)->Range(MIN, MAX)->Complexity();

static void BM_TiledMatMatMulInternalTrans(benchmark::State &state) {
  int n = static_cast<int>(state.range(0));
  std::vector<float> A(n * n), B(n * n), C(n * n);

  // Initialize matrices with random values
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<float> dis(0.0f, 1.0f);
  std::generate(A.begin(), A.end(), [&]() { return dis(gen); });
  std::generate(B.begin(), B.end(), [&]() { return dis(gen); });

  for (auto _ : state) {
    gemm::TiledMatMatMulInternalTrans(A.data(), B.data(), C.data(), n, n, n);
    benchmark::DoNotOptimize(C);
  }

  state.SetComplexityN(n);
}

BENCHMARK(BM_TiledMatMatMulInternalTrans)
    ->RangeMultiplier(2)
    ->Range(MIN, MAX)
    ->Complexity();

static void BM_TiledMatMatTransMul(benchmark::State &state) {
  int n = static_cast<int>(state.range(0));
  std::vector<float> A(n * n), B(n * n), C(n * n);

  // Initialize matrices with random values
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<float> dis(0.0f, 1.0f);
  std::generate(A.begin(), A.end(), [&]() { return dis(gen); });
  std::generate(B.begin(), B.end(), [&]() { return dis(gen); });

  for (auto _ : state) {
    gemm::TiledMatMatTransMul(A.data(), B.data(), C.data(), n, n, n);
    benchmark::DoNotOptimize(C);
  }

  state.SetComplexityN(n);
}

BENCHMARK(BM_TiledMatMatTransMul)
    ->RangeMultiplier(2)
    ->Range(MIN, MAX)
    ->Complexity();

static void BM_TiledMatMatMulInternalTiledPadded(benchmark::State &state) {
  int n = static_cast<int>(state.range(0));
  std::vector<float> A(n * n), B(n * n), C(n * n);

  // Initialize matrices with random values
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<float> dis(0.0f, 1.0f);
  std::generate(A.begin(), A.end(), [&]() { return dis(gen); });
  std::generate(B.begin(), B.end(), [&]() { return dis(gen); });

  for (auto _ : state) {
    gemm::TiledMatMatMulInternalTiledPadded(A.data(), B.data(), C.data(), n, n,
                                            n);
    benchmark::DoNotOptimize(C);
  }

  state.SetComplexityN(n);
}

BENCHMARK(BM_TiledMatMatMulInternalTiledPadded)
    ->RangeMultiplier(2)
    ->Range(MIN, MAX)
    ->Complexity();

static void BM_TiledMatMatMulInternalTransTiledPadded(benchmark::State &state) {
  int n = static_cast<int>(state.range(0));
  std::vector<float> A(n * n), B(n * n), C(n * n);

  // Initialize matrices with random values
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<float> dis(0.0f, 1.0f);
  std::generate(A.begin(), A.end(), [&]() { return dis(gen); });
  std::generate(B.begin(), B.end(), [&]() { return dis(gen); });

  for (auto _ : state) {
    gemm::TiledMatMatMulInternalTransTiledPadded(A.data(), B.data(), C.data(),
                                                 n, n, n);
    benchmark::DoNotOptimize(C);
  }

  state.SetComplexityN(n);
}

BENCHMARK(BM_TiledMatMatMulInternalTransTiledPadded)
    ->RangeMultiplier(2)
    ->Range(MIN, MAX)
    ->Complexity();
BENCHMARK_MAIN();