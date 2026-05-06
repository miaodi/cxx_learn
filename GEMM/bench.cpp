#include "dgemm_nn.h"
#include "gemm.hpp"
#include "ulmBLASgemm.hpp"
#include <algorithm>
#include <benchmark/benchmark.h>
#include <random>
#include <vector>

#define MIN 2
#define MAX 1024
static void BM_gemm_naive(benchmark::State &state) {
  int n = static_cast<int>(state.range(0));
  std::vector<float> A(n * n), B(n * n), C(n * n);

  // Initialize matrices with random values
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<float> dis(0.0f, 1.0f);
  std::generate(A.begin(), A.end(), [&]() { return dis(gen); });
  std::generate(B.begin(), B.end(), [&]() { return dis(gen); });

  for (auto _ : state) {
    gemm::gemm_naive(n, n, n, 1.0f, A.data(), n, B.data(), n, 0.0f, C.data(),
                     n);
    benchmark::DoNotOptimize(C);
  }

  state.SetComplexityN(n);
}

BENCHMARK(BM_gemm_naive)->RangeMultiplier(2)->Range(MIN, MAX)->Complexity();

static void BM_gemm_ikj(benchmark::State &state) {
  int n = static_cast<int>(state.range(0));
  std::vector<float> A(n * n), B(n * n), C(n * n);

  // Initialize matrices with random values
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<float> dis(0.0f, 1.0f);
  std::generate(A.begin(), A.end(), [&]() { return dis(gen); });
  std::generate(B.begin(), B.end(), [&]() { return dis(gen); });

  for (auto _ : state) {
    gemm::gemm_ikj(n, n, n, 1.0f, A.data(), n, B.data(), n, 0.0f, C.data(),
                   n);
    benchmark::DoNotOptimize(C);
  }

  state.SetComplexityN(n);
}

BENCHMARK(BM_gemm_ikj)->RangeMultiplier(2)->Range(MIN, MAX)->Complexity();

static void BM_gemm_blocked(benchmark::State &state) {
  int n = static_cast<int>(state.range(0));
  std::vector<float> A(n * n), B(n * n), C(n * n);

  // Initialize matrices with random values
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<float> dis(0.0f, 1.0f);
  std::generate(A.begin(), A.end(), [&]() { return dis(gen); });
  std::generate(B.begin(), B.end(), [&]() { return dis(gen); });

  for (auto _ : state) {
    gemm::gemm_blocked<float>(n, n, n, 1.0f, A.data(), n, B.data(), n, 0.0f,
                              C.data(), n);
    benchmark::DoNotOptimize(C);
  }

  state.SetComplexityN(n);
}

BENCHMARK(BM_gemm_blocked)->RangeMultiplier(2)->Range(MIN, MAX)->Complexity();

static void BM_gemm_packed_b(benchmark::State &state) {
  int n = static_cast<int>(state.range(0));
  std::vector<float> A(n * n), B(n * n), C(n * n);

  // Initialize matrices with random values
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<float> dis(0.0f, 1.0f);
  std::generate(A.begin(), A.end(), [&]() { return dis(gen); });
  std::generate(B.begin(), B.end(), [&]() { return dis(gen); });

  for (auto _ : state) {
    gemm::gemm_packed_b<float>(n, n, n, 1.0f, A.data(), n, B.data(), n, 0.0f,
                               C.data(), n);
    benchmark::DoNotOptimize(C);
  }

  state.SetComplexityN(n);
}

BENCHMARK(BM_gemm_packed_b)->RangeMultiplier(2)->Range(MIN, MAX)->Complexity();

static void BM_gemm_packed_ab(benchmark::State &state) {
  int n = static_cast<int>(state.range(0));
  std::vector<float> A(n * n), B(n * n), C(n * n);

  // Initialize matrices with random values
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<float> dis(0.0f, 1.0f);
  std::generate(A.begin(), A.end(), [&]() { return dis(gen); });
  std::generate(B.begin(), B.end(), [&]() { return dis(gen); });

  for (auto _ : state) {
    gemm::gemm_packed_ab<float>(n, n, n, 1.0f, A.data(), n, B.data(), n, 0.0f,
                                C.data(), n);
    benchmark::DoNotOptimize(C);
  }

  state.SetComplexityN(n);
}

BENCHMARK(BM_gemm_packed_ab)->RangeMultiplier(2)->Range(MIN, MAX)->Complexity();

static void BM_gemm_packed_ab_register_blocked(benchmark::State &state) {
  int n = static_cast<int>(state.range(0));
  std::vector<float> A(n * n), B(n * n), C(n * n);

  // Initialize matrices with random values
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<float> dis(0.0f, 1.0f);
  std::generate(A.begin(), A.end(), [&]() { return dis(gen); });
  std::generate(B.begin(), B.end(), [&]() { return dis(gen); });

  for (auto _ : state) {
    gemm::gemm_packed_ab_register_blocked<float>(
        n, n, n, 1.0f, A.data(), n, B.data(), n, 0.0f, C.data(), n);
    benchmark::DoNotOptimize(C);
  }

  state.SetComplexityN(n);
}

BENCHMARK(BM_gemm_packed_ab_register_blocked)
    ->RangeMultiplier(2)
    ->Range(MIN, MAX)
    ->Complexity();

static void BM_gemm_packed_ab_register_blocked_avx2_float(
    benchmark::State &state) {
  int n = static_cast<int>(state.range(0));
  std::vector<float> A(n * n), B(n * n), C(n * n);

  // Initialize matrices with random values
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<float> dis(0.0f, 1.0f);
  std::generate(A.begin(), A.end(), [&]() { return dis(gen); });
  std::generate(B.begin(), B.end(), [&]() { return dis(gen); });

  for (auto _ : state) {
    gemm::gemm_packed_ab_register_blocked_avx2_float(
        n, n, n, 1.0f, A.data(), n, B.data(), n, 0.0f, C.data(), n);
    benchmark::DoNotOptimize(C);
  }

  state.SetComplexityN(n);
}

BENCHMARK(BM_gemm_packed_ab_register_blocked_avx2_float)
    ->RangeMultiplier(2)
    ->Range(MIN, MAX)
    ->Complexity();

static void BM_gemm_packed_ab_prepack_a(benchmark::State &state) {
  int n = static_cast<int>(state.range(0));
  std::vector<float> A(n * n), B(n * n), C(n * n);

  // Initialize matrices with random values
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<float> dis(0.0f, 1.0f);
  std::generate(A.begin(), A.end(), [&]() { return dis(gen); });
  std::generate(B.begin(), B.end(), [&]() { return dis(gen); });

  for (auto _ : state) {
    gemm::gemm_packed_ab_prepack_a<float>(
        n, n, n, 1.0f, A.data(), n, B.data(), n, 0.0f, C.data(), n);
    benchmark::DoNotOptimize(C);
  }

  state.SetComplexityN(n);
}

BENCHMARK(BM_gemm_packed_ab_prepack_a)
    ->RangeMultiplier(2)
    ->Range(MIN, MAX)
    ->Complexity();

static void BM_ulmBLASgemm(benchmark::State &state) {
  int n = static_cast<int>(state.range(0));
  std::vector<float> A(n * n), B(n * n), C(n * n);

  // Initialize matrices with random values
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<float> dis(0.0f, 1.0f);
  std::generate(A.begin(), A.end(), [&]() { return dis(gen); });
  std::generate(B.begin(), B.end(), [&]() { return dis(gen); });
  gemm::gemm_pure_c<float> ulmBLASgemm;
  for (auto _ : state) {
    ulmBLASgemm(A.data(), B.data(), C.data(), n, n, n);
    benchmark::DoNotOptimize(C);
  }

  state.SetComplexityN(n);
}

BENCHMARK(BM_ulmBLASgemm)->RangeMultiplier(2)->Range(MIN, MAX)->Complexity();

static void BM_gemm_nn(benchmark::State &state) {
  int n = static_cast<int>(state.range(0));
  std::vector<float> A(n * n), B(n * n), C(n * n);

  // Initialize matrices with random values
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<float> dis(0.0f, 1.0f);
  std::generate(A.begin(), A.end(), [&]() { return dis(gen); });
  std::generate(B.begin(), B.end(), [&]() { return dis(gen); });

  for (auto _ : state) {
    gemm::gemm_nn<float>(n, n, n, 1.0f, A.data(), n, 1, B.data(), n, 1, 0.0f,
                         C.data(), n, 1);
    benchmark::DoNotOptimize(C);
  }

  state.SetComplexityN(n);
}

BENCHMARK(BM_gemm_nn)->RangeMultiplier(2)->Range(MIN, MAX)->Complexity();
BENCHMARK_MAIN();
