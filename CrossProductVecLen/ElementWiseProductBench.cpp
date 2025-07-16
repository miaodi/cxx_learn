#include <algorithm>
#include <benchmark/benchmark.h>
#include <immintrin.h> // AVX2/FMA
#include <iostream>
#include <random>
#include <vector>

// double CrossProdVecLen2(double const *a, double const *b) {
//   double dot_ab = a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
//   double norm2_a = a[0] * a[0] + a[1] * a[1] + a[2] * a[2];
//   double norm2_b = b[0] * b[0] + b[1] * b[1] + b[2] * b[2];
//   return norm2_a * norm2_b - dot_ab * dot_ab;
// }

// static void CrossProdVecLen2BM(benchmark::State &state) {

//   // First create an instance of an engine.
//   std::random_device rnd_device;
//   // Specify the engine and distribution.
//   std::mt19937 mersenne_engine{rnd_device()}; // Generates random integers
//   std::uniform_real_distribution<double> dist{-1.0,
//                                              1.0}; // Range for double values
//   auto gen = [&]() { return dist(mersenne_engine); };
//   const auto size = state.range(0);
//   const size_t vec_size = 1000;
//   std::vector<double[3]> a(vec_size);
//   std::vector<double[3]> b(vec_size);
//   for (size_t i = 0; i < vec_size; ++i) {
//     a[i][0] = gen();
//     a[i][1] = gen();
//     a[i][2] = gen();
//     b[i][0] = gen();
//     b[i][1] = gen();
//     b[i][2] = gen();
//   }
//   for (auto _ : state) {
//     for (size_t i = 0; i < size; i++) {
//       benchmark::DoNotOptimize(
//           CrossProdVecLen2(a[i % vec_size], b[i % vec_size]));
//     }
//   }
// }

// BENCHMARK(CrossProdVecLen2BM)->RangeMultiplier(2)->Range(1 << 1, 1 << 16);

static void ElementWiseProduct3D(benchmark::State &state) {
  // First create an instance of an engine.
  std::random_device rnd_device;
  // Specify the engine and distribution.
  std::mt19937 mersenne_engine{rnd_device()}; // Generates random integers
  std::uniform_real_distribution<double> dist{-1.0,
                                              1.0}; // Range for double values
  auto gen = [&]() { return dist(mersenne_engine); };
  const auto size = state.range(0);
  const size_t vec_size = 1000;

  std::vector<double[3]> a(vec_size);
  std::vector<double[3]> b(vec_size);
  std::vector<double[3]> c(vec_size);

  for (size_t i = 0; i < vec_size; ++i) {
    a[i][0] = gen();
    a[i][1] = gen();
    a[i][2] = gen();
    b[i][0] = gen();
    b[i][1] = gen();
    b[i][2] = gen();
  }

  for (auto _ : state) {
    for (size_t i = 0; i < size; i++) {
      // Element-wise product
      c[i % vec_size][0] = a[i % vec_size][0] * b[i % vec_size][0];
      c[i % vec_size][1] = a[i % vec_size][1] * b[i % vec_size][1];
      c[i % vec_size][2] = a[i % vec_size][2] * b[i % vec_size][2];
      benchmark::DoNotOptimize(c[i % vec_size]);
    }
  }
}
BENCHMARK(ElementWiseProduct3D)->RangeMultiplier(2)->Range(1 << 1, 1 << 16);

struct alignas(32) AlignedDouble3 {
  double data[4];
  AlignedDouble3() : data{0.0, 0.0, 0.0, 0.0} {}
};

static void ElementWiseProduct3DAlign(benchmark::State &state) {
  // First create an instance of an engine.
  std::random_device rnd_device;
  // Specify the engine and distribution.
  std::mt19937 mersenne_engine{rnd_device()}; // Generates random integers
  std::uniform_real_distribution<double> dist{-1.0,
                                              1.0}; // Range for double values
  auto gen = [&]() { return dist(mersenne_engine); };
  const auto size = state.range(0);
  const size_t vec_size = 1000;

  std::vector<AlignedDouble3> a(vec_size);
  std::vector<AlignedDouble3> b(vec_size);
  std::vector<AlignedDouble3> c(vec_size);

  for (size_t i = 0; i < vec_size; ++i) {
    a[i].data[0] = gen();
    a[i].data[1] = gen();
    a[i].data[2] = gen();
    b[i].data[0] = gen();
    b[i].data[1] = gen();
    b[i].data[2] = gen();
  }
  // std::cout << "a[0] address: " << &a[0].data[0] << "\n";
  // std::cout << "a[1] address: " << &a[1].data[0] << "\n";
  // std::cout << "a[2] address: " << &a[2].data[0] << "\n";

  for (auto _ : state) {
    for (size_t i = 0; i < size; i++) {
      // Element-wise product
      c[i % vec_size].data[0] =
          a[i % vec_size].data[0] * b[i % vec_size].data[0];
      c[i % vec_size].data[1] =
          a[i % vec_size].data[1] * b[i % vec_size].data[1];
      c[i % vec_size].data[2] =
          a[i % vec_size].data[2] * b[i % vec_size].data[2];
      benchmark::DoNotOptimize(c[i % vec_size]);
    }
  }
}
BENCHMARK(ElementWiseProduct3DAlign)
    ->RangeMultiplier(2)
    ->Range(1 << 1, 1 << 16);

static void ElementWiseProduct3DAlignAVX2(benchmark::State &state) {
  // First create an instance of an engine.
  std::random_device rnd_device;
  std::mt19937 mersenne_engine{rnd_device()}; // Generates random integers
  std::uniform_real_distribution<double> dist{-1.0,
                                              1.0}; // Range for double values
  auto gen = [&]() { return dist(mersenne_engine); };
  const auto size = state.range(0);
  const size_t vec_size = 1000;

  std::vector<AlignedDouble3> a(vec_size);
  std::vector<AlignedDouble3> b(vec_size);
  std::vector<AlignedDouble3> c(vec_size);

  for (size_t i = 0; i < vec_size; ++i) {
    a[i].data[0] = gen();
    a[i].data[1] = gen();
    a[i].data[2] = gen();
    b[i].data[0] = gen();
    b[i].data[1] = gen();
    b[i].data[2] = gen();
  }

  for (auto _ : state) {
    for (size_t i = 0; i < size; i++) { // Process 4 elements at a time
      // Load 4 AlignedDouble3 elements into AVX registers
      __m256d a0 = _mm256_load_pd(&a[i % vec_size].data[0]);

      __m256d b0 = _mm256_load_pd(&b[i % vec_size].data[0]);

      // Perform element-wise multiplication
      __m256d c0 = _mm256_mul_pd(a0, b0);

      // Store the results back to memory
      _mm256_store_pd(&c[i % vec_size].data[0], c0);

      benchmark::DoNotOptimize(c[i % vec_size]);
    }
  }
}
BENCHMARK(ElementWiseProduct3DAlignAVX2)
    ->RangeMultiplier(2)
    ->Range(1 << 1, 1 << 16);
BENCHMARK_MAIN();