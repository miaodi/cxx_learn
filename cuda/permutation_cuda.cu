#include <../config.h>
#include <algorithm>
#include <benchmark/benchmark.h>
#include <iostream>
#include <iterator>
#include <new>
#include <numeric>
#include <random>
#include <vector>
#include <cmath>

#ifdef USE_CUDA
#define MAX_THREADS 1024

void permute_control(int n, int &blocks, int &threads,
                     int maxThreads = MAX_THREADS)
{
  threads = std::min(static_cast<int>(pow(2, ceil(log2(n)))), MAX_THREADS);
  blocks = (n + threads - 1) / threads;
}

template <typename T, bool backward>
__global__ void permute_core(const T *in, T *out, int *perm, int length)
{
  unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

  /*
   * Return the number of valid entries found
   */
  if (idx < length)
  {
    if (backward)
      out[idx] = in[perm[idx]];
    else
      out[perm[idx]] = in[idx];
  }
}

template <typename T, bool backward>
void permute(const T *in, T *out, int *indices, int length)
{
  int threads;
  int blocks;

  permute_control(length, blocks, threads);
  permute_core<T, backward>
      <<<blocks, threads>>>(in, out, indices, length);
}

static void Cuda_Permute(benchmark::State &state)
{
  // First create an instance of an engine.
  std::random_device rnd_device;
  // Specify the engine and distribution.
  std::mt19937 mersenne_engine{rnd_device()}; // Generates random integers
  std::uniform_real_distribution<double> dist{0, 1};
  auto gen = [&]()
  { return dist(mersenne_engine); };
  const size_t size = state.range(0);
  std::vector<int> perm(size);
  for (size_t i = 0; i < size; ++i)
  {
    perm[i] = i;
  }
  std::shuffle(perm.begin(), perm.end(), mersenne_engine);

  std::vector<double> from(size);
  std::vector<double> to(size);
  std::generate(from.begin(), from.end(), gen);
  double *device_from;
  double *device_to;
  int *perm_device;
  cudaMalloc(&device_from, size * sizeof(double));
  cudaMalloc(&device_to, size * sizeof(double));
  cudaMalloc(&perm_device, size * sizeof(int));

  cudaMemcpy(device_from, from.data(), size * sizeof(double),
             cudaMemcpyHostToDevice);
  cudaMemcpy(perm_device, perm.data(), size * sizeof(int),
             cudaMemcpyHostToDevice);
  for (auto _ : state)
  {
    permute<double, false>(device_from, device_to,
                           perm_device, size);
    std::swap(device_from, device_to);
  }
  state.SetBytesProcessed(int64_t(state.iterations()) *
                          int64_t(state.range(0)) * 8);
  cudaFree(device_from);
  cudaFree(device_to);
  cudaFree(perm_device);
}

BENCHMARK(Cuda_Permute)->RangeMultiplier(4)->Range(1 << 4, 1 << 30);
BENCHMARK_MAIN();
#endif