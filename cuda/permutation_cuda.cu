#include <algorithm>
#include <benchmark/benchmark.h>
#include <iostream>
#include <permute.hpp>
#include <random>

static void Cuda_Permute(benchmark::State &state)
{
  size_t size = state.range(0);
  std::vector<int> perm(size);
  permutation_generate(perm.data(), size);

  std::vector<double> from(size);
  std::vector<double> to(size);
  random_vector_generate(from.data(), size);
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
    permute_cuda<double, false>(device_from, device_to,
                                perm_device, size);
    std::swap(device_from, device_to);
  }
  state.SetBytesProcessed(int64_t(state.iterations()) *
                          int64_t(state.range(0)) * 8);
  cudaFree(device_from);
  cudaFree(device_to);
  cudaFree(perm_device);
}

BENCHMARK(Cuda_Permute)->RangeMultiplier(2)->Range(1 << 8, 1 << 28);
BENCHMARK_MAIN();