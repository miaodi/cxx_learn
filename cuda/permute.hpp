#pragma once

#include <cmath>
#ifdef __NVCC__
#include <cuda_runtime.h>
#endif
#include <new>
#include <numeric>
#include <omp.h>
#include <random>
#include <vector>
#include <algorithm>

#define MAX_THREADS 1024

#ifdef __NVCC__
void permute_control(int n, int &blocks, int &threads,
                     int maxThreads = MAX_THREADS) {
  threads = std::min(static_cast<int>(pow(2, ceil(log2(n)))), MAX_THREADS);
  blocks = (n + threads - 1) / threads;
}

template <typename T, bool backward>
__global__ void permute_core(const T *in, T *out, int *perm, int length) {
  unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

  /*
   * Return the number of valid entries found
   */
  if (idx < length) {
    if constexpr (backward)
      out[idx] = in[perm[idx]];
    else
      out[perm[idx]] = in[idx];
  }
}

template <typename T, bool backward>
void permute_cuda(const T *in, T *out, int *perm, int length) {
  int threads;
  int blocks;
  permute_control(length, blocks, threads);
  permute_core<T, backward><<<blocks, threads>>>(in, out, perm, length);
}
#endif

template <typename T, bool backward, int threads = 1>
void permute_cpu(const T *in, T *out, int *perm, int length) {
  if constexpr (threads <= 1) {
    for (int i = 0; i < length; ++i) {
      if constexpr (backward)
        out[i] = in[perm[i]];
      else
        out[perm[i]] = in[i];
    }
  } else {
#pragma omp parallel for num_threads(threads)
    for (int i = 0; i < length; ++i) {
      if constexpr (backward)
        out[i] = in[perm[i]];
      else
        out[perm[i]] = in[i];
    }
  }
}

void permutation_generate(int *perm, int length) {
  for (size_t i = 0; i < length; ++i) {
    perm[i] = i;
  }
  std::iota(perm, perm + length, 0);
  std::random_device rd;
  std::mt19937 g(rd());
  std::shuffle(perm, perm + length, g);
}

void random_vector_generate(double *vec, int length) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<double> dis(0.0, 1.0);
  for (int i = 0; i < length; ++i) {
    vec[i] = dis(gen);
  }
}