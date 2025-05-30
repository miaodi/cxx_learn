#include <gtest/gtest.h>
#include <permute.hpp>
#include <vector>

TEST(permutation, permutation_generate) {
  int size = 100;
  std::vector<int> perm(size);
  std::vector<bool> used(size, false);
  permutation_generate(perm.data(), size);
  for (int i = 0; i < size; ++i) {
    used[perm[i]] = true;
  }
  for (int i = 0; i < size; ++i) {
    EXPECT_TRUE(used[i]) << "Element " << i
                         << " is not used in the permutation.";
  }
}

TEST(permutation, permutation_cpu_vs_permutaion_cuda) {
  int size = 2097152;
  std::vector<int> perm(size);
  permutation_generate(perm.data(), size);
  std::vector<double> from(size);
  std::vector<double> to_cpu(size);
  std::vector<double> to_cuda(size);
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
  permute_cpu<double, false>(from.data(), to_cpu.data(), perm.data(), size);
  permute_cuda<double, false>(device_from, device_to, perm_device, size);
  cudaMemcpy(to_cuda.data(), device_to, size * sizeof(double),
             cudaMemcpyDeviceToHost);
  cudaFree(device_from);
  cudaFree(device_to);
  cudaFree(perm_device);
  for (int i = 0; i < size; ++i) {
    EXPECT_DOUBLE_EQ(to_cpu[i], to_cuda[i])
        << "Mismatch at index " << i << ": CPU value = " << to_cpu[i]
        << ", CUDA value = " << to_cuda[i];
  }
}