#include <cstddef>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <vector>
#include <cuda_runtime.h>

namespace {

constexpr int kBlockSize = 256;

void check_cuda(cudaError_t status, const char *operation) {
  if (status != cudaSuccess) {
    throw std::runtime_error(operation);
  }
}



[[maybe_unused]] __global__ void reduction_kernel(const float *input,
                                                  float *output,
                                                  std::size_t size) {
    const int tid = threadIdx.x;
    const int block_start = 2 * blockIdx.x * blockDim.x;
    __shared__ float shared_data[kBlockSize];
    const int first = block_start + tid;
    const int second = block_start + blockDim.x + tid;
    shared_data[tid] = first < size ? input[first] : 0.0f;
    if(second < size){
      shared_data[tid] += input[second];
    }
    __syncthreads();
    for(int stride = blockDim.x/2; stride >= 32; stride /= 2){
      if(tid < stride){
        shared_data[tid] += shared_data[tid + stride];
      }
      __syncthreads();
    }
    float value = shared_data[tid];
    if (tid < 32) {
      value += __shfl_down_sync(0xffffffff, value, 16);
      value += __shfl_down_sync(0xffffffff, value, 8);
      value += __shfl_down_sync(0xffffffff, value, 4);
      value += __shfl_down_sync(0xffffffff, value, 2);
      value += __shfl_down_sync(0xffffffff, value, 1);
    }

    if (tid == 0) {
      output[blockIdx.x] = value;
    }
}

float device_reduction(const float *device_input, std::size_t size) {
  // Practice task: allocate any temporary device storage you need, launch
  // reduction_kernel one or more times, copy the final scalar back to the host,
  // and return it.
  float* tmp1;
  check_cuda(cudaMalloc(&tmp1, size * sizeof(float)), "allocate tmp1");
  cudaMemcpy(tmp1, device_input, size * sizeof(float), cudaMemcpyDeviceToDevice);
  float * tmp2;
  int num_blocks = (size + 2 * kBlockSize - 1) / (2 * kBlockSize);
  check_cuda(cudaMalloc(&tmp2, num_blocks * sizeof(float)), "allocate tmp2");
  while(size > 1){
    reduction_kernel<<<num_blocks, kBlockSize>>>(tmp1, tmp2, size);
    check_cuda(cudaGetLastError(), "launch reduction kernel");
    std::swap(tmp1, tmp2);
    size = num_blocks;
    num_blocks = (size + 2 * kBlockSize - 1) / (2 * kBlockSize);
  }
  float result;
  check_cuda(cudaMemcpy(&result, tmp1, sizeof(float), cudaMemcpyDeviceToHost), "copy result to host");
  check_cuda(cudaFree(tmp1), "free tmp1");
  check_cuda(cudaFree(tmp2), "free tmp2");
  return result;
}

std::vector<float> make_input(std::size_t size) {
  std::vector<float> values(size);
  for (std::size_t i = 0; i < values.size(); ++i) {
    values[i] = 1.0f + static_cast<float>(i % 7) * 0.125f;
  }
  return values;
}

double cpu_reference_sum(const std::vector<float> &values) {
  return std::accumulate(values.begin(), values.end(), 0.0);
}

void run_demo(std::size_t size) {
  const std::vector<float> host_input = make_input(size);
  const double cpu_sum = cpu_reference_sum(host_input);

  float *device_input = nullptr;
  check_cuda(cudaMalloc(&device_input, host_input.size() * sizeof(float)),
             "allocate device input");
  check_cuda(cudaMemcpy(device_input, host_input.data(),
                        host_input.size() * sizeof(float),
                        cudaMemcpyHostToDevice),
             "copy input to device");

  const float practice_sum = device_reduction(device_input, size);

  check_cuda(cudaFree(device_input), "free device input");

  std::cout << "CUDA reduction practice\n";
  std::cout << "input elements:        " << size << '\n';
  std::cout << "threads per block:     " << kBlockSize << '\n';
  std::cout << "CPU reference sum:     " << cpu_sum << '\n';
  std::cout << "device_reduction sum:  " << practice_sum
            << "  <-- implement this\n";
}

} // namespace

int main() {
  try {
    run_demo((1 << 20) + 123);
  } catch (const std::exception &error) {
    std::cerr << "error: " << error.what() << '\n';
    return 1;
  }
  return 0;
}
