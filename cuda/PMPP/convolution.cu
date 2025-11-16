#include "convolution.h"
#include <cuda_runtime.h>

// Define kernel in constant memory (max 64KB)
// Typical max kernel size: 7x7 or 11x11 depending on usage
constexpr int MAX_KERNEL_SIZE = 11;
__constant__ float d_kernel[MAX_KERNEL_SIZE * MAX_KERNEL_SIZE];

void convolution_2d(const float *input, float *output, const float *kernel,
                    int width, int height, int kernel_size) {
  int i, j, k, l;
  float sum;
  int half_kernel_size = kernel_size / 2;
  for (i = 0; i < height; i++) {
    for (j = 0; j < width; j++) {
      sum = 0.0f;
      for (k = 0; k < kernel_size; k++) {
        for (l = 0; l < kernel_size; l++) {
          int input_i = i + k - half_kernel_size;
          int input_j = j + l - half_kernel_size;
          if (input_i >= 0 && input_i < height && input_j >= 0 &&
              input_j < width) {
            sum +=
                input[input_i * width + input_j] * kernel[k * kernel_size + l];
          }
        }
      }
      output[i * width + j] = sum;
    }
  }
}

__global__ void convolution_2d_gpu_kernel(const float *input, float *output,
                                          const float *kernel, int width,
                                          int height, int kernel_size) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  if (i < width && j < height) {
    float sum = 0.0f;
    int half_kernel_size = kernel_size / 2;
    for (int k = 0; k < kernel_size; k++) {
      for (int l = 0; l < kernel_size; l++) {
        int input_i = i + k - half_kernel_size;
        int input_j = j + l - half_kernel_size;
        if (input_i >= 0 && input_i < width && input_j >= 0 &&
            input_j < height) {
          sum += input[input_i * width + input_j] * kernel[k * kernel_size + l];
        }
      }
    }
    output[i * width + j] = sum;
  }
}

void convolution_2d_gpu(const float *input, float *output, const float *kernel,
                        int width, int height, int kernel_size) {
  size_t size = width * height * sizeof(float);
  float *input_d, *output_d, *kernel_d;
  cudaMalloc((void **)&input_d, size);
  cudaMalloc((void **)&output_d, size);
  cudaMalloc((void **)&kernel_d, size);
  cudaMemcpy(input_d, input, size, cudaMemcpyHostToDevice);
  cudaMemcpy(kernel_d, kernel, size, cudaMemcpyHostToDevice);
  dim3 threadsPerBlock(16, 16);
  dim3 numBlocks((width + threadsPerBlock.x - 1) / threadsPerBlock.x,
                 (height + threadsPerBlock.y - 1) / threadsPerBlock.y);
  convolution_2d_gpu_kernel<<<numBlocks, threadsPerBlock>>>(
      input_d, output_d, kernel_d, width, height, kernel_size);
  cudaMemcpy(output, output_d, size, cudaMemcpyDeviceToHost);
  cudaFree(input_d);
  cudaFree(output_d);
  cudaFree(kernel_d);
}

__global__ void convolution_2d_constmem_kernel(const float *input,
                                               float *output, int width,
                                               int height, int kernel_size) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  if (i < width && j < height) {
    float sum = 0.0f;
    int half_kernel_size = kernel_size / 2;
    for (int k = 0; k < kernel_size; k++) {
      for (int l = 0; l < kernel_size; l++) {
        int input_i = i + k - half_kernel_size;
        int input_j = j + l - half_kernel_size;
        if (input_i >= 0 && input_i < width && input_j >= 0 &&
            input_j < height) {
          // Access kernel from constant memory symbol
          sum +=
              input[input_i * width + input_j] * d_kernel[k * kernel_size + l];
        }
      }
    }
    output[i * width + j] = sum;
  }
}

void convolution_2d_gpu_constmem(const float *input, float *output,
                                 const float *kernel, int width, int height,
                                 int kernel_size) {
  // Copy kernel to constant memory symbol
  cudaMemcpyToSymbol(d_kernel, kernel,
                     kernel_size * kernel_size * sizeof(float));

  size_t size = width * height * sizeof(float);
  float *input_d, *output_d;
  cudaMalloc((void **)&input_d, size);
  cudaMalloc((void **)&output_d, size);
  cudaMemcpy(input_d, input, size, cudaMemcpyHostToDevice);

  dim3 threadsPerBlock(16, 16);
  dim3 numBlocks((width + threadsPerBlock.x - 1) / threadsPerBlock.x,
                 (height + threadsPerBlock.y - 1) / threadsPerBlock.y);
  convolution_2d_constmem_kernel<<<numBlocks, threadsPerBlock>>>(
      input_d, output_d, width, height, kernel_size);
  cudaMemcpy(output, output_d, size, cudaMemcpyDeviceToHost);
  cudaFree(input_d);
  cudaFree(output_d);
}