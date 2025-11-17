#include "convolution.h"
#include <cuda_runtime.h>
#include <stdexcept>
#include <string>

// CUDA error checking macro
#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t err = call;                                                    \
    if (err != cudaSuccess) {                                                  \
      throw std::runtime_error(std::string("CUDA error at ") +  std::string(__FILE__) + \
                               ":" + std::to_string(__LINE__) + " - " +        \
                               std::string(cudaGetErrorString(err)));          \
    }                                                                          \
  } while (0)

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
  // col = x (width/columns), row = y (height/rows)
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  if (col < width && row < height) {
    float sum = 0.0f;
    int half_kernel_size = kernel_size / 2;
    for (int ky = 0; ky < kernel_size; ky++) {
      for (int kx = 0; kx < kernel_size; kx++) {
        int in_row = row + ky - half_kernel_size;
        int in_col = col + kx - half_kernel_size;
        if (in_row >= 0 && in_row < height && in_col >= 0 && in_col < width) {
          sum += input[in_row * width + in_col] * kernel[ky * kernel_size + kx];
        }
      }
    }
    output[row * width + col] = sum;
  }
}

void convolution_2d_gpu(const float *input, float *output, const float *kernel,
                        int width, int height, int kernel_size) {
  size_t input_size = width * height * sizeof(float);
  size_t kernel_mem_size = kernel_size * kernel_size * sizeof(float);
  
  float *input_d, *output_d, *kernel_d;
  
  // Allocate device memory with error checking
  CUDA_CHECK(cudaMalloc((void **)&input_d, input_size));
  CUDA_CHECK(cudaMalloc((void **)&output_d, input_size));
  CUDA_CHECK(cudaMalloc((void **)&kernel_d, kernel_mem_size));
  
  // Copy data to device
  CUDA_CHECK(cudaMemcpy(input_d, input, input_size, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(kernel_d, kernel, kernel_mem_size, cudaMemcpyHostToDevice));
  
  // Launch kernel
  dim3 threadsPerBlock(16, 16);
  dim3 numBlocks((width + threadsPerBlock.x - 1) / threadsPerBlock.x,
                 (height + threadsPerBlock.y - 1) / threadsPerBlock.y);
  convolution_2d_gpu_kernel<<<numBlocks, threadsPerBlock>>>(
      input_d, output_d, kernel_d, width, height, kernel_size);
  
  // Check for kernel launch errors
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
  
  // Copy result back
  CUDA_CHECK(cudaMemcpy(output, output_d, input_size, cudaMemcpyDeviceToHost));
  
  // Free device memory
  CUDA_CHECK(cudaFree(input_d));
  CUDA_CHECK(cudaFree(output_d));
  CUDA_CHECK(cudaFree(kernel_d));
}

__global__ void convolution_2d_constmem_kernel(const float *input,
                                               float *output, int width,
                                               int height, int kernel_size) {
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  if (col < width && row < height) {
    float sum = 0.0f;
    int half_kernel_size = kernel_size / 2;
    for (int ky = 0; ky < kernel_size; ky++) {
      for (int kx = 0; kx < kernel_size; kx++) {
        int in_row = row + ky - half_kernel_size;
        int in_col = col + kx - half_kernel_size;
        if (in_row >= 0 && in_row < height && in_col >= 0 && in_col < width) {
          sum += input[in_row * width + in_col] * d_kernel[ky * kernel_size + kx];
        }
      }
    }
    output[row * width + col] = sum;
  }
}

void convolution_2d_gpu_constmem(const float *input, float *output,
                                 const float *kernel, int width, int height,
                                 int kernel_size) {
  // Validate kernel size for constant memory
  if (kernel_size > MAX_KERNEL_SIZE) {
    throw std::runtime_error("Kernel size " + std::to_string(kernel_size) + 
                             " exceeds maximum " + std::to_string(MAX_KERNEL_SIZE));
  }
  
  // Copy kernel to constant memory symbol
  CUDA_CHECK(cudaMemcpyToSymbol(d_kernel, kernel,
                     kernel_size * kernel_size * sizeof(float)));

  size_t input_size = width * height * sizeof(float);
  float *input_d, *output_d;
  
  // Allocate device memory with error checking
  CUDA_CHECK(cudaMalloc((void **)&input_d, input_size));
  CUDA_CHECK(cudaMalloc((void **)&output_d, input_size));
  
  // Copy input to device
  CUDA_CHECK(cudaMemcpy(input_d, input, input_size, cudaMemcpyHostToDevice));

  // Launch kernel
  dim3 threadsPerBlock(16, 16);
  dim3 numBlocks((width + threadsPerBlock.x - 1) / threadsPerBlock.x,
                 (height + threadsPerBlock.y - 1) / threadsPerBlock.y);
  convolution_2d_constmem_kernel<<<numBlocks, threadsPerBlock>>>(
      input_d, output_d, width, height, kernel_size);
  
  // Check for kernel launch errors
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
  
  // Copy result back
  CUDA_CHECK(cudaMemcpy(output, output_d, input_size, cudaMemcpyDeviceToHost));
  
  // Free device memory
  CUDA_CHECK(cudaFree(input_d));
  CUDA_CHECK(cudaFree(output_d));
}

// ---------------------------------------------------------------------------
// Shared-memory + constant-memory optimized CUDA convolution kernel.
// Each block processes a BLOCK_SIZE x BLOCK_SIZE output tile. Threads
// cooperatively load a (BLOCK_SIZE + 2*R) halo-extended tile into shared
// memory (where R = kernel_size/2). Kernel coefficients are stored in constant
// memory so all threads in a warp reading the same coefficient benefit from
// broadcast. Out-of-range global pixels are treated as zero (implicit zero
// padding).
// ---------------------------------------------------------------------------

// Removed CONV_BLOCK_SIZE macro; kernel now derives block size from blockDim.

__global__ void convolution_2d_const_shared_kernel(const float *input,
                                                   float *output, int width,
                                                   int height,
                                                   int kernel_size) {
  const int R = kernel_size / 2; // radius
  // Infer block size (assume square blocks for current implementation)
  const int BLOCK_SIZE_X = blockDim.x;
  const int BLOCK_SIZE_Y = blockDim.y;
  // For non-square blocks, we could generalize; currently we rely on square.
  const int BLOCK_SIZE = BLOCK_SIZE_X; // used for x/y tile origin math
  // Shared memory tile dimension (square) including halo. If non-square support
  // is desired, split into sm_w / sm_h.
  const int sm_dim = BLOCK_SIZE + 2 * R;

  extern __shared__ float smem[]; // size: sm_dim * sm_dim
  // Use linear indexing helper for 2D access (avoids non-constant VLAs)
  auto smem_at = [&](int y, int x) -> float& { return smem[y * sm_dim + x]; };

  // Global coordinates this thread is responsible for (output pixel)
  const int out_col = blockIdx.x * BLOCK_SIZE_X + threadIdx.x;
  const int out_row = blockIdx.y * BLOCK_SIZE_Y + threadIdx.y;

  // Cooperative loading of shared memory tile (with halo)
  const int tid = threadIdx.y * blockDim.x + threadIdx.x;
  const int nThreads = blockDim.x * blockDim.y;
  const int sm_size = sm_dim * sm_dim;
  for (int idx = tid; idx < sm_size; idx += nThreads) {
    int local_y = idx / sm_dim;        // [0, sm_dim)
    int local_x = idx % sm_dim;        // [0, sm_dim)
    int global_col = blockIdx.x * BLOCK_SIZE_X + local_x - R;
    int global_row = blockIdx.y * BLOCK_SIZE_Y + local_y - R;
    float val = 0.0f;
    if (global_col >= 0 && global_col < width && global_row >= 0 && global_row < height) {
      val = input[global_row * width + global_col];
    }
  smem_at(local_y, local_x) = val;
  }

  __syncthreads();

  // Guard threads outside output bounds
  if (out_col >= width || out_row >= height) {
    return;
  }

  // Local coordinates inside shared memory corresponding to (out_x, out_y)
  const int sm_x = threadIdx.x + R;
  const int sm_y = threadIdx.y + R;

  float sum = 0.0f;
  // Convolution accumulation using 2D shared memory tile and constant kernel
  for (int ky = 0; ky < kernel_size; ++ky) {
    int k_offset = ky * kernel_size;
    int sm_y_row = sm_y + ky - R;
    for (int kx = 0; kx < kernel_size; ++kx) {
      int sm_x_col = sm_x + kx - R;
  sum += smem_at(sm_y_row, sm_x_col) * d_kernel[k_offset + kx];
    }
  }

  output[out_row * width + out_col] = sum;
}

void convolution_2d_gpu_const_shared(const float *input, float *output,
                                     const float *kernel, int width,
                                     int height, int kernel_size) {
  if (kernel_size > MAX_KERNEL_SIZE) {
    throw std::runtime_error("Kernel size " + std::to_string(kernel_size) +
                             " exceeds maximum " + std::to_string(MAX_KERNEL_SIZE));
  }

  // Copy kernel to constant memory (reuse existing symbol)
  CUDA_CHECK(cudaMemcpyToSymbol(d_kernel, kernel,
                                kernel_size * kernel_size * sizeof(float)));

  size_t num_pixels = static_cast<size_t>(width) * height;
  size_t input_size = num_pixels * sizeof(float);

  float *input_d = nullptr;
  float *output_d = nullptr;
  CUDA_CHECK(cudaMalloc((void **)&input_d, input_size));
  CUDA_CHECK(cudaMalloc((void **)&output_d, input_size));
  CUDA_CHECK(cudaMemcpy(input_d, input, input_size, cudaMemcpyHostToDevice));

  // Default block size (square). Could be tuned or passed as parameter later.
  const int BLOCK_SIZE = 16;
  dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
  dim3 numBlocks((width + BLOCK_SIZE - 1) / BLOCK_SIZE,
                 (height + BLOCK_SIZE - 1) / BLOCK_SIZE);

  const int R = kernel_size / 2;
  int sm_dim = BLOCK_SIZE + 2 * R;
  size_t shared_bytes = static_cast<size_t>(sm_dim) * sm_dim * sizeof(float);

  convolution_2d_const_shared_kernel<<<numBlocks, threadsPerBlock, shared_bytes>>>(
      input_d, output_d, width, height, kernel_size);

  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  CUDA_CHECK(cudaMemcpy(output, output_d, input_size, cudaMemcpyDeviceToHost));

  CUDA_CHECK(cudaFree(input_d));
  CUDA_CHECK(cudaFree(output_d));
}

// (Templated interface removed; only direct variant functions remain.)