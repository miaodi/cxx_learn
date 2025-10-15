#include "vector_add.h"
__global__ void gpu_vector_add_kernel(const float *A, const float *B, float *C, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N)
    {
        C[idx] = A[idx] + B[idx];
    }
}

void gpu_vector_add(const float *A, const float *B, float *C, int N)
{
    int blockSize = 256;
    int numBlocks = (N + blockSize - 1) / blockSize;

    // copy to device memory
    float *d_A, *d_B, *d_C;
    cudaMalloc((void **)&d_A, N * sizeof(float));
    cudaMalloc((void **)&d_B, N * sizeof(float));
    cudaMalloc((void **)&d_C, N * sizeof(float));

    // copy to device memory
    cudaMemcpy(d_A, A, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, N * sizeof(float), cudaMemcpyHostToDevice);

    gpu_vector_add_kernel<<<numBlocks, blockSize>>>(d_A, d_B, d_C, N);
    cudaDeviceSynchronize();

    // copy back to host memory
    cudaMemcpy(C, d_C, N * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

void cpu_vector_add(const float *A, const float *B, float *C, int N)
{
#pragma omp parallel for
    for (int i = 0; i < N; ++i)
    {
        C[i] = A[i] + B[i];
    }
}