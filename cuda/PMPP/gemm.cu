#include "gemm.h"

__global__ void gemm_kernel(const float *A, const float *B, float *C, int M,
                            int N, int K)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    int row = idx / N;
    int col = idx % N;

    if (row < M && col < N)
    {
        float sum = 0.0f;
        for (int i = 0; i < K; ++i)
        {
            sum += A[row * K + i] * B[i * N + col];
        }
        C[row * N + col] = sum;
    }
}

void gpu_gemm(const float *A, const float *B, float *C, int M, int N, int K)
{
    int totalThreads = M * N;
    int blockSize = 256;
    int numBlocks = (totalThreads + blockSize - 1) / blockSize;

    float *d_A, *d_B, *d_C;
    size_t sizeA = M * K * sizeof(float);
    size_t sizeB = K * N * sizeof(float);
    size_t sizeC = M * N * sizeof(float);
    cudaMalloc(&d_A, sizeA);
    cudaMalloc(&d_B, sizeB);
    cudaMalloc(&d_C, sizeC);

    cudaMemcpy(d_A, A, sizeA, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, sizeB, cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, C, sizeC, cudaMemcpyHostToDevice);

    gemm_kernel<<<numBlocks, blockSize>>>(d_A, d_B, d_C, M, N, K);
    cudaDeviceSynchronize();

    cudaMemcpy(C, d_C, sizeC, cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}