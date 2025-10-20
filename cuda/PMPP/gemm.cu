#include "gemm.h"
#include <cuda_runtime.h>
#include <stdexcept>

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

// Simple matrix transpose kernel (renamed from transpoe_kernel -> transpose_kernel)
__global__ void transpose_kernel(const float *input, float *output, int rows,
                                 int cols)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int row = idx / cols;
    int col = idx % cols;
    if (row < rows && col < cols)
    {
        output[col * rows + row] = input[row * cols + col];
    }
}

// // Host wrapper to launch transpose kernel (optional utility)
// void gpu_transpose(const float *input, float *output, int rows, int cols)
// {
//     int total = rows * cols;
//     int blockSize = 256;
//     int gridSize = (total + blockSize - 1) / blockSize;
//     transpose_kernel<<<gridSize, blockSize>>>(input, output, rows, cols);
//     cudaDeviceSynchronize();
// }

void gpu_gemm_tiled(const float *A, const float *B, float *C, int M, int N,
                    int K, int tileSize)
{

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
    cudaDeviceSynchronize();

    int gridX = (N + tileSize - 1) / tileSize;
    int gridY = (M + tileSize - 1) / tileSize;

    dim3 gridDim(gridX, gridY);
    dim3 blockDim(tileSize, tileSize);

    switch (tileSize)
    {
    case 16:
        gemm_tiled_kernel<16><<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K);
        break;
    case 32:
        gemm_tiled_kernel<32><<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K);
        break;
    default:
        // Handle unsupported tile sizes
        throw std::invalid_argument("Unsupported tile size");
    }

    cudaDeviceSynchronize();

    cudaMemcpy(C, d_C, sizeC, cudaMemcpyDeviceToHost);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

template <int TILE>
__global__ void gemm_tiled_kernel(
    const float *__restrict__ A, // M×K
    const float *__restrict__ B, // K×N
    float *__restrict__ C,       // M×N (output)
    int M, int N, int K)
{
    // Global row/col this thread computes in C
    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;

    // Shared tiles
    __shared__ float sA[TILE][TILE];
    __shared__ float sB[TILE][TILE];

    float acc = 0.0f;

    // Number of K-tiles
    int numTiles = (K + TILE - 1) / TILE;
    for (int t = 0; t < numTiles; ++t)
    {
        // Indices along K to load
        int aCol = t * TILE + threadIdx.x; // A column (K dimension)
        int bRow = t * TILE + threadIdx.y; // B row    (K dimension)

        // Cooperative loads (guarded)
        sA[threadIdx.y][threadIdx.x] =
            (row < M && aCol < K) ? A[row * K + aCol] : 0.0f;

        sB[threadIdx.y][threadIdx.x] =
            (bRow < K && col < N) ? B[bRow * N + col] : 0.0f;

        __syncthreads();

        // Accumulate this tile's contribution to C(row, col)
        for (int k = 0; k < TILE; ++k)
        {
            acc += sA[threadIdx.y][k] * sB[k][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < M && col < N)
    {
        C[row * N + col] = acc;
    }
}

template __global__ void gemm_tiled_kernel<16>(
    const float *__restrict__ A,
    const float *__restrict__ B,
    float *__restrict__ C,
    int M, int N, int K);
template __global__ void gemm_tiled_kernel<32>(
    const float *__restrict__ A,
    const float *__restrict__ B,
    float *__restrict__ C,
    int M, int N, int K);