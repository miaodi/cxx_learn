#include "vector_add.h"
#include <chrono>
#include <cmath>
#include <gtest/gtest.h>
#include <iomanip>
#include <random>
#include <vector>
#include "gemm.h"
#include "convolution.h"
#include "reduction.h"

// ------------------------------ Convolution Tests ------------------------------
class ConvolutionTest : public ::testing::Test {
protected:
  void SetUp() override {
    rng.seed(2025);
    dist = std::uniform_real_distribution<float>(-1.0f, 1.0f);
  }

  void generateRandomImage(std::vector<float> &img, int width, int height) {
    img.resize(width * height);
    for (int i = 0; i < width * height; ++i) img[i] = dist(rng);
  }

  void generateRandomKernel(std::vector<float> &kernel, int k) {
    kernel.resize(k * k);
    for (int i = 0; i < k * k; ++i) kernel[i] = dist(rng);
  }

  void verifyConvolution(const std::vector<float> &ref, const std::vector<float> &got,
                         float tol = 1e-4f) {
    ASSERT_EQ(ref.size(), got.size());
    for (size_t i = 0; i < ref.size(); ++i) {
      EXPECT_NEAR(ref[i], got[i], tol) << "Convolution mismatch at index " << i;
    }
  }

  std::mt19937 rng;
  std::uniform_real_distribution<float> dist;
};

// Small 3x3 kernel test on 32x32 image
TEST_F(ConvolutionTest, SmallKernel3_Image32x32) {
  const int W = 32, H = 32, K = 3;
  std::vector<float> input, kernel, cpu_out(W * H), gpu_out(W * H), gpu_const(W * H), gpu_shared(W * H);
  generateRandomImage(input, W, H);
  generateRandomKernel(kernel, K);
  convolution_2d(input.data(), cpu_out.data(), kernel.data(), W, H, K);
  convolution_2d_gpu(input.data(), gpu_out.data(), kernel.data(), W, H, K);
  convolution_2d_gpu_constmem(input.data(), gpu_const.data(), kernel.data(), W, H, K);
  convolution_2d_gpu_const_shared(input.data(), gpu_shared.data(), kernel.data(), W, H, K);
  verifyConvolution(cpu_out, gpu_out);
  verifyConvolution(cpu_out, gpu_const);
  verifyConvolution(cpu_out, gpu_shared);
}

// Rectangular image with 5x5 kernel
TEST_F(ConvolutionTest, Kernel5_RectangularImage) {
  const int W = 48, H = 37, K = 5;
  std::vector<float> input, kernel, cpu_out(W * H), gpu_out(W * H), gpu_const(W * H), gpu_shared(W * H);
  generateRandomImage(input, W, H);
  generateRandomKernel(kernel, K);
  convolution_2d(input.data(), cpu_out.data(), kernel.data(), W, H, K);
  convolution_2d_gpu(input.data(), gpu_out.data(), kernel.data(), W, H, K);
  convolution_2d_gpu_constmem(input.data(), gpu_const.data(), kernel.data(), W, H, K);
  convolution_2d_gpu_const_shared(input.data(), gpu_shared.data(), kernel.data(), W, H, K);
  verifyConvolution(cpu_out, gpu_out);
  verifyConvolution(cpu_out, gpu_const);
  verifyConvolution(cpu_out, gpu_shared);
}

// Edge handling with 7x7 kernel on odd-sized image
TEST_F(ConvolutionTest, Kernel7_OddImageSizes) {
  const int W = 53, H = 50, K = 7;
  std::vector<float> input, kernel, cpu_out(W * H), gpu_out(W * H), gpu_const(W * H), gpu_shared(W * H);
  generateRandomImage(input, W, H);
  generateRandomKernel(kernel, K);
  convolution_2d(input.data(), cpu_out.data(), kernel.data(), W, H, K);
  convolution_2d_gpu(input.data(), gpu_out.data(), kernel.data(), W, H, K);
  convolution_2d_gpu_constmem(input.data(), gpu_const.data(), kernel.data(), W, H, K);
  convolution_2d_gpu_const_shared(input.data(), gpu_shared.data(), kernel.data(), W, H, K);
  verifyConvolution(cpu_out, gpu_out, 2e-4f); // slightly relaxed tolerance for larger kernel
  verifyConvolution(cpu_out, gpu_const, 2e-4f);
  verifyConvolution(cpu_out, gpu_shared, 2e-4f);
}

// Max kernel size (11x11) on moderate image
TEST_F(ConvolutionTest, MaxKernel11_Image64x64) {
  const int W = 64, H = 64, K = 11;
  std::vector<float> input, kernel, cpu_out(W * H), gpu_out(W * H), gpu_const(W * H), gpu_shared(W * H);
  generateRandomImage(input, W, H);
  generateRandomKernel(kernel, K);
  convolution_2d(input.data(), cpu_out.data(), kernel.data(), W, H, K);
  convolution_2d_gpu(input.data(), gpu_out.data(), kernel.data(), W, H, K);
  convolution_2d_gpu_constmem(input.data(), gpu_const.data(), kernel.data(), W, H, K);
  convolution_2d_gpu_const_shared(input.data(), gpu_shared.data(), kernel.data(), W, H, K);
  verifyConvolution(cpu_out, gpu_out, 5e-4f);
  verifyConvolution(cpu_out, gpu_const, 5e-4f);
  verifyConvolution(cpu_out, gpu_shared, 5e-4f);
}

// Zero padding correctness: all-ones input and kernel, check border vs interior
TEST_F(ConvolutionTest, ZeroPaddingAllOnes) {
  const int W = 16, H = 16, K = 7; // radius=3
  std::vector<float> input(W * H, 1.0f), kernel(K * K, 1.0f);
  std::vector<float> cpu_out(W * H), gpu_shared(W * H);
  convolution_2d(input.data(), cpu_out.data(), kernel.data(), W, H, K);
  convolution_2d_gpu_const_shared(input.data(), gpu_shared.data(), kernel.data(), W, H, K);
  verifyConvolution(cpu_out, gpu_shared);
  // Interior pixel (not near border) should equal K*K
  int center = (H/2) * W + (W/2);
  EXPECT_FLOAT_EQ(cpu_out[center], static_cast<float>(K*K));
}

// Compare GPU constmem vs shared variant directly on random data
TEST_F(ConvolutionTest, ConstMemVsSharedConsistency) {
  const int W = 40, H = 40, K = 5;
  std::vector<float> input, kernel, gpu_const(W * H), gpu_shared(W * H);
  generateRandomImage(input, W, H);
  generateRandomKernel(kernel, K);
  convolution_2d_gpu_constmem(input.data(), gpu_const.data(), kernel.data(), W, H, K);
  convolution_2d_gpu_const_shared(input.data(), gpu_shared.data(), kernel.data(), W, H, K);
  verifyConvolution(gpu_const, gpu_shared, 1e-4f);
}

class VectorAddTest : public ::testing::Test {
protected:
  void SetUp() override {
    // Initialize random number generator with fixed seed for reproducibility
    rng.seed(42);
    dist = std::uniform_real_distribution<float>(-1000.0f, 1000.0f);
  }

  void TearDown() override {
    // Cleanup if needed
  }

  // Helper function to generate random data
  void generateRandomData(std::vector<float> &vec, size_t size) {
    vec.resize(size);
    for (size_t i = 0; i < size; ++i) {
      vec[i] = dist(rng);
    }
  }

  // Helper function to verify results with tolerance
  void verifyResults(const std::vector<float> &expected,
                     const std::vector<float> &actual,
                     float tolerance = 1e-6f) {
    ASSERT_EQ(expected.size(), actual.size());
    for (size_t i = 0; i < expected.size(); ++i) {
      EXPECT_NEAR(expected[i], actual[i], tolerance)
          << "Mismatch at index " << i;
    }
  }

  // Helper function to compute reference result
  void computeReference(const std::vector<float> &A,
                        const std::vector<float> &B, std::vector<float> &C) {
    C.resize(A.size());
    for (size_t i = 0; i < A.size(); ++i) {
      C[i] = A[i] + B[i];
    }
  }

  std::mt19937 rng;
  std::uniform_real_distribution<float> dist;
};

// Test medium sized vectors with random data
TEST_F(VectorAddTest, RandomDataMediumCPU) {
  const int N = 1000;
  std::vector<float> A, B, C(N), expected;

  generateRandomData(A, N);
  generateRandomData(B, N);
  computeReference(A, B, expected);

  cpu_vector_add(A.data(), B.data(), C.data(), N);

  verifyResults(expected, C);
}

TEST_F(VectorAddTest, RandomDataMediumGPU) {
  const int N = 1000;
  std::vector<float> A, B, C(N), expected;

  generateRandomData(A, N);
  generateRandomData(B, N);
  computeReference(A, B, expected);

  gpu_vector_add(A.data(), B.data(), C.data(), N);

  verifyResults(expected, C);
}

// Test large vectors
TEST_F(VectorAddTest, LargeVectorCPU) {
  const int N = 1000000;
  std::vector<float> A, B, C(N), expected;

  generateRandomData(A, N);
  generateRandomData(B, N);
  computeReference(A, B, expected);

  cpu_vector_add(A.data(), B.data(), C.data(), N);

  verifyResults(expected, C);
}

TEST_F(VectorAddTest, LargeVectorGPU) {
  const int N = 1000000;
  std::vector<float> A, B, C(N), expected;

  generateRandomData(A, N);
  generateRandomData(B, N);
  computeReference(A, B, expected);

  gpu_vector_add(A.data(), B.data(), C.data(), N);

  verifyResults(expected, C);
}

// Test CPU vs GPU consistency
TEST_F(VectorAddTest, CPUvsGPUConsistency) {
  const int N = 10000;
  std::vector<float> A, B, C_cpu(N), C_gpu(N);

  generateRandomData(A, N);
  generateRandomData(B, N);

  cpu_vector_add(A.data(), B.data(), C_cpu.data(), N);
  gpu_vector_add(A.data(), B.data(), C_gpu.data(), N);

  verifyResults(C_cpu, C_gpu);
}

// Test different vector sizes that test threading boundaries
TEST_F(VectorAddTest, ThreadingBoundariesCPU) {
  std::vector<int> sizes = {255, 256, 257, 511, 512, 513, 1023, 1024, 1025};

  for (int N : sizes) {
    std::vector<float> A, B, C(N), expected;

    generateRandomData(A, N);
    generateRandomData(B, N);
    computeReference(A, B, expected);

    cpu_vector_add(A.data(), B.data(), C.data(), N);

    verifyResults(expected, C);
  }
}

TEST_F(VectorAddTest, ThreadingBoundariesGPU) {
  std::vector<int> sizes = {255, 256, 257, 511, 512, 513, 1023, 1024, 1025};

  for (int N : sizes) {
    std::vector<float> A, B, C(N), expected;

    generateRandomData(A, N);
    generateRandomData(B, N);
    computeReference(A, B, expected);

    gpu_vector_add(A.data(), B.data(), C.data(), N);

    verifyResults(expected, C);
  }
}

// GEMM Test Class
class GEMMTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Initialize random number generator with fixed seed for reproducibility
        rng.seed(12345);
        dist = std::uniform_real_distribution<float>(-10.0f, 10.0f);
    }

    void TearDown() override {
        // Cleanup if needed
    }

    // Helper function to generate random matrix data
    void generateRandomMatrix(std::vector<float>& matrix, int rows, int cols) {
        matrix.resize(rows * cols);
        for (int i = 0; i < rows * cols; ++i) {
            matrix[i] = dist(rng);
        }
    }

    // Helper function to initialize matrix with specific pattern
    void initializeMatrix(std::vector<float>& matrix, int rows, int cols, float value) {
        matrix.resize(rows * cols);
        std::fill(matrix.begin(), matrix.end(), value);
    }

    // Helper function to initialize identity matrix
    void initializeIdentityMatrix(std::vector<float>& matrix, int size) {
        matrix.resize(size * size);
        std::fill(matrix.begin(), matrix.end(), 0.0f);
        for (int i = 0; i < size; ++i) {
            matrix[i * size + i] = 1.0f;
        }
    }

    // Reference CPU GEMM implementation for verification
    void cpu_gemm_reference(const std::vector<float>& A, const std::vector<float>& B, 
                           std::vector<float>& C, int M, int N, int K) {
        C.resize(M * N);
        std::fill(C.begin(), C.end(), 0.0f);
        
        for (int m = 0; m < M; ++m) {
            for (int n = 0; n < N; ++n) {
                float sum = 0.0f;
                for (int k = 0; k < K; ++k) {
                    sum += A[m * K + k] * B[k * N + n];
                }
                C[m * N + n] = sum;
            }
        }
    }

    // Helper function to verify results with tolerance
    void verifyGEMMResults(const std::vector<float>& expected, 
                          const std::vector<float>& actual, 
                          float tolerance = 1e-4f) {
        ASSERT_EQ(expected.size(), actual.size());
        for (size_t i = 0; i < expected.size(); ++i) {
            EXPECT_NEAR(expected[i], actual[i], tolerance) 
                << "Mismatch at index " << i;
        }
    }

    std::mt19937 rng;
    std::uniform_real_distribution<float> dist;
};

// Test basic GEMM functionality with small matrices
TEST_F(GEMMTest, SmallMatrices) {
    const int M = 4, N = 3, K = 2;
    
    // A = [[1, 2], [3, 4], [5, 6], [7, 8]]  (4x2)
    std::vector<float> A = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
    
    // B = [[1, 2, 3], [4, 5, 6]]  (2x3)
    std::vector<float> B = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    
    std::vector<float> C_gpu(M * N);
    std::vector<float> C_expected;
    
    // Compute reference result
    cpu_gemm_reference(A, B, C_expected, M, N, K);
    
    // Compute GPU result
    gpu_gemm(A.data(), B.data(), C_gpu.data(), M, N, K);
    
    verifyGEMMResults(C_expected, C_gpu);
}

// Test tiled GEMM (tile size 16 and 32) correctness against CPU reference
TEST_F(GEMMTest, TiledGEMM_16_32) {
  const int M = 64, N = 48, K = 32; // Non-square to exercise indexing
  std::vector<float> A, B, C_ref, C_tiled(M * N);
  generateRandomMatrix(A, M, K);
  generateRandomMatrix(B, K, N);
  cpu_gemm_reference(A, B, C_ref, M, N, K);

  // Tile size 16
  std::fill(C_tiled.begin(), C_tiled.end(), 0.0f);
  gpu_gemm_tiled(A.data(), B.data(), C_tiled.data(), M, N, K, 16);
  verifyGEMMResults(C_ref, C_tiled, 1e-3f);

  // Tile size 32 (works even if dimensions not multiples due to guards)
  std::fill(C_tiled.begin(), C_tiled.end(), 0.0f);
  gpu_gemm_tiled(A.data(), B.data(), C_tiled.data(), M, N, K, 32);
  verifyGEMMResults(C_ref, C_tiled, 1e-3f);
}

// Test square matrices
TEST_F(GEMMTest, SquareMatrices) {
    const int N = 8;  // 8x8 matrices
    
    std::vector<float> A, B, C_gpu(N * N), C_expected;
    
    generateRandomMatrix(A, N, N);
    generateRandomMatrix(B, N, N);
    
    cpu_gemm_reference(A, B, C_expected, N, N, N);
    gpu_gemm(A.data(), B.data(), C_gpu.data(), N, N, N);
    
    verifyGEMMResults(C_expected, C_gpu);
}

// Test identity matrix multiplication
TEST_F(GEMMTest, IdentityMatrix) {
    const int N = 5;
    
    std::vector<float> A, I, C_gpu(N * N), C_expected;
    
    generateRandomMatrix(A, N, N);
    initializeIdentityMatrix(I, N);
    
    // A * I should equal A
    cpu_gemm_reference(A, I, C_expected, N, N, N);
    gpu_gemm(A.data(), I.data(), C_gpu.data(), N, N, N);
    
    verifyGEMMResults(C_expected, C_gpu);
    
    // Also verify that result equals original matrix A
    verifyGEMMResults(A, C_gpu);
}

// Test zero matrices
TEST_F(GEMMTest, ZeroMatrices) {
    const int M = 6, N = 4, K = 3;
    
    std::vector<float> A, B, C_gpu(M * N), C_expected;
    
    initializeMatrix(A, M, K, 0.0f);  // Zero matrix
    generateRandomMatrix(B, K, N);
    
    cpu_gemm_reference(A, B, C_expected, M, N, K);
    gpu_gemm(A.data(), B.data(), C_gpu.data(), M, N, K);
    
    verifyGEMMResults(C_expected, C_gpu);
    
    // Result should be all zeros
    for (float val : C_gpu) {
        EXPECT_FLOAT_EQ(val, 0.0f);
    }
}

// Test single element matrices
TEST_F(GEMMTest, SingleElement) {
    const int M = 1, N = 1, K = 1;
    
    std::vector<float> A = {3.5f};
    std::vector<float> B = {2.0f};
    std::vector<float> C_gpu(1);
    std::vector<float> C_expected;
    
    cpu_gemm_reference(A, B, C_expected, M, N, K);
    gpu_gemm(A.data(), B.data(), C_gpu.data(), M, N, K);
    
    verifyGEMMResults(C_expected, C_gpu);
    EXPECT_FLOAT_EQ(C_gpu[0], 7.0f);  // 3.5 * 2.0 = 7.0
}

// Test rectangular matrices (tall and wide)
TEST_F(GEMMTest, RectangularMatrices) {
    // Test case 1: Tall matrix * Wide matrix
    const int M1 = 10, N1 = 3, K1 = 2;
    
    std::vector<float> A1, B1, C1_gpu(M1 * N1), C1_expected;
    
    generateRandomMatrix(A1, M1, K1);  // 10x2
    generateRandomMatrix(B1, K1, N1);  // 2x3
    
    cpu_gemm_reference(A1, B1, C1_expected, M1, N1, K1);
    gpu_gemm(A1.data(), B1.data(), C1_gpu.data(), M1, N1, K1);
    
    verifyGEMMResults(C1_expected, C1_gpu);
    
    // Test case 2: Wide matrix * Tall matrix
    const int M2 = 3, N2 = 10, K2 = 2;
    
    std::vector<float> A2, B2, C2_gpu(M2 * N2), C2_expected;
    
    generateRandomMatrix(A2, M2, K2);  // 3x2
    generateRandomMatrix(B2, K2, N2);  // 2x10
    
    cpu_gemm_reference(A2, B2, C2_expected, M2, N2, K2);
    gpu_gemm(A2.data(), B2.data(), C2_gpu.data(), M2, N2, K2);
    
    verifyGEMMResults(C2_expected, C2_gpu);
}

// Test with different block sizes (threading boundaries)
TEST_F(GEMMTest, ThreadingBoundaries) {
    std::vector<int> sizes = {15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 128, 129, 255, 256, 257};
    
    for (int size : sizes) {
        std::vector<float> A, B, C_gpu(size * size), C_expected;
        
        generateRandomMatrix(A, size, size);
        generateRandomMatrix(B, size, size);
        
        cpu_gemm_reference(A, B, C_expected, size, size, size);
        gpu_gemm(A.data(), B.data(), C_gpu.data(), size, size, size);
        
        verifyGEMMResults(C_expected, C_gpu, 1e-3f);  // Slightly larger tolerance for larger matrices
    }
}

// Test medium-sized matrices
TEST_F(GEMMTest, MediumMatrices) {
    const int M = 64, N = 48, K = 32;
    
    std::vector<float> A, B, C_gpu(M * N), C_expected;
    
    generateRandomMatrix(A, M, K);
    generateRandomMatrix(B, K, N);
    
    cpu_gemm_reference(A, B, C_expected, M, N, K);
    gpu_gemm(A.data(), B.data(), C_gpu.data(), M, N, K);
    
    verifyGEMMResults(C_expected, C_gpu, 1e-3f);
}

// Thread boundary style test for tiled kernel across sizes and tile options
TEST_F(GEMMTest, TiledGEMM_ThreadBoundaries) {
  // Sizes around tile boundaries; include some not divisible by tile
  struct Case { int M,N,K; int tile; }; 
  std::vector<Case> cases = {
    {15,15,15,16}, {16,16,16,16}, {17,17,17,16},
    {31,31,31,32}, {32,32,32,32}, {33,33,33,32},
    {63,48,32,16}, {64,64,48,16}, {65,64,48,16},
    {63,63,63,32}, {64,64,64,32}, {65,65,65,32}
  };
  for(const auto &c : cases) {
    std::vector<float> A,B,C_ref,C_tiled(c.M * c.N);
    generateRandomMatrix(A, c.M, c.K);
    generateRandomMatrix(B, c.K, c.N);
    cpu_gemm_reference(A,B,C_ref,c.M,c.N,c.K);
    std::fill(C_tiled.begin(), C_tiled.end(), 0.0f);
    gpu_gemm_tiled(A.data(), B.data(), C_tiled.data(), c.M, c.N, c.K, c.tile);
    verifyGEMMResults(C_ref, C_tiled, 1e-3f);
  }
}

// Test with special floating-point values
TEST_F(GEMMTest, SpecialFloatValues) {
    const int M = 3, N = 2, K = 2;
    
    // Matrix with special values
    std::vector<float> A = {
        1.0f, std::numeric_limits<float>::epsilon(),
        std::numeric_limits<float>::max() / 1e6f, 1.0f,
        -1.0f, 0.0f
    };
    
    std::vector<float> B = {
        2.0f, -2.0f,
        1.0f, 1.0f
    };
    
    std::vector<float> C_gpu(M * N), C_expected;
    
    cpu_gemm_reference(A, B, C_expected, M, N, K);
    gpu_gemm(A.data(), B.data(), C_gpu.data(), M, N, K);
    
    verifyGEMMResults(C_expected, C_gpu, 1e-5f);
}

// Test associativity: (A * B) * C vs A * (B * C) - Note: This tests numerical stability
TEST_F(GEMMTest, NumericalStability) {
    const int N = 16;  // Small size for manageable computation
    
    std::vector<float> A, B, C, AB(N * N), BC(N * N), AB_C(N * N), A_BC(N * N);
    std::vector<float> AB_expected, BC_expected, AB_C_expected, A_BC_expected;
    
    generateRandomMatrix(A, N, N);
    generateRandomMatrix(B, N, N);
    generateRandomMatrix(C, N, N);
    
    // Compute (A * B) * C
    gpu_gemm(A.data(), B.data(), AB.data(), N, N, N);
    gpu_gemm(AB.data(), C.data(), AB_C.data(), N, N, N);
    
    // Compute A * (B * C)
    gpu_gemm(B.data(), C.data(), BC.data(), N, N, N);
    gpu_gemm(A.data(), BC.data(), A_BC.data(), N, N, N);
    
    // Results should be close (within numerical precision)
    verifyGEMMResults(AB_C, A_BC, 1e-2f);  // Relaxed tolerance due to accumulated errors
}

// Performance test (disabled by default)
TEST_F(GEMMTest, DISABLED_PerformanceTest) {
    const int M = 512, N = 512, K = 512;
    const int num_runs = 5;
    
    std::vector<float> A, B, C_gpu(M * N);
    generateRandomMatrix(A, M, K);
    generateRandomMatrix(B, K, N);
    
    // Warm up
    gpu_gemm(A.data(), B.data(), C_gpu.data(), M, N, K);
    
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < num_runs; ++i) {
        gpu_gemm(A.data(), B.data(), C_gpu.data(), M, N, K);
    }
    auto end = std::chrono::high_resolution_clock::now();
    
    auto time_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    double gflops = (2.0 * M * N * K * num_runs) / (time_ms.count() * 1e6);  // GFLOPs
    
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "GEMM Performance Results (" << M << "x" << N << "x" << K << "):\n";
    std::cout << "Time: " << time_ms.count() / static_cast<double>(num_runs) << " ms per run\n";
    std::cout << "Performance: " << gflops << " GFLOPS\n";
    
    EXPECT_GT(gflops, 1.0);  // Expect at least 1 GFLOPS
}

// Large matrix test (disabled by default due to memory requirements)
TEST_F(GEMMTest, DISABLED_LargeMatrices) {
    const int M = 1024, N = 1024, K = 1024;
    
    std::vector<float> A, B, C_gpu(M * N);
    
    generateRandomMatrix(A, M, K);
    generateRandomMatrix(B, K, N);
    
    // Just test that it doesn't crash and produces reasonable results
    auto start = std::chrono::high_resolution_clock::now();
    gpu_gemm(A.data(), B.data(), C_gpu.data(), M, N, K);
    auto end = std::chrono::high_resolution_clock::now();
    
    auto time_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    std::cout << "Large GEMM (" << M << "x" << N << "x" << K << ") completed in " 
              << time_ms.count() << " ms\n";
    
    // Basic sanity check - result shouldn't be all zeros
    bool has_nonzero = false;
    for (float val : C_gpu) {
        if (std::abs(val) > 1e-6f) {
            has_nonzero = true;
            break;
        }
    }
    EXPECT_TRUE(has_nonzero);
}

// ------------------------------ Reduction Tests ------------------------------
class ReductionTest : public ::testing::Test {
protected:
  void SetUp() override {
    rng.seed(2025);
    dist = std::uniform_real_distribution<float>(-10.0f, 10.0f);
  }

  void generateRandomArray(std::vector<float> &arr, int size) {
    arr.resize(size);
    for (int i = 0; i < size; ++i) arr[i] = dist(rng);
  }

  float cpuReduction(const std::vector<float> &arr) {
    float sum = 0.0f;
    for (float val : arr) sum += val;
    return sum;
  }

  std::mt19937 rng;
  std::uniform_real_distribution<float> dist;
};

// Test with a small array
TEST_F(ReductionTest, SmallArray) {
  const int size = 100;
  std::vector<float> input;
  generateRandomArray(input, size);
  
  float cpu_result = cpuReduction(input);
  float gpu_result = simple_reduction(input.data(), size);
  
  EXPECT_NEAR(cpu_result, gpu_result, 1e-3f) 
    << "Reduction mismatch for size " << size;
}

// Test with array size equal to block size
TEST_F(ReductionTest, SingleBlockSize) {
  const int size = 256;
  std::vector<float> input;
  generateRandomArray(input, size);
  
  float cpu_result = cpuReduction(input);
  float gpu_result = simple_reduction(input.data(), size);
  
  EXPECT_NEAR(cpu_result, gpu_result, 1e-3f) 
    << "Reduction mismatch for size " << size;
}

// Test with medium array
TEST_F(ReductionTest, MediumArray) {
  const int size = 4096;
  std::vector<float> input;
  generateRandomArray(input, size);
  
  float cpu_result = cpuReduction(input);
  float gpu_result = simple_reduction(input.data(), size);
  
  EXPECT_NEAR(cpu_result, gpu_result, 1e-2f) 
    << "Reduction mismatch for size " << size;
}

// Test with large array
TEST_F(ReductionTest, LargeArray) {
  const int size = 1 << 20; // 1M elements
  std::vector<float> input;
  generateRandomArray(input, size);
  
  float cpu_result = cpuReduction(input);
  float gpu_result = simple_reduction(input.data(), size);
  
  float relative_error = std::abs(cpu_result - gpu_result) / std::abs(cpu_result);
  EXPECT_LT(relative_error, 1e-3f)  // Relaxed tolerance for large arrays due to floating-point accumulation
    << "Reduction mismatch for size " << size 
    << " (CPU: " << cpu_result << ", GPU: " << gpu_result << ")";
}

// Test with non-power-of-2 size
TEST_F(ReductionTest, NonPowerOfTwoSize) {
  const int size = 12345;
  std::vector<float> input;
  generateRandomArray(input, size);
  
  float cpu_result = cpuReduction(input);
  float gpu_result = simple_reduction(input.data(), size);
  
  EXPECT_NEAR(cpu_result, gpu_result, 1e-2f) 
    << "Reduction mismatch for size " << size;
}

// Test with all zeros
TEST_F(ReductionTest, AllZeros) {
  const int size = 1000;
  std::vector<float> input(size, 0.0f);
  
  float cpu_result = cpuReduction(input);
  float gpu_result = simple_reduction(input.data(), size);
  
  EXPECT_FLOAT_EQ(cpu_result, gpu_result);
  EXPECT_FLOAT_EQ(0.0f, gpu_result);
}

// Test with all ones
TEST_F(ReductionTest, AllOnes) {
  const int size = 1000;
  std::vector<float> input(size, 1.0f);
  
  float cpu_result = cpuReduction(input);
  float gpu_result = simple_reduction(input.data(), size);
  
  EXPECT_NEAR(cpu_result, gpu_result, 1e-3f);
  EXPECT_NEAR(static_cast<float>(size), gpu_result, 1e-3f);
}

// Test with alternating positive and negative values
TEST_F(ReductionTest, AlternatingValues) {
  const int size = 10000;
  std::vector<float> input(size);
  for (int i = 0; i < size; ++i) {
    input[i] = (i % 2 == 0) ? 1.0f : -1.0f;
  }
  
  float cpu_result = cpuReduction(input);
  float gpu_result = simple_reduction(input.data(), size);
  
  EXPECT_NEAR(cpu_result, gpu_result, 1e-3f);
  EXPECT_FLOAT_EQ(0.0f, cpu_result); // Should sum to 0
}

// Test Thrust reduction
TEST_F(ReductionTest, ThrustReduction) {
  const int size = 10000;
  std::vector<float> input;
  generateRandomArray(input, size);
  
  float cpu_result = cpuReduction(input);
  float thrust_result = thrust_reduction(input.data(), size);
  
  EXPECT_NEAR(cpu_result, thrust_result, 1e-3f)
    << "Thrust reduction mismatch";
}

// ------------------------------ Coarsened Reduction Tests ------------------------------
class CoarsenedReductionTest : public ::testing::Test {
protected:
  void SetUp() override {
    rng.seed(2026);
    dist = std::uniform_real_distribution<float>(-10.0f, 10.0f);
  }

  void generateRandomArray(std::vector<float> &arr, int size) {
    arr.resize(size);
    for (int i = 0; i < size; ++i) arr[i] = dist(rng);
  }

  float cpuReduction(const std::vector<float> &arr) {
    float sum = 0.0f;
    for (float val : arr) sum += val;
    return sum;
  }

  std::mt19937 rng;
  std::uniform_real_distribution<float> dist;
};

// Test coarsened reduction with small array (auto coarsening factor)
TEST_F(CoarsenedReductionTest, SmallArray) {
  const int size = 1000;
  std::vector<float> input;
  generateRandomArray(input, size);
  
  float cpu_result = cpuReduction(input);
  float gpu_result = coarsened_reduction(input.data(), size);
  
  EXPECT_NEAR(cpu_result, gpu_result, 1e-2f) 
    << "Coarsened reduction mismatch for small array";
}

// Test coarsened reduction with medium array
TEST_F(CoarsenedReductionTest, MediumArray) {
  const int size = 8192;
  std::vector<float> input;
  generateRandomArray(input, size);
  
  float cpu_result = cpuReduction(input);
  float gpu_result = coarsened_reduction(input.data(), size);
  
  EXPECT_NEAR(cpu_result, gpu_result, 1e-2f) 
    << "Coarsened reduction mismatch for medium array";
}

// Test coarsened reduction with larger array
TEST_F(CoarsenedReductionTest, LargerArray) {
  const int size = 65536;
  std::vector<float> input;
  generateRandomArray(input, size);
  
  float cpu_result = cpuReduction(input);
  float gpu_result = coarsened_reduction(input.data(), size);
  
  EXPECT_NEAR(cpu_result, gpu_result, 1e-2f) 
    << "Coarsened reduction mismatch for larger array";
}

// Test large array with coarsening
TEST_F(CoarsenedReductionTest, LargeArrayCoarsened) {
  const int size = 1 << 20; // 1M elements
  std::vector<float> input;
  generateRandomArray(input, size);
  
  float cpu_result = cpuReduction(input);
  float gpu_result = coarsened_reduction(input.data(), size);
  
  float relative_error = std::abs(cpu_result - gpu_result) / std::abs(cpu_result);
  EXPECT_LT(relative_error, 1e-3f) 
    << "Coarsened reduction mismatch for large array (CPU: " 
    << cpu_result << ", GPU: " << gpu_result << ")";
}

// Test very large array (should use high coarsening factor automatically)
TEST_F(CoarsenedReductionTest, VeryLargeArray) {
  const int size = 1 << 24; // 16M elements
  std::vector<float> input;
  generateRandomArray(input, size);
  
  float cpu_result = cpuReduction(input);
  float gpu_result = coarsened_reduction(input.data(), size);
  
  float relative_error = std::abs(cpu_result - gpu_result) / std::abs(cpu_result);
  EXPECT_LT(relative_error, 2e-3f)  // Relaxed tolerance for very large arrays
    << "Coarsened reduction mismatch for very large array";
}

// Test non-power-of-2 size with coarsening
TEST_F(CoarsenedReductionTest, NonPowerOfTwoCoarsened) {
  const int size = 12345;
  std::vector<float> input;
  generateRandomArray(input, size);
  
  float cpu_result = cpuReduction(input);
  float gpu_result = coarsened_reduction(input.data(), size);
  
  EXPECT_NEAR(cpu_result, gpu_result, 1e-2f) 
    << "Coarsened reduction mismatch for non-power-of-2 size";
}

// Compare simple vs coarsened reduction
TEST_F(CoarsenedReductionTest, CompareSimpleVsCoarsened) {
  const int size = 10000;
  std::vector<float> input;
  generateRandomArray(input, size);
  
  float simple_result = simple_reduction(input.data(), size);
  float coarsened_result = coarsened_reduction(input.data(), size);
  
  EXPECT_NEAR(simple_result, coarsened_result, 1e-2f) 
    << "Simple and coarsened reduction should produce similar results";
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}

// (Removed templated convolution wrapper test; direct variant tests above already ensure correctness.)
