#include "vector_add.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <gtest/gtest.h>
#include <iomanip>
#include <random>
#include <vector>
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
