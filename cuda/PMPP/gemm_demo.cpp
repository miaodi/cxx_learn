#include "gemm_comparison.h"
#include <iostream>

int main() {
  std::cout << "=== GEMM Implementation Comparison Demo ===\n\n";

  // Demo 1: Quick CPU vs GPU comparison
  std::cout << "Demo 1: Quick CPU vs GPU Comparison (256x256x256)\n";
  gemm_comparison::GEMMComparison::quickCPUvsGPUComparison<float>(256, 256,
                                                                  256);

  // Demo 2: Custom comparison with different matrix sizes
  std::cout << "\nDemo 2: Custom Comparison (1024x1024x1024)\n";

  const int M = 1024, N = 1024, K = 1024;
  std::vector<float> A(M * K), B(K * N);

  // Initialize with known pattern for verification
  for (int i = 0; i < M * K; ++i)
    A[i] = (i % 10) * 0.1f;
  for (int i = 0; i < K * N; ++i)
    B[i] = (i % 7) * 0.2f;

  std::vector<gemm_comparison::GEMMComparison::Implementation> implementations =
      {gemm_comparison::GEMMComparison::Implementation::CPU_NAIVE,
       gemm_comparison::GEMMComparison::Implementation::CPU_TILED_8,
       gemm_comparison::GEMMComparison::Implementation::CPU_TILED_16,
       gemm_comparison::GEMMComparison::Implementation::CPU_ULMBLAS,
       gemm_comparison::GEMMComparison::Implementation::GPU_CUDA};

  gemm_comparison::GEMMComparison::compareImplementations(implementations, A, B,
                                                          M, N, K, 5);

  // Demo 3: Rectangular matrices
  std::cout << "\nDemo 3: Rectangular Matrix Comparison (1024x64x128)\n";
  const int M3 = 1024, N3 = 64, K3 = 128;
  std::vector<float> A3(M3 * K3), B3(K3 * N3);

  std::random_device rd;
  std::mt19937 gen(42); // Fixed seed for reproducibility
  std::uniform_real_distribution<float> dist(-2.0f, 2.0f);

  for (auto &val : A3)
    val = dist(gen);
  for (auto &val : B3)
    val = dist(gen);

  std::vector<gemm_comparison::GEMMComparison::Implementation> rect_impls = {
      gemm_comparison::GEMMComparison::Implementation::CPU_NAIVE,
      gemm_comparison::GEMMComparison::Implementation::CPU_TILED_8,
      gemm_comparison::GEMMComparison::Implementation::CPU_TILED_16,
      gemm_comparison::GEMMComparison::Implementation::CPU_ULMBLAS,
      gemm_comparison::GEMMComparison::Implementation::GPU_CUDA};

  gemm_comparison::GEMMComparison::compareImplementations(rect_impls, A3, B3,
                                                          M3, N3, K3, 100);

  // Demo 4: Individual function usage example
  std::cout << "\nDemo 4: Individual Function Usage Example\n";
  const int M4 = 32, N4 = 32, K4 = 32;
  std::vector<float> A4(M4 * K4, 1.0f), B4(K4 * N4, 2.0f), C4;

  double gpu_time = gemm_comparison::GEMMComparison::runGEMM(
      gemm_comparison::GEMMComparison::Implementation::GPU_CUDA, A4, B4, C4, M4,
      N4, K4);

  std::cout << "GPU GEMM execution time: " << gpu_time << " ms\n";
  std::cout << "Expected result C[0,0]: " << 2.0f * K4 << ", Actual: " << C4[0]
            << "\n";
  std::cout << "Result verification: "
            << (std::abs(C4[0] - 2.0f * K4) < 1e-5f ? "PASS" : "FAIL") << "\n";

  std::cout << "\n=== Demo Complete ===\n";
  return 0;
}