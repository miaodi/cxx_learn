#include <cutlass/cutlass.h>
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/layout/matrix.h>

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr int kM = 128;
constexpr int kN = 96;
constexpr int kK = 64;
constexpr float kAlpha = 1.0f;
constexpr float kBeta = 0.0f;

// Keep the example small and direct: fail fast with the CUDA/CUTLASS call that
// failed instead of threading error codes through the teaching code.
void check_cuda(cudaError_t status, const char *what) {
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string(what) + ": " +
                             cudaGetErrorString(status));
  }
}

void check_cutlass(cutlass::Status status, const char *what) {
  if (status != cutlass::Status::kSuccess) {
    throw std::runtime_error(std::string(what) + ": CUTLASS call failed");
  }
}

std::vector<float> random_values(int count) {
  std::vector<float> values(count);
  std::mt19937 rng(2026);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  for (float &value : values) {
    value = dist(rng);
  }
  return values;
}

std::vector<float> reference_sgemm(const std::vector<float> &A,
                                   const std::vector<float> &B) {
  std::vector<float> C(kM * kN, 0.0f);
  for (int row = 0; row < kM; ++row) {
    for (int col = 0; col < kN; ++col) {
      float sum = 0.0f;
      for (int kk = 0; kk < kK; ++kk) {
        sum += A[row * kK + kk] * B[kk * kN + col];
      }
      C[row * kN + col] = kAlpha * sum + kBeta * C[row * kN + col];
    }
  }
  return C;
}

float max_abs_error(const std::vector<float> &expected,
                    const std::vector<float> &actual) {
  float error = 0.0f;
  for (std::size_t i = 0; i < expected.size(); ++i) {
    error = std::max(error, std::abs(expected[i] - actual[i]));
  }
  return error;
}

} // namespace

// This executable is a minimal CUTLASS SGEMM driver.
//
// It computes:
//
//   D = alpha * A * B + beta * C
//
// with row-major matrices:
//
//   A: M x K
//   B: K x N
//   C: M x N
//   D: M x N
//
// In other words, A * B has the same shape as C and D. The result is written
// to D so that the input C matrix can be treated as the beta-scaled source term.
int main() {
  try {
    // CUTLASS device GEMM expects device pointers. Host vectors are only used
    // here to initialize inputs and verify the result after the kernel runs.
    const std::vector<float> A = random_values(kM * kK);
    const std::vector<float> B = random_values(kK * kN);
    const std::vector<float> C(kM * kN, 0.0f);
    std::vector<float> D(kM * kN, 0.0f);

    float *d_A = nullptr;
    float *d_B = nullptr;
    float *d_C = nullptr;
    float *d_D = nullptr;

    check_cuda(cudaMalloc(&d_A, A.size() * sizeof(float)), "cudaMalloc A");
    check_cuda(cudaMalloc(&d_B, B.size() * sizeof(float)), "cudaMalloc B");
    check_cuda(cudaMalloc(&d_C, C.size() * sizeof(float)), "cudaMalloc C");
    check_cuda(cudaMalloc(&d_D, D.size() * sizeof(float)), "cudaMalloc D");

    check_cuda(cudaMemcpy(d_A, A.data(), A.size() * sizeof(float),
                          cudaMemcpyHostToDevice),
               "copy A");
    check_cuda(cudaMemcpy(d_B, B.data(), B.size() * sizeof(float),
                          cudaMemcpyHostToDevice),
               "copy B");
    check_cuda(cudaMemcpy(d_C, C.data(), C.size() * sizeof(float),
                          cudaMemcpyHostToDevice),
               "copy C");

    // CUTLASS supports several layouts. The learning kernels in this directory
    // use row-major indexing, so this example uses row-major A, B, C, and D.
    using RowMajor = cutlass::layout::RowMajor;

    // This type selects a device-wide GEMM kernel:
    //
    //   element A, layout A,
    //   element B, layout B,
    //   element C/D, layout C/D,
    //   accumulator element.
    //
    // This simple alias lets CUTLASS choose a default kernel for the target
    // architecture. Later examples can replace it with an explicit tiled kernel
    // configuration to study threadblock, warp, and instruction shapes.
    using CutlassSgemm =
        cutlass::gemm::device::Gemm<float, RowMajor, float, RowMajor, float,
                                    RowMajor, float>;

    CutlassSgemm gemm;

    // Arguments describe one GEMM problem instance. The first tuple is the
    // logical problem size: M rows of D, N columns of D, and reduction length K.
    //
    // Each matrix operand is passed as {pointer, leading_dimension}. For
    // row-major contiguous matrices, leading dimensions are:
    //
    //   A: K, because A is M x K
    //   B: N, because B is K x N
    //   C: N, because C is M x N
    //   D: N, because D is M x N
    //
    // The last tuple is the linear-combination epilogue:
    //
    //   D = alpha * accumulator + beta * C
    CutlassSgemm::Arguments args(
        {kM, kN, kK},
        {d_A, kK},
        {d_B, kN},
        {d_C, kN},
        {d_D, kN},
        {kAlpha, kBeta});

    // can_implement checks static and runtime constraints before launching:
    // layout support, alignment requirements, valid dimensions, and whether the
    // selected kernel can handle this problem on the current device.
    check_cutlass(gemm.can_implement(args), "can_implement");

    // operator() launches the CUTLASS kernel. The call is asynchronous, like a
    // normal CUDA kernel launch, so synchronize before reading D on the host.
    check_cutlass(gemm(args), "gemm");
    check_cuda(cudaDeviceSynchronize(), "synchronize");

    check_cuda(cudaMemcpy(D.data(), d_D, D.size() * sizeof(float),
                          cudaMemcpyDeviceToHost),
               "copy D");

    const std::vector<float> expected = reference_sgemm(A, B);
    const float error = max_abs_error(expected, D);

    check_cuda(cudaFree(d_A), "free A");
    check_cuda(cudaFree(d_B), "free B");
    check_cuda(cudaFree(d_C), "free C");
    check_cuda(cudaFree(d_D), "free D");

    std::cout << "CUTLASS row-major SGEMM example\n";
    std::cout << "M=" << kM << " N=" << kN << " K=" << kK << "\n";
    std::cout << "max_abs_error=" << error << "\n";
    return error < 1e-3f ? 0 : 1;
  } catch (const std::exception &error) {
    std::cerr << error.what() << "\n";
    return 1;
  }
}
