#include "scan.h"

#include <cuda_runtime.h>

namespace pmpp::scan {
namespace {

cudaError_t validate_scan_arguments(const float *d_input, const float *d_output,
                                    int num_items) {
  if (num_items < 0) {
    return cudaErrorInvalidValue;
  }
  if (num_items == 0) {
    return cudaSuccess;
  }
  if (d_input == nullptr || d_output == nullptr) {
    return cudaErrorInvalidValue;
  }
  return cudaSuccess;
}

cudaError_t placeholder_inclusive_scan(const float *d_input, float *d_output,
                                       int num_items, cudaStream_t stream) {
  (void)stream;

  const cudaError_t status =
      validate_scan_arguments(d_input, d_output, num_items);
  if (status != cudaSuccess || num_items == 0) {
    return status;
  }

  return cudaErrorNotSupported;
}

__global__ void brent_kung_inclusive_scan_kernel(const float *input,
                                                 float *output,
                                                 int num_items) {
  __shared__ float tile[kBrentKungItemsPerBlock];

  const int tid = threadIdx.x;
  const int first = tid;
  const int second = tid + blockDim.x;

  tile[first] = (first < num_items) ? input[first] : 0.0f;
  tile[second] = (second < num_items) ? input[second] : 0.0f;
  __syncthreads();

  for (int stride = 1; stride <= blockDim.x; stride *= 2) {
    const int index = (tid + 1) * 2 * stride - 1;
    if (index < kBrentKungItemsPerBlock) {
      tile[index] += tile[index - stride];
    }
    __syncthreads();
  }

  for (int stride = kBrentKungItemsPerBlock / 4; stride > 0; stride /= 2) {
    const int index = (tid + 1) * 2 * stride - 1;
    if (index + stride < kBrentKungItemsPerBlock) {
      tile[index + stride] += tile[index];
    }
    __syncthreads();
  }

  if (first < num_items) {
    output[first] = tile[first];
  }
  if (second < num_items) {
    output[second] = tile[second];
  }
}

} // namespace

cudaError_t kogge_stone_inclusive_scan(const float *d_input, float *d_output,
                                       int num_items, cudaStream_t stream) {
  return placeholder_inclusive_scan(d_input, d_output, num_items, stream);
}

cudaError_t brent_kung_inclusive_scan(const float *d_input, float *d_output,
                                      int num_items, cudaStream_t stream) {
  const cudaError_t status =
      validate_scan_arguments(d_input, d_output, num_items);
  if (status != cudaSuccess || num_items == 0) {
    return status;
  }
  if (num_items > kBrentKungItemsPerBlock) {
    return cudaErrorNotSupported;
  }

  brent_kung_inclusive_scan_kernel<<<1, kBrentKungBlockSize, 0, stream>>>(
      d_input, d_output, num_items);
  return cudaGetLastError();
}

cudaError_t coarsened_brent_kung_inclusive_scan(
    const float *d_input, float *d_output, int num_items, cudaStream_t stream) {
  return placeholder_inclusive_scan(d_input, d_output, num_items, stream);
}

cudaError_t hierarchical_inclusive_scan(const float *d_input, float *d_output,
                                        int num_items, cudaStream_t stream) {
  return placeholder_inclusive_scan(d_input, d_output, num_items, stream);
}

cudaError_t cub_inclusive_scan_reference(const float *d_input, float *d_output,
                                         int num_items, cudaStream_t stream) {
  return placeholder_inclusive_scan(d_input, d_output, num_items, stream);
}

} // namespace pmpp::scan
