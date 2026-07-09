#pragma once

#include <cuda_runtime_api.h>

namespace pmpp::scan {

inline constexpr int kBrentKungBlockSize = 256;
inline constexpr int kBrentKungItemsPerBlock = 2 * kBrentKungBlockSize;

// Inclusive scan over device-resident float arrays.
// The input and output buffers may alias once implementations support it.
cudaError_t kogge_stone_inclusive_scan(const float *d_input, float *d_output,
                                       int num_items,
                                       cudaStream_t stream = nullptr);

cudaError_t brent_kung_inclusive_scan(const float *d_input, float *d_output,
                                      int num_items,
                                      cudaStream_t stream = nullptr);

cudaError_t coarsened_brent_kung_inclusive_scan(
    const float *d_input, float *d_output, int num_items,
    cudaStream_t stream = nullptr);

cudaError_t hierarchical_inclusive_scan(const float *d_input, float *d_output,
                                        int num_items,
                                        cudaStream_t stream = nullptr);

cudaError_t cub_inclusive_scan_reference(const float *d_input, float *d_output,
                                         int num_items,
                                         cudaStream_t stream = nullptr);

} // namespace pmpp::scan
