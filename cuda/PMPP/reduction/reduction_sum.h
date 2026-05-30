#pragma once

#include <cstddef>

namespace pmpp::reduction {

struct DeviceReductionBuffers {
  float *scratch = nullptr;
  float *partials_a = nullptr;
  float *partials_b = nullptr;
  std::size_t scratch_capacity = 0;
  std::size_t partial_capacity = 0;
};

void allocate_device_reduction_buffers(DeviceReductionBuffers &buffers,
                                       std::size_t size);
void free_device_reduction_buffers(DeviceReductionBuffers &buffers);

float simple_stride_sum(const float *input, std::size_t size);
float sequential_addressing_sum(const float *input, std::size_t size);
float coarsened_shared_memory_sum(const float *input, std::size_t size);
float coarsened_shared_memory_sum_optimized(const float *input,
                                            std::size_t size);
float thrust_reference_sum(const float *input, std::size_t size);

float simple_stride_sum_device(const float *device_input, std::size_t size,
                               DeviceReductionBuffers &buffers);
float sequential_addressing_sum_device(const float *device_input,
                                       std::size_t size,
                                       DeviceReductionBuffers &buffers);
float coarsened_shared_memory_sum_device(const float *device_input,
                                          std::size_t size,
                                          DeviceReductionBuffers &buffers);
float coarsened_shared_memory_sum_optimized_device(
    const float *device_input, std::size_t size,
    DeviceReductionBuffers &buffers);
float thrust_reference_sum_device(const float *device_input, std::size_t size);

} // namespace pmpp::reduction
