#pragma once

#include <cstdint>
#include <cstddef>

namespace pmpp {
namespace radix_sort {

/**
 * @brief CUDA implementation of radix sort for uint32_t arrays
 * 
 * @tparam RADIX_BITS Number of bits to process per iteration (must divide 32 evenly)
 * @param d_input Input array on device
 * @param d_output Output array on device (sorted)
 * @param n Number of elements
 */
template<int RADIX_BITS = 4>
void radix_sort(const uint32_t* d_input, uint32_t* d_output, size_t n);

/**
 * @brief In-place CUDA radix sort for uint32_t arrays
 * 
 * @tparam RADIX_BITS Number of bits to process per iteration (must divide 32 evenly)
 * @param d_data Array on device to be sorted in-place
 * @param n Number of elements
 */
template<int RADIX_BITS = 4>
void radix_sort_inplace(uint32_t* d_data, size_t n);

/**
 * @brief Host wrapper that handles memory allocation and transfer
 * 
 * @tparam RADIX_BITS Number of bits to process per iteration (must divide 32 evenly)
 * @param h_input Host input array
 * @param h_output Host output array (sorted)
 * @param n Number of elements
 */
template<int RADIX_BITS = 4>
void radix_sort_host(const uint32_t* h_input, uint32_t* h_output, size_t n);

} // namespace radix_sort
} // namespace pmpp
