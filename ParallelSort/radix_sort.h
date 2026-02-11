#pragma once

#include <cstdint>
#include <vector>

namespace radix {

// Radix sort for unsigned 32-bit integers
// Sorts in ascending order using LSD (Least Significant Digit) approach
void sort(std::vector<uint32_t>& arr);

// Radix sort with range-based interface
template<typename RandomIt>
void sort(RandomIt first, RandomIt last) {
    if (first == last) return;
    
    std::vector<uint32_t> temp(first, last);
    sort(temp);
    std::copy(temp.begin(), temp.end(), first);
}

// In-place radix sort for signed integers
// Handles negative numbers by treating sign bit separately
void sort_signed(std::vector<int32_t>& arr);

} // namespace radix
