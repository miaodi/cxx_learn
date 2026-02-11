#include "radix_sort.h"

#include <algorithm>

namespace radix {

namespace {

// Get the digit at position 'digit' for radix 'base'
uint32_t get_digit(uint32_t value, uint32_t digit, uint32_t base) {
    return (value / base) % 10;
}

// Counting sort by a specific digit (used as subroutine in radix sort)
void counting_sort_by_digit(std::vector<uint32_t>& arr, uint32_t digit, uint32_t base) {
    const size_t n = arr.size();
    std::vector<uint32_t> output(n);
    std::vector<uint32_t> count(10, 0);
    
    // Count occurrences of each digit
    for (size_t i = 0; i < n; ++i) {
        uint32_t d = get_digit(arr[i], digit, base);
        count[d]++;
    }
    
    // Compute cumulative count (positions)
    for (size_t i = 1; i < 10; ++i) {
        count[i] += count[i - 1];
    }
    
    // Build output array by placing elements in sorted order
    // Process from right to left to maintain stability
    for (size_t i = n; i > 0; --i) {
        uint32_t d = get_digit(arr[i - 1], digit, base);
        output[count[d] - 1] = arr[i - 1];
        count[d]--;
    }
    
    // Copy output back to original array
    arr = std::move(output);
}

} // anonymous namespace

// Radix sort for unsigned 32-bit integers
// Sorts in ascending order using LSD (Least Significant Digit) approach
void sort(std::vector<uint32_t>& arr) {
    if (arr.empty()) return;
    
    // Find maximum number to determine number of digits
    uint32_t max_val = *std::max_element(arr.begin(), arr.end());
    
    // Process each digit from least significant to most significant
    uint32_t base = 1;
    while (max_val / base > 0) {
        counting_sort_by_digit(arr, 0, base);
        base *= 10;
    }
}

// In-place radix sort for signed integers
// Handles negative numbers by treating sign bit separately
void sort_signed(std::vector<int32_t>& arr) {
    if (arr.empty()) return;
    
    // Separate negative and positive numbers
    std::vector<int32_t> negatives;
    std::vector<uint32_t> positives;
    
    for (int32_t val : arr) {
        if (val < 0) {
            negatives.push_back(val);
        } else {
            positives.push_back(static_cast<uint32_t>(val));
        }
    }
    
    // Sort negatives (convert to unsigned, sort descending, convert back)
    std::vector<uint32_t> neg_unsigned;
    for (int32_t val : negatives) {
        neg_unsigned.push_back(static_cast<uint32_t>(-val));
    }
    sort(neg_unsigned);
    
    // Rebuild array: negatives (reversed) + positives
    arr.clear();
    for (auto it = neg_unsigned.rbegin(); it != neg_unsigned.rend(); ++it) {
        arr.push_back(-static_cast<int32_t>(*it));
    }
    
    sort(positives);
    for (uint32_t val : positives) {
        arr.push_back(static_cast<int32_t>(val));
    }
}

} // namespace radix
