#include "radix_sort.h"

#include <algorithm>
#include <iostream>
#include <random>
#include <vector>

// Test helper function
template<typename T>
bool is_sorted(const std::vector<T>& arr) {
    for (size_t i = 1; i < arr.size(); ++i) {
        if (arr[i] < arr[i-1]) return false;
    }
    return true;
}

// Test helper: print vector
template<typename T>
void print_vector(const std::vector<T>& arr, const std::string& name) {
    std::cout << name << ": [";
    for (size_t i = 0; i < arr.size(); ++i) {
        std::cout << arr[i];
        if (i < arr.size() - 1) std::cout << ", ";
    }
    std::cout << "]\n";
}

void test_empty_array() {
    std::cout << "Test: Empty array\n";
    std::vector<uint32_t> arr;
    radix::sort(arr);
    std::cout << "  Result: " << (arr.empty() ? "PASS" : "FAIL") << "\n\n";
}

void test_single_element() {
    std::cout << "Test: Single element\n";
    std::vector<uint32_t> arr = {42};
    radix::sort(arr);
    std::cout << "  Result: " << (arr[0] == 42 ? "PASS" : "FAIL") << "\n\n";
}

void test_already_sorted() {
    std::cout << "Test: Already sorted array\n";
    std::vector<uint32_t> arr = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    radix::sort(arr);
    bool passed = is_sorted(arr);
    print_vector(arr, "  Output");
    std::cout << "  Result: " << (passed ? "PASS" : "FAIL") << "\n\n";
}

void test_reverse_sorted() {
    std::cout << "Test: Reverse sorted array\n";
    std::vector<uint32_t> arr = {10, 9, 8, 7, 6, 5, 4, 3, 2, 1};
    print_vector(arr, "  Input ");
    radix::sort(arr);
    bool passed = is_sorted(arr);
    print_vector(arr, "  Output");
    std::cout << "  Result: " << (passed ? "PASS" : "FAIL") << "\n\n";
}

void test_duplicates() {
    std::cout << "Test: Array with duplicates\n";
    std::vector<uint32_t> arr = {5, 2, 8, 2, 9, 1, 5, 5};
    print_vector(arr, "  Input ");
    radix::sort(arr);
    bool passed = is_sorted(arr);
    print_vector(arr, "  Output");
    std::cout << "  Result: " << (passed ? "PASS" : "FAIL") << "\n\n";
}

void test_all_same() {
    std::cout << "Test: All same elements\n";
    std::vector<uint32_t> arr(100, 42);
    radix::sort(arr);
    bool passed = is_sorted(arr) && std::all_of(arr.begin(), arr.end(), 
                                                  [](uint32_t x) { return x == 42; });
    std::cout << "  Result: " << (passed ? "PASS" : "FAIL") << "\n\n";
}

void test_random_small() {
    std::cout << "Test: Random small array\n";
    std::vector<uint32_t> arr = {170, 45, 75, 90, 802, 24, 2, 66};
    print_vector(arr, "  Input ");
    radix::sort(arr);
    bool passed = is_sorted(arr);
    print_vector(arr, "  Output");
    std::cout << "  Result: " << (passed ? "PASS" : "FAIL") << "\n\n";
}

void test_large_random() {
    std::cout << "Test: Large random array (1000 elements)\n";
    std::vector<uint32_t> arr(1000);
    
    std::random_device rd;
    std::mt19937 gen(42); // Fixed seed for reproducibility
    std::uniform_int_distribution<uint32_t> dis(0, 1000000);
    
    for (auto& val : arr) {
        val = dis(gen);
    }
    
    // Create a copy and sort with std::sort for verification
    std::vector<uint32_t> expected = arr;
    std::sort(expected.begin(), expected.end());
    
    radix::sort(arr);
    
    bool passed = is_sorted(arr) && (arr == expected);
    std::cout << "  Result: " << (passed ? "PASS" : "FAIL") << "\n\n";
}

void test_signed_integers() {
    std::cout << "Test: Signed integers (negative and positive)\n";
    std::vector<int32_t> arr = {-5, 20, -10, 0, 15, -3, 8, -20, 50, -1};
    
    std::cout << "  Input : [";
    for (size_t i = 0; i < arr.size(); ++i) {
        std::cout << arr[i];
        if (i < arr.size() - 1) std::cout << ", ";
    }
    std::cout << "]\n";
    
    radix::sort_signed(arr);
    
    bool passed = is_sorted(arr);
    std::cout << "  Output: [";
    for (size_t i = 0; i < arr.size(); ++i) {
        std::cout << arr[i];
        if (i < arr.size() - 1) std::cout << ", ";
    }
    std::cout << "]\n";
    std::cout << "  Result: " << (passed ? "PASS" : "FAIL") << "\n\n";
}

void test_range_interface() {
    std::cout << "Test: Range-based interface\n";
    std::vector<uint32_t> arr = {64, 25, 12, 22, 11};
    print_vector(arr, "  Input ");
    radix::sort(arr.begin(), arr.end());
    bool passed = is_sorted(arr);
    print_vector(arr, "  Output");
    std::cout << "  Result: " << (passed ? "PASS" : "FAIL") << "\n\n";
}

int main() {
    std::cout << "=== Radix Sort Tests ===\n\n";
    
    test_empty_array();
    test_single_element();
    test_already_sorted();
    test_reverse_sorted();
    test_duplicates();
    test_all_same();
    test_random_small();
    test_large_random();
    test_signed_integers();
    test_range_interface();
    
    std::cout << "=== All tests completed ===\n";
    
    return 0;
}
