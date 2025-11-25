#include <iostream>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/transform.h>
#include <thrust/reduce.h>
#include <thrust/functional.h>
#include <thrust/sort.h>

// Example: Simple vector addition using Thrust
void example_vector_add() {
    std::cout << "=== Vector Addition ===" << std::endl;
    
    // Create host vectors
    thrust::host_vector<float> h_a(1000);
    thrust::host_vector<float> h_b(1000);
    
    // Initialize with values
    for (int i = 0; i < 1000; ++i) {
        h_a[i] = i;
        h_b[i] = i * 2.0f;
    }
    
    // Transfer to device
    thrust::device_vector<float> d_a = h_a;
    thrust::device_vector<float> d_b = h_b;
    thrust::device_vector<float> d_c(1000);
    
    // Perform addition using transform
    thrust::transform(d_a.begin(), d_a.end(), d_b.begin(), d_c.begin(),
                     thrust::plus<float>());
    
    // Copy result back to host
    thrust::host_vector<float> h_c = d_c;
    
    std::cout << "First 5 results: ";
    for (int i = 0; i < 5; ++i) {
        std::cout << h_c[i] << " ";
    }
    std::cout << std::endl;
}

// Example: Reduction
void example_reduction() {
    std::cout << "\n=== Reduction (Sum) ===" << std::endl;
    
    thrust::device_vector<int> d_vec(1000);
    thrust::fill(d_vec.begin(), d_vec.end(), 1);
    
    int sum = thrust::reduce(d_vec.begin(), d_vec.end(), 0, thrust::plus<int>());
    
    std::cout << "Sum of 1000 ones: " << sum << std::endl;
}

// Example: Sorting
void example_sorting() {
    std::cout << "\n=== Sorting ===" << std::endl;
    
    thrust::host_vector<int> h_vec(10);
    h_vec[0] = 5; h_vec[1] = 2; h_vec[2] = 8;
    h_vec[3] = 1; h_vec[4] = 9; h_vec[5] = 3;
    h_vec[6] = 7; h_vec[7] = 6; h_vec[8] = 4; h_vec[9] = 0;
    
    std::cout << "Before sort: ";
    for (int i = 0; i < 10; ++i) {
        std::cout << h_vec[i] << " ";
    }
    std::cout << std::endl;
    
    thrust::device_vector<int> d_vec = h_vec;
    thrust::sort(d_vec.begin(), d_vec.end());
    
    h_vec = d_vec;
    std::cout << "After sort:  ";
    for (int i = 0; i < 10; ++i) {
        std::cout << h_vec[i] << " ";
    }
    std::cout << std::endl;
}

int main() {
    std::cout << "Thrust Practice Examples" << std::endl;
    std::cout << "========================" << std::endl;
    
    example_vector_add();
    example_reduction();
    example_sorting();
    
    std::cout << "\nAll examples completed successfully!" << std::endl;
    return 0;
}
