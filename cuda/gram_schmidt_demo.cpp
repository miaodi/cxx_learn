#include "gram_schmidt.h"
#include <iostream>
#include <iomanip>
#include <vector>

template<typename T>
void runDemo(const std::string& type_name) {
    std::cout << "=== Templated Gram-Schmidt Demo (" << type_name << ") ===\n\n";
    
    try {
        // Demo 1: Small matrix test with detailed output
        std::cout << "Demo 1: Small Matrix (6x4) - " << type_name << "\n";
        const int m1 = 6, n1 = 4;
        
        std::vector<T> input_matrix1(m1 * n1);
        gram_schmidt::generateRandomMatrix(input_matrix1.data(), m1, n1, 42);
        
        std::cout << "Original matrix (column-major):\n";
        for (int i = 0; i < m1; ++i) {
            for (int j = 0; j < n1; ++j) {
                std::cout << std::fixed << std::setprecision(3)
                          << input_matrix1[j * m1 + i] << "\t";
            }
            std::cout << "\n";
        }
        std::cout << "\n";
        
        std::vector<gram_schmidt::MemoryScheme> schemes = {
            gram_schmidt::MemoryScheme::DEVICE_MEMORY,
            gram_schmidt::MemoryScheme::HOST_MEMORY,
            gram_schmidt::MemoryScheme::UNIFIED_MEMORY
        };
        
        std::vector<std::string> scheme_names = {
            "Device Memory",
            "Host Memory", 
            "Unified Memory"
        };
        
        for (size_t s = 0; s < schemes.size(); ++s) {
            std::cout << "Testing " << scheme_names[s] << ":\n";
            
            gram_schmidt::GramSchmidtOrthogonalizer<T> orthogonalizer(m1, n1, schemes[s]);
            T* result_matrix;
            double elapsed = orthogonalizer.orthogonalize(input_matrix1.data(), &result_matrix);
            
            std::vector<T> host_result(m1 * n1);
            if (schemes[s] == gram_schmidt::MemoryScheme::DEVICE_MEMORY) {
                orthogonalizer.copyResultToHost(host_result.data());
            } else {
                std::memcpy(host_result.data(), result_matrix, m1 * n1 * sizeof(T));
            }
            
            bool is_orthogonal = gram_schmidt::verifyOrthogonality(host_result.data(), m1, n1, static_cast<T>(1e-4));
            
            std::cout << "Execution time: " << elapsed << " ms\n";
            std::cout << "Orthogonality check: " << (is_orthogonal ? "PASS" : "FAIL") << "\n";
            std::cout << "Orthogonalized matrix:\n";
            for (int i = 0; i < m1; ++i) {
                for (int j = 0; j < n1; ++j) {
                    std::cout << std::fixed << std::setprecision(3)
                              << host_result[j * m1 + i] << "\t";
                }
                std::cout << "\n";
            }
            std::cout << "\n";
        }
        
        // Demo 2: Performance comparison
        std::cout << "Demo 2: Performance Benchmarks - " << type_name << "\n\n";
        
        std::cout << "Small matrices:\n";
        gram_schmidt::benchmarkGramSchmidt<T>(64, 32, 3);
        
        std::cout << "Medium matrices:\n";
        gram_schmidt::benchmarkGramSchmidt<T>(256, 128, 3);
        
        std::cout << "Large matrices:\n";
        gram_schmidt::benchmarkGramSchmidt<T>(1024, 256, 3);
        
        // Demo 3: Square matrix test  
        std::cout << "Demo 3: Square Matrix (128x128) - " << type_name << "\n";
        const int m3 = 128, n3 = 128;
        std::vector<T> input_matrix3(m3 * n3);
        gram_schmidt::generateRandomMatrix(input_matrix3.data(), m3, n3, 54321);
        
        for (size_t s = 0; s < schemes.size(); ++s) {
            gram_schmidt::GramSchmidtOrthogonalizer<T> orthogonalizer(m3, n3, schemes[s]);
            T* result_matrix;
            double elapsed = orthogonalizer.orthogonalize(input_matrix3.data(), &result_matrix);
            
            std::vector<T> host_result(m3 * n3);
            if (schemes[s] == gram_schmidt::MemoryScheme::DEVICE_MEMORY) {
                orthogonalizer.copyResultToHost(host_result.data());
            } else {
                std::memcpy(host_result.data(), result_matrix, m3 * n3 * sizeof(T));
            }
            
            bool is_orthogonal = gram_schmidt::verifyOrthogonality(host_result.data(), m3, n3, static_cast<T>(1e-3));
            
            std::cout << scheme_names[s] << ": " << elapsed << " ms, Orthogonal: " 
                      << (is_orthogonal ? "YES" : "NO") << "\n";
        }
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
    }
    
    std::cout << "\n=== Demo Complete (" << type_name << ") ===\n\n";
}

int main() {
    // Run demo for both float and double
    runDemo<float>("Float");
    runDemo<double>("Double");
    
    return 0;
}