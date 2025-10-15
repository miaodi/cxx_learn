#pragma once

#include <vector>
#include <chrono>
#include <iostream>
#include <iomanip>
#include <random>

// GPU implementation
#include "gemm.h"

// CPU implementations from GEMM folder
#include "../../GEMM/gemm.hpp"
#include "../../GEMM/ulmBLASgemm.hpp"
#include "../../GEMM/dgemm_nn.h"

namespace gemm_comparison {

/**
 * @brief Comprehensive GEMM comparison utility
 * 
 * This class provides easy-to-use functions for comparing different GEMM implementations
 * across CPU (naive, optimized, ulmBLAS) and GPU versions.
 */
class GEMMComparison {
public:
    enum class Implementation {
        CPU_NAIVE,           // Simple triple-loop implementation
        CPU_TILED_8,         // 8x8 tiled CPU implementation
        CPU_TILED_16,        // 16x16 tiled CPU implementation
        CPU_ULMBLAS,         // ulmBLAS optimized implementation
        CPU_DGEMM_NN,        // dgemm_nn implementation (double precision)
        GPU_CUDA             // CUDA GPU implementation
    };

private:
    // Timing utilities
    using TimePoint = std::chrono::high_resolution_clock::time_point;
    
    static TimePoint now() {
        return std::chrono::high_resolution_clock::now();
    }
    
    static double duration_ms(TimePoint start, TimePoint end) {
        return std::chrono::duration<double, std::milli>(end - start).count();
    }

public:
    /**
     * @brief Run GEMM with specified implementation
     * @param impl Implementation to use
     * @param A Input matrix A (M x K)
     * @param B Input matrix B (K x N) 
     * @param C Output matrix C (M x N)
     * @param M Number of rows in A and C
     * @param N Number of columns in B and C
     * @param K Number of columns in A and rows in B
     * @return Execution time in milliseconds
     */
    template<typename T>
    static double runGEMM(Implementation impl, 
                         const std::vector<T>& A, 
                         const std::vector<T>& B,
                         std::vector<T>& C, 
                         int M, int N, int K) {
        C.resize(M * N);
        std::fill(C.begin(), C.end(), T(0));
        
        auto start = now();
        
        switch (impl) {
            case Implementation::CPU_NAIVE:
                gemm::MatMatMul<T>(A.data(), B.data(), C.data(), M, N, K);
                break;
                
            case Implementation::CPU_TILED_8:
                gemm::TiledMatMatMulInternalTiledPadded<T, 8>(A.data(), B.data(), C.data(), M, N, K);
                break;
                
            case Implementation::CPU_TILED_16:
                gemm::TiledMatMatMulInternalTiledPadded<T, 16>(A.data(), B.data(), C.data(), M, N, K);
                break;
                
            case Implementation::CPU_ULMBLAS: {
                gemm::gemm_pure_c<T> gemm_impl;
                gemm_impl(A.data(), B.data(), C.data(), M, N, K);
                break;
            }
            
            case Implementation::CPU_DGEMM_NN:
                if constexpr (std::is_same_v<T, double>) {
                    gemm::gemm_nn<double>(M, N, K, 1.0, A.data(), K, B.data(), N, 0.0, C.data(), N);
                } else {
                    throw std::runtime_error("CPU_DGEMM_NN only supports double precision");
                }
                break;
                
            case Implementation::GPU_CUDA:
                if constexpr (std::is_same_v<T, float>) {
                    gpu_gemm(A.data(), B.data(), C.data(), M, N, K);
                } else {
                    throw std::runtime_error("GPU_CUDA currently only supports float precision");
                }
                break;
        }
        
        auto end = now();
        return duration_ms(start, end);
    }

    /**
     * @brief Compare multiple GEMM implementations
     * @param implementations List of implementations to compare
     * @param A Input matrix A 
     * @param B Input matrix B
     * @param M, N, K Matrix dimensions
     * @param num_runs Number of runs for timing (default: 5)
     */
    template<typename T>
    static void compareImplementations(const std::vector<Implementation>& implementations,
                                     const std::vector<T>& A, 
                                     const std::vector<T>& B,
                                     int M, int N, int K,
                                     int num_runs = 5) {
        std::cout << std::fixed << std::setprecision(2);
        std::cout << "\n=== GEMM Performance Comparison ===\n";
        std::cout << "Matrix dimensions: " << M << " x " << N << " x " << K << "\n";
        std::cout << "Number of runs: " << num_runs << "\n\n";
        
        std::vector<std::vector<T>> results;
        std::vector<double> times;
        std::vector<std::string> names;
        
        for (auto impl : implementations) {
            std::string name = getImplementationName(impl);
            names.push_back(name);
            
            std::vector<T> C;
            double total_time = 0.0;
            
            // Warm up
            runGEMM(impl, A, B, C, M, N, K);
            
            // Timing runs
            for (int run = 0; run < num_runs; ++run) {
                total_time += runGEMM(impl, A, B, C, M, N, K);
            }
            
            double avg_time = total_time / num_runs;
            times.push_back(avg_time);
            results.push_back(C);
            
            // Calculate GFLOPS
            double gflops = (2.0 * M * N * K) / (avg_time * 1e6);
            
            std::cout << std::setw(20) << name << ": " 
                      << std::setw(8) << avg_time << " ms, "
                      << std::setw(6) << gflops << " GFLOPS\n";
        }
        
        // Verify all results are consistent (compare against first implementation)
        if (results.size() > 1) {
            std::cout << "\n=== Correctness Verification ===\n";
            const auto& reference = results[0];
            const T tolerance = std::is_same_v<T, float> ? T(1e-4) : T(1e-12);
            
            for (size_t i = 1; i < results.size(); ++i) {
                bool match = verifyResults(reference, results[i], tolerance);
                std::cout << names[0] << " vs " << names[i] << ": " 
                          << (match ? "PASS" : "FAIL") << "\n";
            }
        }
        
        // Show speedups relative to first implementation
        if (times.size() > 1) {
            std::cout << "\n=== Speedup Analysis ===\n";
            double baseline = times[0];
            for (size_t i = 1; i < times.size(); ++i) {
                double speedup = baseline / times[i];
                std::cout << names[i] << " vs " << names[0] << ": " 
                          << speedup << "x speedup\n";
            }
        }
        std::cout << "\n";
    }

    /**
     * @brief Quick comparison of CPU vs GPU GEMM
     */
    template<typename T = float>
    static void quickCPUvsGPUComparison(int M = 256, int N = 256, int K = 256) {
        static_assert(std::is_same_v<T, float>, "GPU implementation currently only supports float");
        
        // Generate random test matrices
        std::vector<T> A(M * K), B(K * N);
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<T> dist(-1.0, 1.0);
        
        for (auto& val : A) val = dist(gen);
        for (auto& val : B) val = dist(gen);
        
        std::vector<Implementation> impls = {
            Implementation::CPU_NAIVE,
            Implementation::CPU_TILED_16,
            Implementation::CPU_ULMBLAS,
            Implementation::GPU_CUDA
        };
        
        compareImplementations(impls, A, B, M, N, K);
    }

private:
    static std::string getImplementationName(Implementation impl) {
        switch (impl) {
            case Implementation::CPU_NAIVE: return "CPU Naive";
            case Implementation::CPU_TILED_8: return "CPU Tiled 8x8";
            case Implementation::CPU_TILED_16: return "CPU Tiled 16x16";
            case Implementation::CPU_ULMBLAS: return "CPU ulmBLAS";
            case Implementation::CPU_DGEMM_NN: return "CPU dgemm_nn";
            case Implementation::GPU_CUDA: return "GPU CUDA";
        }
        return "Unknown";
    }
    
    template<typename T>
    static bool verifyResults(const std::vector<T>& reference, 
                             const std::vector<T>& test, 
                             T tolerance) {
        if (reference.size() != test.size()) return false;
        
        for (size_t i = 0; i < reference.size(); ++i) {
            if (std::abs(reference[i] - test[i]) > tolerance) {
                return false;
            }
        }
        return true;
    }
};

} // namespace gemm_comparison