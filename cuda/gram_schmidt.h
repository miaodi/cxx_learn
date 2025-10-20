#pragma once

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <vector>
#include <memory>

namespace gram_schmidt {

/**
 * @brief Memory scheme for Gram-Schmidt orthogonalization
 */
enum class MemoryScheme {
    DEVICE_MEMORY,    // Orthogonalized matrix stays on device
    HOST_MEMORY,      // Orthogonalized matrix copied back to host
    UNIFIED_MEMORY    // Using CUDA unified memory
};

/**
 * @brief RAII wrapper for cuBLAS handle
 */
class CublasHandle {
public:
    CublasHandle();
    ~CublasHandle();
    
    // Non-copyable but movable
    CublasHandle(const CublasHandle&) = delete;
    CublasHandle& operator=(const CublasHandle&) = delete;
    CublasHandle(CublasHandle&& other) noexcept;
    CublasHandle& operator=(CublasHandle&& other) noexcept;
    
    cublasHandle_t get() const { return handle_; }
    
private:
    cublasHandle_t handle_;
};

/**
 * @brief Templated Gram-Schmidt Orthogonalization using cuBLAS
 * 
 * This class implements the classical Gram-Schmidt process for orthogonalizing
 * a set of column vectors using cuBLAS for dot products and norms.
 * 
 * @tparam T Floating point type (float or double)
 */
template<typename T>
class GramSchmidtOrthogonalizer {
public:
    /**
     * @brief Constructor
     * @param m Number of rows
     * @param n Number of columns  
     * @param scheme Memory scheme to use
     */
    GramSchmidtOrthogonalizer(int m, int n, MemoryScheme scheme);
    
    /**
     * @brief Destructor - handles cleanup
     */
    ~GramSchmidtOrthogonalizer();
    
    /**
     * @brief Orthogonalize a column-major matrix
     * @param input_matrix Column-major matrix of size m x n (host memory)
     * @param output_matrix Orthogonalized matrix (location depends on memory scheme)
     * @return Execution time in milliseconds
     */
    double orthogonalize(const T* input_matrix, T** output_matrix);
    
    /**
     * @brief Get pointer to result matrix (depends on memory scheme)
     */
    T* getResultMatrix() const { return result_matrix_; }
    
    /**
     * @brief Get memory scheme being used
     */
    MemoryScheme getMemoryScheme() const { return scheme_; }
    
    /**
     * @brief Copy result from device to host (for DEVICE_MEMORY scheme)
     */
    void copyResultToHost(T* host_matrix) const;

private:
    // Disable copy constructor and assignment
    GramSchmidtOrthogonalizer(const GramSchmidtOrthogonalizer&) = delete;
    GramSchmidtOrthogonalizer& operator=(const GramSchmidtOrthogonalizer&) = delete;
    
    // Helper functions
    void allocateMemory();
    void deallocateMemory();
    void copyInputToDevice(const T* input_matrix);
    double performOrthogonalization();
    void handleResult();
    
    // Gram-Schmidt algorithm implementation
    void orthogonalizeColumn(int col_idx);
    T computeDotProduct(int col1, int col2);
    T computeNorm(int col_idx);
    void scaleColumn(int col_idx, T scale_factor);
    void subtractProjection(int target_col, int basis_col, T projection);
    
    // Device-only versions for DEVICE_MEMORY and UNIFIED_MEMORY schemes
    void computeDotProductDevice(int col1, int col2, T* d_result);
    void computeNormDevice(int col_idx, T* d_result);
    void scaleColumnDevice(int col_idx, T* d_scale_factor);
    void subtractProjectionDevice(int target_col, int basis_col, T* d_projection);
    bool checkNormThresholdDevice(T* d_norm, T threshold);
    void reciprocalDevice(T* d_value);  // Compute 1/value on device
    
    // Members
    int m_, n_;                    // Matrix dimensions
    MemoryScheme scheme_;          // Memory scheme
    CublasHandle cublas_handle_;   // cuBLAS handle
    
    // Memory pointers
    T* device_matrix_;             // Device memory for working matrix
    T* result_matrix_;             // Result matrix (location depends on scheme)
    T* host_result_;               // Host result (for HOST_MEMORY scheme)
    
    // Temporary vectors for cuBLAS operations
    T* temp_vector_;               // Temporary vector on device
    
    // Device storage for intermediate scalar values (used for all memory schemes)
    T* d_dot_result_;              // Device storage for dot products
    T* d_norm_result_;             // Device storage for norms
    T* d_scale_factor_;            // Device storage for scale factors
    T* d_projection_;              // Device storage for projection values
    T* d_neg_one_;                 // Device storage for -1 constant
    int* d_threshold_result_;      // Device storage for threshold check results
    
    // Stream for async operations
    cudaStream_t stream_;
};

// Type aliases for convenience
using GramSchmidtOrthogonalizerFloat = GramSchmidtOrthogonalizer<float>;
using GramSchmidtOrthogonalizerDouble = GramSchmidtOrthogonalizer<double>;

/**
 * @brief Benchmark different memory schemes for Gram-Schmidt orthogonalization
 * @tparam T Floating point type (float or double)
 * @param m Number of rows
 * @param n Number of columns
 * @param num_runs Number of benchmark runs
 */
template<typename T>
void benchmarkGramSchmidt(int m, int n, int num_runs = 5);

/**
 * @brief Verify orthogonality of result matrix
 * @tparam T Floating point type (float or double)
 * @param matrix Column-major orthogonal matrix
 * @param m Number of rows
 * @param n Number of columns
 * @param tolerance Tolerance for orthogonality check
 * @return true if matrix is orthogonal within tolerance
 */
template<typename T>
bool verifyOrthogonality(const T* matrix, int m, int n, T tolerance = static_cast<T>(1e-5));

/**
 * @brief Generate random column-major matrix for testing
 * @tparam T Floating point type (float or double)
 * @param matrix Output matrix (pre-allocated)
 * @param m Number of rows
 * @param n Number of columns
 * @param seed Random seed
 */
template<typename T>
void generateRandomMatrix(T* matrix, int m, int n, unsigned int seed = 42);

// Explicit instantiation declarations
extern template class GramSchmidtOrthogonalizer<float>;
extern template class GramSchmidtOrthogonalizer<double>;

extern template void benchmarkGramSchmidt<float>(int m, int n, int num_runs);
extern template void benchmarkGramSchmidt<double>(int m, int n, int num_runs);

extern template bool verifyOrthogonality<float>(const float* matrix, int m, int n, float tolerance);
extern template bool verifyOrthogonality<double>(const double* matrix, int m, int n, double tolerance);

extern template void generateRandomMatrix<float>(float* matrix, int m, int n, unsigned int seed);
extern template void generateRandomMatrix<double>(double* matrix, int m, int n, unsigned int seed);

} // namespace gram_schmidt