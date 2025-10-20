#include "gram_schmidt.h"
#include <iostream>
#include <stdexcept>
#include <chrono>
#include <random>
#include <cmath>
#include <iomanip>
#include <cstring>

namespace gram_schmidt {

// CUDA kernels for device-only operations
template<typename T>
__global__ void reciprocalKernel(T* value) {
    *value = static_cast<T>(1.0) / (*value);
}

template<typename T>
__global__ void checkThresholdKernel(const T* value, T threshold, int* result) {
    *result = (*value > threshold) ? 1 : 0;
}

// Helper trait for cuBLAS function selection
template<typename T>
struct CublasTraits;

template<>
struct CublasTraits<float> {
    static cublasStatus_t dot(cublasHandle_t handle, int n, const float* x, int incx, const float* y, int incy, float* result) {
        return cublasSdot(handle, n, x, incx, y, incy, result);
    }
    
    static cublasStatus_t nrm2(cublasHandle_t handle, int n, const float* x, int incx, float* result) {
        return cublasSnrm2(handle, n, x, incx, result);
    }
    
    static cublasStatus_t scal(cublasHandle_t handle, int n, const float* alpha, float* x, int incx) {
        return cublasSscal(handle, n, alpha, x, incx);
    }
    
    static cublasStatus_t axpy(cublasHandle_t handle, int n, const float* alpha, const float* x, int incx, float* y, int incy) {
        return cublasSaxpy(handle, n, alpha, x, incx, y, incy);
    }
};

template<>
struct CublasTraits<double> {
    static cublasStatus_t dot(cublasHandle_t handle, int n, const double* x, int incx, const double* y, int incy, double* result) {
        return cublasDdot(handle, n, x, incx, y, incy, result);
    }
    
    static cublasStatus_t nrm2(cublasHandle_t handle, int n, const double* x, int incx, double* result) {
        return cublasDnrm2(handle, n, x, incx, result);
    }
    
    static cublasStatus_t scal(cublasHandle_t handle, int n, const double* alpha, double* x, int incx) {
        return cublasDscal(handle, n, alpha, x, incx);
    }
    
    static cublasStatus_t axpy(cublasHandle_t handle, int n, const double* alpha, const double* x, int incx, double* y, int incy) {
        return cublasDaxpy(handle, n, alpha, x, incx, y, incy);
    }
};

// CublasHandle implementation (unchanged)
CublasHandle::CublasHandle() {
    cublasStatus_t status = cublasCreate(&handle_);
    if (status != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error("Failed to create cuBLAS handle: " + std::to_string(status));
    }
}

CublasHandle::~CublasHandle() {
    if (handle_) {
        cublasDestroy(handle_);
    }
}

CublasHandle::CublasHandle(CublasHandle&& other) noexcept : handle_(other.handle_) {
    other.handle_ = nullptr;
}

CublasHandle& CublasHandle::operator=(CublasHandle&& other) noexcept {
    if (this != &other) {
        if (handle_) {
            cublasDestroy(handle_);
        }
        handle_ = other.handle_;
        other.handle_ = nullptr;
    }
    return *this;
}

// GramSchmidtOrthogonalizer implementation
template<typename T>
GramSchmidtOrthogonalizer<T>::GramSchmidtOrthogonalizer(int m, int n, MemoryScheme scheme)
    : m_(m), n_(n), scheme_(scheme), device_matrix_(nullptr), result_matrix_(nullptr),
      host_result_(nullptr), temp_vector_(nullptr), d_dot_result_(nullptr),
      d_norm_result_(nullptr), d_scale_factor_(nullptr), d_projection_(nullptr), 
      d_neg_one_(nullptr), d_threshold_result_(nullptr), stream_(0) {
    
    if (m <= 0 || n <= 0) {
        throw std::invalid_argument("Matrix dimensions must be positive");
    }
    
    // Create CUDA stream
    cudaStreamCreate(&stream_);
    cublasSetStream(cublas_handle_.get(), stream_);
    
    allocateMemory();
}

template<typename T>
GramSchmidtOrthogonalizer<T>::~GramSchmidtOrthogonalizer() {
    deallocateMemory();
    if (stream_) {
        cudaStreamDestroy(stream_);
    }
}

template<typename T>
void GramSchmidtOrthogonalizer<T>::allocateMemory() {
    size_t matrix_size = m_ * n_ * sizeof(T);
    size_t vector_size = m_ * sizeof(T);
    
    // Allocate device matrix for computation
    cudaMalloc(&device_matrix_, matrix_size);
    
    // Allocate temporary vector for operations
    cudaMalloc(&temp_vector_, vector_size);
    
    // Always allocate device storage for intermediate scalar values (unified approach)
    cudaMalloc(&d_dot_result_, sizeof(T));
    cudaMalloc(&d_norm_result_, sizeof(T));
    cudaMalloc(&d_scale_factor_, sizeof(T));
    cudaMalloc(&d_projection_, sizeof(T));
    cudaMalloc(&d_neg_one_, sizeof(T));
    cudaMalloc(&d_threshold_result_, sizeof(int));
    
    // Initialize the -1 constant on device
    T neg_one_host = static_cast<T>(-1.0);
    cudaMemcpy(d_neg_one_, &neg_one_host, sizeof(T), cudaMemcpyHostToDevice);
    
    // Allocate result matrix based on memory scheme
    switch (scheme_) {
        case MemoryScheme::DEVICE_MEMORY:
            result_matrix_ = device_matrix_; // Result stays on device
            break;
            
        case MemoryScheme::HOST_MEMORY:
            host_result_ = new T[m_ * n_];
            result_matrix_ = host_result_;
            break;
            
        case MemoryScheme::UNIFIED_MEMORY:
            cudaMallocManaged(&result_matrix_, matrix_size);
            break;
    }
}

template<typename T>
void GramSchmidtOrthogonalizer<T>::deallocateMemory() {
    if (device_matrix_ && scheme_ != MemoryScheme::DEVICE_MEMORY) {
        cudaFree(device_matrix_);
    }
    
    if (temp_vector_) {
        cudaFree(temp_vector_);
    }
    
    // Always free device storage for intermediate scalar values
    if (d_dot_result_) {
        cudaFree(d_dot_result_);
    }
    if (d_norm_result_) {
        cudaFree(d_norm_result_);
    }
    if (d_scale_factor_) {
        cudaFree(d_scale_factor_);
    }
    if (d_projection_) {
        cudaFree(d_projection_);
    }
    if (d_neg_one_) {
        cudaFree(d_neg_one_);
    }
    if (d_threshold_result_) {
        cudaFree(d_threshold_result_);
    }
    
    switch (scheme_) {
        case MemoryScheme::DEVICE_MEMORY:
            if (device_matrix_) {
                cudaFree(device_matrix_);
            }
            break;
            
        case MemoryScheme::HOST_MEMORY:
            delete[] host_result_;
            break;
            
        case MemoryScheme::UNIFIED_MEMORY:
            if (result_matrix_) {
                cudaFree(result_matrix_);
            }
            break;
    }
}

template<typename T>
double GramSchmidtOrthogonalizer<T>::orthogonalize(const T* input_matrix, T** output_matrix) {
    // Copy input to device
    copyInputToDevice(input_matrix);
    
    // Perform orthogonalization and measure time
    double elapsed_time = performOrthogonalization();
    
    // Handle result based on memory scheme
    handleResult();
    
    // Set output pointer
    *output_matrix = result_matrix_;
    
    return elapsed_time;
}

template<typename T>
void GramSchmidtOrthogonalizer<T>::copyInputToDevice(const T* input_matrix) {
    size_t matrix_size = m_ * n_ * sizeof(T);
    
    if (scheme_ == MemoryScheme::UNIFIED_MEMORY) {
        // Copy to unified memory
        std::memcpy(result_matrix_, input_matrix, matrix_size);
        // Copy to device working matrix
        cudaMemcpy(device_matrix_, input_matrix, matrix_size, cudaMemcpyHostToDevice);
    } else {
        cudaMemcpy(device_matrix_, input_matrix, matrix_size, cudaMemcpyHostToDevice);
    }
}

template<typename T>
double GramSchmidtOrthogonalizer<T>::performOrthogonalization() {
    auto start = std::chrono::high_resolution_clock::now();
    
    // Use device-optimized version for DEVICE_MEMORY and UNIFIED_MEMORY
    if (scheme_ == MemoryScheme::DEVICE_MEMORY || scheme_ == MemoryScheme::UNIFIED_MEMORY) {
        // Set cuBLAS to device pointer mode for device-only computation
        cublasSetPointerMode(cublas_handle_.get(), CUBLAS_POINTER_MODE_DEVICE);
        
        // Classical Gram-Schmidt process with device-only intermediate values
        for (int j = 0; j < n_; ++j) {
            // Orthogonalize column j against all previous columns
            for (int i = 0; i < j; ++i) {
                // Compute projection coefficient: <v_j, q_i> (stored on device)
                computeDotProductDevice(j, i, d_projection_);
                
                // Subtract projection: v_j = v_j - projection * q_i
                subtractProjectionDevice(j, i, d_projection_);
            }
            
            // Normalize column j
            computeNormDevice(j, d_norm_result_);
            
            // Check if norm is above threshold and compute reciprocal if so
            if (checkNormThresholdDevice(d_norm_result_, static_cast<T>(1e-10))) {
                reciprocalDevice(d_norm_result_);  // Convert norm to 1/norm
                scaleColumnDevice(j, d_norm_result_);
            }
        }
        
        // Reset cuBLAS to host pointer mode
        cublasSetPointerMode(cublas_handle_.get(), CUBLAS_POINTER_MODE_HOST);
        
    } else {
        // Original host-based version for HOST_MEMORY scheme
        for (int j = 0; j < n_; ++j) {
            // Orthogonalize column j against all previous columns
            for (int i = 0; i < j; ++i) {
                // Compute projection coefficient: <v_j, q_i>
                T projection = computeDotProduct(j, i);
                
                // Subtract projection: v_j = v_j - projection * q_i
                subtractProjection(j, i, projection);
            }
            
            // Normalize column j
            T norm = computeNorm(j);
            if (norm > static_cast<T>(1e-10)) {  // Avoid division by zero
                scaleColumn(j, static_cast<T>(1.0) / norm);
            }
        }
    }
    
    // Synchronize to ensure all operations are complete
    cudaStreamSynchronize(stream_);
    
    auto end = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double, std::milli>(end - start).count();
}

template<typename T>
void GramSchmidtOrthogonalizer<T>::handleResult() {
    size_t matrix_size = m_ * n_ * sizeof(T);
    
    switch (scheme_) {
        case MemoryScheme::DEVICE_MEMORY:
            // Result already on device (device_matrix_ == result_matrix_)
            break;
            
        case MemoryScheme::HOST_MEMORY:
            // Copy result back to host
            cudaMemcpy(host_result_, device_matrix_, matrix_size, cudaMemcpyDeviceToHost);
            break;
            
        case MemoryScheme::UNIFIED_MEMORY:
            // Copy from device working matrix to unified memory result
            cudaMemcpy(result_matrix_, device_matrix_, matrix_size, cudaMemcpyDeviceToDevice);
            // Synchronize to ensure copy is complete
            cudaStreamSynchronize(stream_);
            break;
    }
}

template<typename T>
T GramSchmidtOrthogonalizer<T>::computeDotProduct(int col1, int col2) {
    T result;
    
    // Get pointers to the columns
    const T* x = device_matrix_ + col1 * m_;  // Column col1
    const T* y = device_matrix_ + col2 * m_;  // Column col2
    
    // Compute dot product using cuBLAS
    cublasStatus_t status = CublasTraits<T>::dot(cublas_handle_.get(), m_, x, 1, y, 1, &result);
    
    if (status != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error("cuBLAS dot product failed: " + std::to_string(status));
    }
    
    return result;
}

template<typename T>
T GramSchmidtOrthogonalizer<T>::computeNorm(int col_idx) {
    T result;
    
    // Get pointer to the column
    const T* x = device_matrix_ + col_idx * m_;
    
    // Compute L2 norm using cuBLAS
    cublasStatus_t status = CublasTraits<T>::nrm2(cublas_handle_.get(), m_, x, 1, &result);
    
    if (status != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error("cuBLAS norm computation failed: " + std::to_string(status));
    }
    
    return result;
}

template<typename T>
void GramSchmidtOrthogonalizer<T>::scaleColumn(int col_idx, T scale_factor) {
    // Get pointer to the column
    T* x = device_matrix_ + col_idx * m_;
    
    // Scale the column using cuBLAS
    cublasStatus_t status = CublasTraits<T>::scal(cublas_handle_.get(), m_, &scale_factor, x, 1);
    
    if (status != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error("cuBLAS scale operation failed: " + std::to_string(status));
    }
}

template<typename T>
void GramSchmidtOrthogonalizer<T>::subtractProjection(int target_col, int basis_col, T projection) {
    // Get pointers to the columns
    T* y = device_matrix_ + target_col * m_;        // Target column
    const T* x = device_matrix_ + basis_col * m_;   // Basis column
    
    // Subtract projection using cuBLAS: y = y - projection * x
    T neg_projection = -projection;
    cublasStatus_t status = CublasTraits<T>::axpy(cublas_handle_.get(), m_, &neg_projection, x, 1, y, 1);
    
    if (status != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error("cuBLAS axpy operation failed: " + std::to_string(status));
    }
}

// Device-only versions for DEVICE_MEMORY and UNIFIED_MEMORY schemes
template<typename T>
void GramSchmidtOrthogonalizer<T>::computeDotProductDevice(int col1, int col2, T* d_result) {
    // Get pointers to the columns
    const T* x = device_matrix_ + col1 * m_;  // Column col1
    const T* y = device_matrix_ + col2 * m_;  // Column col2
    
    // Compute dot product using cuBLAS (result stored on device)
    cublasStatus_t status = CublasTraits<T>::dot(cublas_handle_.get(), m_, x, 1, y, 1, d_result);
    
    if (status != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error("cuBLAS device dot product failed: " + std::to_string(status));
    }
}

template<typename T>
void GramSchmidtOrthogonalizer<T>::computeNormDevice(int col_idx, T* d_result) {
    // Get pointer to the column
    const T* x = device_matrix_ + col_idx * m_;
    
    // Compute L2 norm using cuBLAS (result stored on device)
    cublasStatus_t status = CublasTraits<T>::nrm2(cublas_handle_.get(), m_, x, 1, d_result);
    
    if (status != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error("cuBLAS device norm computation failed: " + std::to_string(status));
    }
}

template<typename T>
void GramSchmidtOrthogonalizer<T>::scaleColumnDevice(int col_idx, T* d_scale_factor) {
    // Get pointer to the column
    T* x = device_matrix_ + col_idx * m_;
    
    // Scale the column using cuBLAS (scale factor from device memory)
    cublasStatus_t status = CublasTraits<T>::scal(cublas_handle_.get(), m_, d_scale_factor, x, 1);
    
    if (status != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error("cuBLAS device scale operation failed: " + std::to_string(status));
    }
}

template<typename T>
void GramSchmidtOrthogonalizer<T>::subtractProjectionDevice(int target_col, int basis_col, T* d_projection) {
    // Get pointers to the columns
    T* y = device_matrix_ + target_col * m_;        // Target column
    const T* x = device_matrix_ + basis_col * m_;   // Basis column
    
    // We need to negate the projection value on device
    // Copy projection to d_scale_factor_ and negate it
    cudaMemcpy(d_scale_factor_, d_projection, sizeof(T), cudaMemcpyDeviceToDevice);
    
    // Negate the value using cuBLAS scal with pre-stored device -1
    CublasTraits<T>::scal(cublas_handle_.get(), 1, d_neg_one_, d_scale_factor_, 1);
    
    // Subtract projection using cuBLAS: y = y + d_scale_factor_ * x (where d_scale_factor_ = -projection)
    cublasStatus_t status = CublasTraits<T>::axpy(cublas_handle_.get(), m_, d_scale_factor_, x, 1, y, 1);
    
    if (status != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error("cuBLAS device axpy operation failed: " + std::to_string(status));
    }
}

template<typename T>
bool GramSchmidtOrthogonalizer<T>::checkNormThresholdDevice(T* d_norm, T threshold) {
    // Launch kernel to check threshold using pre-allocated device memory
    checkThresholdKernel<<<1, 1, 0, stream_>>>(d_norm, threshold, d_threshold_result_);
    
    // Copy result back to host
    int host_result;
    cudaMemcpy(&host_result, d_threshold_result_, sizeof(int), cudaMemcpyDeviceToHost);
    
    return host_result == 1;
}

template<typename T>
void GramSchmidtOrthogonalizer<T>::reciprocalDevice(T* d_value) {
    // Launch kernel to compute reciprocal
    reciprocalKernel<<<1, 1, 0, stream_>>>(d_value);
    
    // Check for kernel launch errors
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        throw std::runtime_error("Reciprocal kernel launch failed: " + std::string(cudaGetErrorString(error)));
    }
}

template<typename T>
void GramSchmidtOrthogonalizer<T>::copyResultToHost(T* host_matrix) const {
    if (scheme_ == MemoryScheme::DEVICE_MEMORY) {
        size_t matrix_size = m_ * n_ * sizeof(T);
        cudaMemcpy(host_matrix, result_matrix_, matrix_size, cudaMemcpyDeviceToHost);
    } else {
        // For other schemes, result is already accessible from host
        std::memcpy(host_matrix, result_matrix_, m_ * n_ * sizeof(T));
    }
}

// Utility functions
template<typename T>
void benchmarkGramSchmidt(int m, int n, int num_runs) {
    std::cout << std::fixed << std::setprecision(3);
    std::cout << "\n=== Gram-Schmidt Orthogonalization Benchmark (" << 
                 (sizeof(T) == 4 ? "float" : "double") << ") ===\n";
    std::cout << "Matrix size: " << m << " x " << n << "\n";
    std::cout << "Number of runs: " << num_runs << "\n\n";
    
    // Generate test matrix
    std::vector<T> input_matrix(m * n);
    generateRandomMatrix(input_matrix.data(), m, n);
    
    std::vector<MemoryScheme> schemes = {
        MemoryScheme::DEVICE_MEMORY,
        MemoryScheme::HOST_MEMORY,
        MemoryScheme::UNIFIED_MEMORY
    };
    
    std::vector<std::string> scheme_names = {
        "Device Memory",
        "Host Memory",
        "Unified Memory"
    };
    
    for (size_t i = 0; i < schemes.size(); ++i) {
        std::cout << "Testing " << scheme_names[i] << ":\n";
        
        double total_time = 0.0;
        bool orthogonality_verified = false;
        
        for (int run = 0; run < num_runs; ++run) {
            try {
                GramSchmidtOrthogonalizer<T> orthogonalizer(m, n, schemes[i]);
                T* result_matrix;
                double elapsed = orthogonalizer.orthogonalize(input_matrix.data(), &result_matrix);
                total_time += elapsed;
                
                // Verify orthogonality on the first run
                if (run == 0) {
                    std::vector<T> host_result(m * n);
                    if (schemes[i] == MemoryScheme::DEVICE_MEMORY) {
                        orthogonalizer.copyResultToHost(host_result.data());
                    } else {
                        std::memcpy(host_result.data(), result_matrix, m * n * sizeof(T));
                    }
                    orthogonality_verified = verifyOrthogonality(host_result.data(), m, n);
                }
                
                std::cout << "  Run " << (run + 1) << ": " << elapsed << " ms\n";
                
            } catch (const std::exception& e) {
                std::cerr << "  Error in run " << (run + 1) << ": " << e.what() << "\n";
            }
        }
        
        double avg_time = total_time / num_runs;
        std::cout << "  Average: " << avg_time << " ms\n";
        std::cout << "  Orthogonality check: " << (orthogonality_verified ? "PASS" : "FAIL") << "\n\n";
    }
}

template<typename T>
bool verifyOrthogonality(const T* matrix, int m, int n, T tolerance) {
    // Check if Q^T * Q = I (approximately)
    std::vector<T> qtq(n * n, static_cast<T>(0.0));
    
    // Compute Q^T * Q using simple CPU implementation
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            T dot_product = static_cast<T>(0.0);
            for (int k = 0; k < m; ++k) {
                dot_product += matrix[i * m + k] * matrix[j * m + k];  // Column-major access
            }
            qtq[i * n + j] = dot_product;
        }
    }
    
    // Check if result is close to identity matrix
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            T expected = (i == j) ? static_cast<T>(1.0) : static_cast<T>(0.0);
            T actual = qtq[i * n + j];
            if (std::abs(actual - expected) > tolerance) {
                return false;
            }
        }
    }
    
    return true;
}

template<typename T>
void generateRandomMatrix(T* matrix, int m, int n, unsigned int seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<T> dist(static_cast<T>(-1.0), static_cast<T>(1.0));
    
    for (int i = 0; i < m * n; ++i) {
        matrix[i] = dist(rng);
    }
}

// Explicit template instantiations
template class GramSchmidtOrthogonalizer<float>;
template class GramSchmidtOrthogonalizer<double>;

template void benchmarkGramSchmidt<float>(int m, int n, int num_runs);
template void benchmarkGramSchmidt<double>(int m, int n, int num_runs);

template bool verifyOrthogonality<float>(const float* matrix, int m, int n, float tolerance);
template bool verifyOrthogonality<double>(const double* matrix, int m, int n, double tolerance);

template void generateRandomMatrix<float>(float* matrix, int m, int n, unsigned int seed);
template void generateRandomMatrix<double>(double* matrix, int m, int n, unsigned int seed);

} // namespace gram_schmidt