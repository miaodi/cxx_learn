#include <gtest/gtest.h>
#include "gram_schmidt.h"
#include <vector>
#include <cmath>

using namespace gram_schmidt;

template<typename T>
class GramSchmidtTemplatedTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Common test setup
    }
    
    static constexpr T tolerance() {
        if constexpr (std::is_same_v<T, float>) {
            return static_cast<T>(1e-4);
        } else {
            return static_cast<T>(1e-10);
        }
    }
};

using TestTypes = ::testing::Types<float, double>;
TYPED_TEST_SUITE(GramSchmidtTemplatedTest, TestTypes);

TYPED_TEST(GramSchmidtTemplatedTest, SmallMatrixDeviceMemory) {
    using T = TypeParam;
    const int m = 6, n = 4;
    
    std::vector<T> input_matrix(m * n);
    generateRandomMatrix(input_matrix.data(), m, n, 12345);
    
    GramSchmidtOrthogonalizer<T> orthogonalizer(m, n, MemoryScheme::DEVICE_MEMORY);
    T* result_matrix;
    double elapsed = orthogonalizer.orthogonalize(input_matrix.data(), &result_matrix);
    
    EXPECT_GT(elapsed, 0.0);
    
    // Copy result to host for verification
    std::vector<T> host_result(m * n);
    orthogonalizer.copyResultToHost(host_result.data());
    
    // Verify orthogonality
    bool is_orthogonal = verifyOrthogonality(host_result.data(), m, n, TestFixture::tolerance());
    EXPECT_TRUE(is_orthogonal);
}

TYPED_TEST(GramSchmidtTemplatedTest, SmallMatrixHostMemory) {
    using T = TypeParam;
    const int m = 6, n = 4;
    
    std::vector<T> input_matrix(m * n);
    generateRandomMatrix(input_matrix.data(), m, n, 12345);
    
    GramSchmidtOrthogonalizer<T> orthogonalizer(m, n, MemoryScheme::HOST_MEMORY);
    T* result_matrix;
    double elapsed = orthogonalizer.orthogonalize(input_matrix.data(), &result_matrix);
    
    EXPECT_GT(elapsed, 0.0);
    
    // Result is already on host
    bool is_orthogonal = verifyOrthogonality(result_matrix, m, n, TestFixture::tolerance());
    EXPECT_TRUE(is_orthogonal);
}

TYPED_TEST(GramSchmidtTemplatedTest, SmallMatrixUnifiedMemory) {
    using T = TypeParam;
    const int m = 6, n = 4;
    
    std::vector<T> input_matrix(m * n);
    generateRandomMatrix(input_matrix.data(), m, n, 12345);
    
    GramSchmidtOrthogonalizer<T> orthogonalizer(m, n, MemoryScheme::UNIFIED_MEMORY);
    T* result_matrix;
    double elapsed = orthogonalizer.orthogonalize(input_matrix.data(), &result_matrix);
    
    EXPECT_GT(elapsed, 0.0);
    
    // Result is accessible from host (unified memory)
    bool is_orthogonal = verifyOrthogonality(result_matrix, m, n, TestFixture::tolerance());
    EXPECT_TRUE(is_orthogonal);
}

TYPED_TEST(GramSchmidtTemplatedTest, SquareMatrix) {
    using T = TypeParam;
    const int n = 32;  // Square matrix
    
    std::vector<T> input_matrix(n * n);
    generateRandomMatrix(input_matrix.data(), n, n, 54321);
    
    GramSchmidtOrthogonalizer<T> orthogonalizer(n, n, MemoryScheme::DEVICE_MEMORY);
    T* result_matrix;
    double elapsed = orthogonalizer.orthogonalize(input_matrix.data(), &result_matrix);
    
    EXPECT_GT(elapsed, 0.0);
    
    std::vector<T> host_result(n * n);
    orthogonalizer.copyResultToHost(host_result.data());
    
    bool is_orthogonal = verifyOrthogonality(host_result.data(), n, n, TestFixture::tolerance());
    EXPECT_TRUE(is_orthogonal);
}

TYPED_TEST(GramSchmidtTemplatedTest, TallMatrix) {
    using T = TypeParam;
    const int m = 128, n = 32;  // Tall matrix
    
    std::vector<T> input_matrix(m * n);
    generateRandomMatrix(input_matrix.data(), m, n, 98765);
    
    GramSchmidtOrthogonalizer<T> orthogonalizer(m, n, MemoryScheme::UNIFIED_MEMORY);
    T* result_matrix;
    double elapsed = orthogonalizer.orthogonalize(input_matrix.data(), &result_matrix);
    
    EXPECT_GT(elapsed, 0.0);
    
    bool is_orthogonal = verifyOrthogonality(result_matrix, m, n, TestFixture::tolerance());
    EXPECT_TRUE(is_orthogonal);
}

TYPED_TEST(GramSchmidtTemplatedTest, InvalidDimensions) {
    using T = TypeParam;
    
    // Test invalid dimensions
    EXPECT_THROW(GramSchmidtOrthogonalizer<T>(0, 5, MemoryScheme::DEVICE_MEMORY), std::invalid_argument);
    EXPECT_THROW(GramSchmidtOrthogonalizer<T>(5, 0, MemoryScheme::HOST_MEMORY), std::invalid_argument);
    EXPECT_THROW(GramSchmidtOrthogonalizer<T>(-1, 5, MemoryScheme::UNIFIED_MEMORY), std::invalid_argument);
}

TYPED_TEST(GramSchmidtTemplatedTest, MemorySchemeConsistency) {
    using T = TypeParam;
    const int m = 16, n = 8;
    
    std::vector<T> input_matrix(m * n);
    generateRandomMatrix(input_matrix.data(), m, n, 11111);
    
    std::vector<MemoryScheme> schemes = {
        MemoryScheme::DEVICE_MEMORY,
        MemoryScheme::HOST_MEMORY,
        MemoryScheme::UNIFIED_MEMORY
    };
    
    std::vector<std::vector<T>> results;
    
    // Test all memory schemes
    for (auto scheme : schemes) {
        GramSchmidtOrthogonalizer<T> orthogonalizer(m, n, scheme);
        T* result_matrix;
        orthogonalizer.orthogonalize(input_matrix.data(), &result_matrix);
        
        std::vector<T> host_result(m * n);
        if (scheme == MemoryScheme::DEVICE_MEMORY) {
            orthogonalizer.copyResultToHost(host_result.data());
        } else {
            std::memcpy(host_result.data(), result_matrix, m * n * sizeof(T));
        }
        
        results.push_back(host_result);
        
        // Verify orthogonality for each scheme
        bool is_orthogonal = verifyOrthogonality(host_result.data(), m, n, TestFixture::tolerance());
        EXPECT_TRUE(is_orthogonal);
    }
    
    // Verify that all schemes produce similar results (within tolerance)
    T comparison_tolerance = static_cast<T>(1e-3);
    for (size_t i = 1; i < results.size(); ++i) {
        for (size_t j = 0; j < results[0].size(); ++j) {
            EXPECT_NEAR(results[0][j], results[i][j], comparison_tolerance)
                << "Mismatch at element " << j << " between schemes";
        }
    }
}

TYPED_TEST(GramSchmidtTemplatedTest, UtilityFunctions) {
    using T = TypeParam;
    const int m = 10, n = 5;
    
    // Test generateRandomMatrix
    std::vector<T> matrix1(m * n);
    std::vector<T> matrix2(m * n);
    generateRandomMatrix(matrix1.data(), m, n, 123);
    generateRandomMatrix(matrix2.data(), m, n, 123);
    
    // Same seed should produce same matrix
    for (size_t i = 0; i < matrix1.size(); ++i) {
        EXPECT_EQ(matrix1[i], matrix2[i]);
    }
    
    // Different seed should produce different matrix
    generateRandomMatrix(matrix2.data(), m, n, 456);
    bool different = false;
    for (size_t i = 0; i < matrix1.size(); ++i) {
        if (matrix1[i] != matrix2[i]) {
            different = true;
            break;
        }
    }
    EXPECT_TRUE(different);
}

// Precision comparison test
TEST(GramSchmidtPrecisionTest, FloatVsDoubleAccuracy) {
    const int m = 64, n = 32;
    
    // Generate the same input for both float and double
    std::vector<float> input_float(m * n);
    std::vector<double> input_double(m * n);
    generateRandomMatrix(input_float.data(), m, n, 42);
    
    // Convert float input to double
    for (size_t i = 0; i < input_float.size(); ++i) {
        input_double[i] = static_cast<double>(input_float[i]);
    }
    
    // Test with device memory scheme
    GramSchmidtOrthogonalizer<float> orthogonalizer_float(m, n, MemoryScheme::DEVICE_MEMORY);
    GramSchmidtOrthogonalizer<double> orthogonalizer_double(m, n, MemoryScheme::DEVICE_MEMORY);
    
    float* result_float;
    double* result_double;
    
    orthogonalizer_float.orthogonalize(input_float.data(), &result_float);
    orthogonalizer_double.orthogonalize(input_double.data(), &result_double);
    
    // Copy results to host
    std::vector<float> host_result_float(m * n);
    std::vector<double> host_result_double(m * n);
    orthogonalizer_float.copyResultToHost(host_result_float.data());
    orthogonalizer_double.copyResultToHost(host_result_double.data());
    
    // Verify both are orthogonal
    bool float_orthogonal = verifyOrthogonality(host_result_float.data(), m, n, 1e-4f);
    bool double_orthogonal = verifyOrthogonality(host_result_double.data(), m, n, 1e-10);
    
    EXPECT_TRUE(float_orthogonal);
    EXPECT_TRUE(double_orthogonal);
    
    // Double precision should be more accurate
    // (This is more of a demonstration than a strict test)
    std::cout << "Float precision orthogonality: " << (float_orthogonal ? "PASS" : "FAIL") << "\n";
    std::cout << "Double precision orthogonality: " << (double_orthogonal ? "PASS" : "FAIL") << "\n";
}

// Performance comparison test (disabled by default)
TEST(GramSchmidtPrecisionTest, DISABLED_PerformanceComparison) {
    const int m = 256, n = 128;
    const int num_runs = 5;
    
    std::cout << "\n=== Performance Comparison: Float vs Double ===\n";
    
    benchmarkGramSchmidt<float>(m, n, num_runs);
    benchmarkGramSchmidt<double>(m, n, num_runs);
}