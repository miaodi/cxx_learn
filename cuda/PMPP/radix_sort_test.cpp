#include "radix_sort.cuh"

#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <random>
#include <vector>

using namespace pmpp::radix_sort;

class RadixSortTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Set random seed for reproducibility
        rng.seed(42);
    }

    std::mt19937 rng;
};

TEST_F(RadixSortTest, EmptyArray) {
    std::vector<uint32_t> input;
    std::vector<uint32_t> output;
    
    radix_sort_host<4>(input.data(), output.data(), 0);
    
    EXPECT_TRUE(output.empty());
}

TEST_F(RadixSortTest, SingleElement) {
    std::vector<uint32_t> input = {42};
    std::vector<uint32_t> output(1);
    
    radix_sort_host<4>(input.data(), output.data(), input.size());
    
    EXPECT_EQ(output[0], 42);
}

TEST_F(RadixSortTest, TwoElements) {
    std::vector<uint32_t> input = {10, 5};
    std::vector<uint32_t> output(2);
    
    radix_sort_host<4>(input.data(), output.data(), input.size());
    
    EXPECT_EQ(output[0], 5);
    EXPECT_EQ(output[1], 10);
}

TEST_F(RadixSortTest, AlreadySorted) {
    std::vector<uint32_t> input = {1, 2, 3, 4, 5};
    std::vector<uint32_t> output(5);
    
    radix_sort_host<4>(input.data(), output.data(), input.size());
    
    for (size_t i = 0; i < input.size(); i++) {
        EXPECT_EQ(output[i], i + 1);
    }
}

TEST_F(RadixSortTest, ReverseSorted) {
    std::vector<uint32_t> input = {5, 4, 3, 2, 1};
    std::vector<uint32_t> output(5);
    
    radix_sort_host<4>(input.data(), output.data(), input.size());
    
    for (size_t i = 0; i < input.size(); i++) {
        EXPECT_EQ(output[i], i + 1);
    }
}

TEST_F(RadixSortTest, DuplicateElements) {
    std::vector<uint32_t> input = {3, 1, 4, 1, 5, 9, 2, 6, 5, 3};
    std::vector<uint32_t> output(input.size());
    std::vector<uint32_t> expected = input;
    std::sort(expected.begin(), expected.end());
    
    radix_sort_host<4>(input.data(), output.data(), input.size());
    
    for (size_t i = 0; i < input.size(); i++) {
        EXPECT_EQ(output[i], expected[i]);
    }
}

TEST_F(RadixSortTest, SmallRandom) {
    const size_t n = 100;
    std::vector<uint32_t> input(n);
    std::uniform_int_distribution<uint32_t> dist(0, 1000);
    
    for (size_t i = 0; i < n; i++) {
        input[i] = dist(rng);
    }
    
    std::vector<uint32_t> output(n);
    std::vector<uint32_t> expected = input;
    std::sort(expected.begin(), expected.end());
    
    radix_sort_host<4>(input.data(), output.data(), input.size());
    
    for (size_t i = 0; i < n; i++) {
        EXPECT_EQ(output[i], expected[i]) << "Mismatch at index " << i;
    }
}

TEST_F(RadixSortTest, MediumRandom) {
    const size_t n = 10000;
    std::vector<uint32_t> input(n);
    std::uniform_int_distribution<uint32_t> dist(0, UINT32_MAX);
    
    for (size_t i = 0; i < n; i++) {
        input[i] = dist(rng);
    }
    
    std::vector<uint32_t> output(n);
    std::vector<uint32_t> expected = input;
    std::sort(expected.begin(), expected.end());
    
    radix_sort_host<4>(input.data(), output.data(), input.size());
    
    for (size_t i = 0; i < n; i++) {
        EXPECT_EQ(output[i], expected[i]) << "Mismatch at index " << i;
    }
}

TEST_F(RadixSortTest, LargeRandom) {
    const size_t n = 1000000;
    std::vector<uint32_t> input(n);
    std::uniform_int_distribution<uint32_t> dist(0, UINT32_MAX);
    
    for (size_t i = 0; i < n; i++) {
        input[i] = dist(rng);
    }
    
    std::vector<uint32_t> output(n);
    std::vector<uint32_t> expected = input;
    std::sort(expected.begin(), expected.end());
    
    radix_sort_host<4>(input.data(), output.data(), input.size());
    
    for (size_t i = 0; i < n; i++) {
        EXPECT_EQ(output[i], expected[i]) << "Mismatch at index " << i;
    }
}

TEST_F(RadixSortTest, AllSame) {
    const size_t n = 1000;
    std::vector<uint32_t> input(n, 42);
    std::vector<uint32_t> output(n);
    
    radix_sort_host<4>(input.data(), output.data(), input.size());
    
    for (size_t i = 0; i < n; i++) {
        EXPECT_EQ(output[i], 42);
    }
}

TEST_F(RadixSortTest, PowersOfTwo) {
    std::vector<uint32_t> input = {1024, 1, 256, 4, 64, 16, 128, 2, 8, 32, 512, 2048};
    std::vector<uint32_t> output(input.size());
    std::vector<uint32_t> expected = {1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048};
    
    radix_sort_host<4>(input.data(), output.data(), input.size());
    
    for (size_t i = 0; i < input.size(); i++) {
        EXPECT_EQ(output[i], expected[i]);
    }
}

TEST_F(RadixSortTest, MaxValues) {
    std::vector<uint32_t> input = {UINT32_MAX, 0, UINT32_MAX - 1, 1, UINT32_MAX - 2, 2};
    std::vector<uint32_t> output(input.size());
    std::vector<uint32_t> expected = {0, 1, 2, UINT32_MAX - 2, UINT32_MAX - 1, UINT32_MAX};
    
    radix_sort_host<4>(input.data(), output.data(), input.size());
    
    for (size_t i = 0; i < input.size(); i++) {
        EXPECT_EQ(output[i], expected[i]);
    }
}

TEST_F(RadixSortTest, InplaceSort) {
    const size_t n = 10000;
    std::vector<uint32_t> input(n);
    std::uniform_int_distribution<uint32_t> dist(0, UINT32_MAX);
    
    for (size_t i = 0; i < n; i++) {
        input[i] = dist(rng);
    }
    
    std::vector<uint32_t> expected = input;
    std::sort(expected.begin(), expected.end());
    
    // Copy to device
    uint32_t* d_data;
    cudaMalloc(&d_data, n * sizeof(uint32_t));
    cudaMemcpy(d_data, input.data(), n * sizeof(uint32_t), cudaMemcpyHostToDevice);
    
    // Sort in place
    radix_sort_inplace<4>(d_data, n);
    
    // Copy back
    std::vector<uint32_t> output(n);
    cudaMemcpy(output.data(), d_data, n * sizeof(uint32_t), cudaMemcpyDeviceToHost);
    cudaFree(d_data);
    
    for (size_t i = 0; i < n; i++) {
        EXPECT_EQ(output[i], expected[i]) << "Mismatch at index " << i;
    }
}

TEST_F(RadixSortTest, DifferentRadixBits) {
    const size_t n = 1000;
    std::vector<uint32_t> input(n);
    std::uniform_int_distribution<uint32_t> dist(0, UINT32_MAX);
    
    for (size_t i = 0; i < n; i++) {
        input[i] = dist(rng);
    }
    
    std::vector<uint32_t> expected = input;
    std::sort(expected.begin(), expected.end());
    
    // Test with different radix bit sizes (1, 2, 4, 8 only - 16 requires too much shared memory)
    for (int radix_bits : {1, 2, 4, 8}) {
        std::vector<uint32_t> output(n);
        
        switch(radix_bits) {
            case 1:
                radix_sort_host<1>(input.data(), output.data(), n);
                break;
            case 2:
                radix_sort_host<2>(input.data(), output.data(), n);
                break;
            case 4:
                radix_sort_host<4>(input.data(), output.data(), n);
                break;
            case 8:
                radix_sort_host<8>(input.data(), output.data(), n);
                break;
        }
        
        for (size_t i = 0; i < n; i++) {
            EXPECT_EQ(output[i], expected[i]) << "Mismatch at index " << i 
                                               << " with radix_bits=" << radix_bits;
        }
    }
}
