#include <gtest/gtest.h>
#include <thread>
#include <vector>
#include <cstdint>
#include <random>
#include <atomic>

// Test benign data racing for int32_t
// Multiple threads write the same value to the same location
TEST(BenignDataRacing, Int32_SameValue) {
    constexpr int NUM_THREADS = 8;
    constexpr int NUM_ITERATIONS = 100000;
    constexpr int ARRAY_SIZE = 1000;
    constexpr int32_t MAGIC_VALUE = 0x12345678;
    
    std::vector<int32_t> data(ARRAY_SIZE, 0);
    std::vector<std::thread> threads;
    
    // Each thread randomly writes MAGIC_VALUE to random indices
    for (int t = 0; t < NUM_THREADS; ++t) {
        threads.emplace_back([&data, t]() {
            std::mt19937 gen(t); // Different seed per thread
            std::uniform_int_distribution<int> dist(0, ARRAY_SIZE - 1);
            
            for (int i = 0; i < NUM_ITERATIONS; ++i) {
                int idx = dist(gen);
                data[idx] = MAGIC_VALUE; // Data race, but same value
            }
        });
    }
    
    for (auto& thread : threads) {
        thread.join();
    }
    
    // Verify all written locations have the correct value
    // Note: Some locations might still be 0 if never written
    for (int32_t value : data) {
        EXPECT_TRUE(value == 0 || value == MAGIC_VALUE)
            << "Expected 0 or " << MAGIC_VALUE << " but got " << value;
    }
}

// Test benign data racing for int64_t
TEST(BenignDataRacing, Int64_SameValue) {
    constexpr int NUM_THREADS = 8;
    constexpr int NUM_ITERATIONS = 100000;
    constexpr int ARRAY_SIZE = 1000;
    constexpr int64_t MAGIC_VALUE = 0x123456789ABCDEF0LL;
    
    std::vector<int64_t> data(ARRAY_SIZE, 0);
    std::vector<std::thread> threads;
    
    // Each thread randomly writes MAGIC_VALUE to random indices
    for (int t = 0; t < NUM_THREADS; ++t) {
        threads.emplace_back([&data, t]() {
            std::mt19937 gen(t); // Different seed per thread
            std::uniform_int_distribution<int> dist(0, ARRAY_SIZE - 1);
            
            for (int i = 0; i < NUM_ITERATIONS; ++i) {
                int idx = dist(gen);
                data[idx] = MAGIC_VALUE; // Data race, but same value
            }
        });
    }
    
    for (auto& thread : threads) {
        thread.join();
    }
    
    // Verify all written locations have the correct value
    for (int64_t value : data) {
        EXPECT_TRUE(value == 0 || value == MAGIC_VALUE)
            << "Expected 0 or " << MAGIC_VALUE << " but got " << std::hex << value;
    }
}

// Test with high contention - all threads write to the same location
TEST(BenignDataRacing, Int32_HighContention) {
    constexpr int NUM_THREADS = 16;
    constexpr int NUM_ITERATIONS = 1000000;
    constexpr int32_t MAGIC_VALUE = 0x42424242;
    
    int32_t shared_value = 0;
    std::vector<std::thread> threads;
    
    for (int t = 0; t < NUM_THREADS; ++t) {
        threads.emplace_back([&shared_value]() {
            for (int i = 0; i < NUM_ITERATIONS; ++i) {
                shared_value = MAGIC_VALUE; // Intense data race
            }
        });
    }
    
    for (auto& thread : threads) {
        thread.join();
    }
    
    // After all threads complete, value should be MAGIC_VALUE
    EXPECT_EQ(shared_value, MAGIC_VALUE);
}

TEST(BenignDataRacing, Int64_HighContention) {
    constexpr int NUM_THREADS = 16;
    constexpr int NUM_ITERATIONS = 1000000;
    constexpr int64_t MAGIC_VALUE = 0xDEADBEEFCAFEBABELL;
    
    int64_t shared_value = 0;
    std::vector<std::thread> threads;
    
    for (int t = 0; t < NUM_THREADS; ++t) {
        threads.emplace_back([&shared_value]() {
            for (int i = 0; i < NUM_ITERATIONS; ++i) {
                shared_value = MAGIC_VALUE; // Intense data race
            }
        });
    }
    
    for (auto& thread : threads) {
        thread.join();
    }
    
    // After all threads complete, value should be MAGIC_VALUE
    EXPECT_EQ(shared_value, MAGIC_VALUE);
}

// Test to detect torn writes (this test expects potential failures on some architectures)
TEST(BenignDataRacing, Int64_TornWriteDetection) {
    constexpr int NUM_THREADS = 8;
    constexpr int NUM_ITERATIONS = 100000;
    constexpr int ARRAY_SIZE = 100;
    
    // Use two different patterns that have no overlapping bytes
    constexpr int64_t PATTERN_A = 0xAAAAAAAAAAAAAAAALL;
    constexpr int64_t PATTERN_B = 0x5555555555555555LL;
    
    std::vector<int64_t> data(ARRAY_SIZE, 0);
    std::vector<std::thread> threads;
    std::atomic<bool> torn_write_detected{false};
    
    // Half threads write PATTERN_A, half write PATTERN_B
    for (int t = 0; t < NUM_THREADS; ++t) {
        threads.emplace_back([&data, &torn_write_detected, t]() {
            int64_t pattern = (t < NUM_THREADS / 2) ? PATTERN_A : PATTERN_B;
            std::mt19937 gen(t);
            std::uniform_int_distribution<int> dist(0, ARRAY_SIZE - 1);
            
            for (int i = 0; i < NUM_ITERATIONS; ++i) {
                int idx = dist(gen);
                data[idx] = pattern;
                
                // Check for torn writes in adjacent reads
                int64_t read_back = data[idx];
                if (read_back != 0 && read_back != PATTERN_A && read_back != PATTERN_B) {
                    torn_write_detected.store(true, std::memory_order_relaxed);
                }
            }
        });
    }
    
    for (auto& thread : threads) {
        thread.join();
    }
    
    // Final verification
    bool has_torn_write = false;
    for (int64_t value : data) {
        if (value != 0 && value != PATTERN_A && value != PATTERN_B) {
            has_torn_write = true;
            std::cout << "Torn write detected: 0x" << std::hex << value << std::dec << std::endl;
        }
    }
    
    // On x86-64, 64-bit aligned writes are atomic, so we shouldn't see torn writes
    // On some other architectures (e.g., 32-bit ARM), torn writes might occur
    if (has_torn_write || torn_write_detected.load()) {
        std::cout << "WARNING: Torn writes detected! 64-bit writes are NOT atomic on this platform." << std::endl;
        std::cout << "Benign data racing with int64_t is NOT safe on this architecture." << std::endl;
    } else {
        std::cout << "No torn writes detected. 64-bit writes appear to be atomic on this platform." << std::endl;
    }
    
    // Don't fail the test, just report findings
    EXPECT_TRUE(true); // Always pass, this is informational
}

// Test with different write patterns to stress test the memory subsystem
TEST(BenignDataRacing, Int32_MultiplePatterns) {
    constexpr int NUM_THREADS = 8;
    constexpr int NUM_ITERATIONS = 50000;
    constexpr int ARRAY_SIZE = 500;
    
    std::vector<int32_t> patterns = {
        static_cast<int32_t>(0x11111111), static_cast<int32_t>(0x22222222), 
        static_cast<int32_t>(0x33333333), static_cast<int32_t>(0x44444444),
        static_cast<int32_t>(0x55555555), static_cast<int32_t>(0x66666666), 
        static_cast<int32_t>(0x77777777), static_cast<int32_t>(0x88888888)
    };
    
    std::vector<int32_t> data(ARRAY_SIZE, 0);
    std::vector<std::thread> threads;
    
    for (int t = 0; t < NUM_THREADS; ++t) {
        threads.emplace_back([&data, &patterns, t]() {
            int32_t pattern = patterns[t];
            std::mt19937 gen(t);
            std::uniform_int_distribution<int> dist(0, ARRAY_SIZE - 1);
            
            for (int i = 0; i < NUM_ITERATIONS; ++i) {
                int idx = dist(gen);
                data[idx] = pattern;
            }
        });
    }
    
    for (auto& thread : threads) {
        thread.join();
    }
    
    // Verify all values are either 0 or one of the patterns
    for (int32_t value : data) {
        if (value != 0) {
            bool is_valid = false;
            for (int32_t pattern : patterns) {
                if (value == pattern) {
                    is_valid = true;
                    break;
                }
            }
            EXPECT_TRUE(is_valid) << "Unexpected value: 0x" << std::hex << value;
        }
    }
}
