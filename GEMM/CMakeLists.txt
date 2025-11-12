cmake_minimum_required(VERSION 3.18)
project(GEMM LANGUAGES CXX)

# ============================================================================
# Benchmark Executable
# ============================================================================

add_executable(GEMM_bench bench.cpp)
target_link_libraries(GEMM_bench PRIVATE 
  benchmark::benchmark
  GTest::gtest_main
)

# Enable SIMD optimizations
if(COMMAND target_enable_simd)
  target_enable_simd(GEMM_bench)
endif()

# ============================================================================
# Test Executable
# ============================================================================

add_executable(GEMM_test test.cpp)
target_link_libraries(GEMM_test PRIVATE 
  GTest::gtest_main
)

# Enable SIMD optimizations
if(COMMAND target_enable_simd)
  target_enable_simd(GEMM_test)
endif()

# Enable testing
enable_testing()
add_test(NAME GEMM_test COMMAND GEMM_test)