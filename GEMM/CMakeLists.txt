cmake_minimum_required(VERSION 3.18)
project(GEMM LANGUAGES CXX)

# ============================================================================
# Benchmark Executable
# ============================================================================

if(CXX_LEARNING_HAS_BENCHMARK AND CXX_LEARNING_HAS_GTEST)
  add_executable(GEMM_bench bench.cpp)
  target_link_libraries(GEMM_bench PRIVATE 
    benchmark::benchmark
    GTest::gtest_main
  )

  # Enable SIMD optimizations
  if(COMMAND target_enable_simd)
    target_enable_simd(GEMM_bench)
  endif()
endif()

# ============================================================================
# Test Executable
# ============================================================================

if(CXX_LEARNING_HAS_GTEST)
  add_executable(GEMM_test test.cpp)
  target_link_libraries(GEMM_test PRIVATE 
    GTest::gtest_main
  )

  # Enable SIMD optimizations
  if(COMMAND target_enable_simd)
    target_enable_simd(GEMM_test)
  endif()

  add_test(NAME GEMM_test COMMAND GEMM_test)
endif()
