cmake_minimum_required(VERSION 3.18)
project(CrossProductVecLen LANGUAGES CXX)

# Only build executables if AVX2 is supported
if(NOT AVX2_CPU_SUPPORTS)
  message(WARNING "AVX2 is not supported - skipping CrossProductVecLen executables")
  return()
endif()

message(STATUS "AVX2 is supported - building CrossProductVecLen executables")

# ============================================================================
# Library
# ============================================================================

add_library(VectorOps STATIC VectorOps.cpp)
target_include_directories(VectorOps PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

# Enable SIMD optimizations
if(COMMAND target_enable_simd)
  target_enable_simd(VectorOps)
endif()

# ============================================================================
# Benchmark Executables
# ============================================================================

# ElementWiseProductBench
add_executable(ElementWiseProductBench ElementWiseProductBench.cpp)
target_link_libraries(ElementWiseProductBench PRIVATE
  benchmark::benchmark
  GTest::gtest_main
  GTest::gmock_main
  OpenMP::OpenMP_CXX
  VectorOps
)

# CrossProductLenBench
add_executable(CrossProductLenBench CrossProductLenBench.cpp)
target_link_libraries(CrossProductLenBench PRIVATE
  benchmark::benchmark
  GTest::gtest_main
  GTest::gmock_main
  OpenMP::OpenMP_CXX
  VectorOps
)

# VectorOpsBench
add_executable(VectorOpsBench VectorOpsBench.cpp)
target_link_libraries(VectorOpsBench PRIVATE
  benchmark::benchmark
  GTest::gtest_main
  GTest::gmock_main
  OpenMP::OpenMP_CXX
  VectorOps
)

# fmaBench
add_executable(fmaBench fmaBench.cpp)
target_link_libraries(fmaBench PRIVATE
  benchmark::benchmark
  GTest::gtest_main
  GTest::gmock_main
  OpenMP::OpenMP_CXX
  VectorOps
)

# ============================================================================
# Test Executable
# ============================================================================

add_executable(CrossProductVecLen_tests test.cpp)
target_link_libraries(CrossProductVecLen_tests PRIVATE
  GTest::gtest_main
  GTest::gmock_main
  OpenMP::OpenMP_CXX
  VectorOps
)

# Enable testing
enable_testing()
add_test(NAME CrossProductVecLen_tests COMMAND CrossProductVecLen_tests)