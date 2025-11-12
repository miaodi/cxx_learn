cmake_minimum_required(VERSION 3.18)
project(misc LANGUAGES CXX)

# ============================================================================
# Library
# ============================================================================

add_library(mylib STATIC func.cpp)
target_include_directories(mylib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

# ============================================================================
# Benchmark Executable
# ============================================================================

add_executable(misc_bench bench.cpp)
target_link_libraries(misc_bench PRIVATE
  benchmark::benchmark
  GTest::gtest_main
  GTest::gmock_main
  OpenMP::OpenMP_CXX
  mylib
)

# ============================================================================
# Test Executable
# ============================================================================

add_executable(misc_tests test.cpp)
target_link_libraries(misc_tests PRIVATE
  GTest::gtest_main
  GTest::gmock_main
  OpenMP::OpenMP_CXX
  mylib
)

# Enable testing
enable_testing()
add_test(NAME misc_tests COMMAND misc_tests)