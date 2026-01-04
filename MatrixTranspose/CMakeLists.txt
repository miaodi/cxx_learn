cmake_minimum_required(VERSION 3.18)

project(MatrixTranspose LANGUAGES CXX)

# Header-only library
add_library(matrix_transpose INTERFACE)
target_include_directories(matrix_transpose INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_features(matrix_transpose INTERFACE cxx_std_17)

# Benchmark executable
add_executable(transpose_bench bench.cpp)
target_link_libraries(transpose_bench PRIVATE
    matrix_transpose
    benchmark::benchmark
)
target_compile_features(transpose_bench PRIVATE cxx_std_17)

# Test executable
add_executable(transpose_test test.cpp)
target_link_libraries(transpose_test PRIVATE
    matrix_transpose
    GTest::gtest_main
)
target_compile_features(transpose_test PRIVATE cxx_std_17)

# Enable test discovery
include(GoogleTest)
gtest_discover_tests(transpose_test)
