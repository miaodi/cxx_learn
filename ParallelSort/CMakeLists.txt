cmake_minimum_required(VERSION 3.18)
project(ParallelSort LANGUAGES CXX)

add_executable(ParallelSort_bench bench.cpp)
target_link_libraries(ParallelSort_bench PRIVATE 
    benchmark::benchmark
    OpenMP::OpenMP_CXX
)
add_executable(ParallelSort_tests parallel_sort_tests.cpp)
target_link_libraries(ParallelSort_tests PRIVATE OpenMP::OpenMP_CXX)

find_package(GTest QUIET)
if(GTest_FOUND)
    target_compile_definitions(ParallelSort_tests PRIVATE USE_GTEST)
    target_link_libraries(ParallelSort_tests PRIVATE GTest::gtest GTest::gtest_main pthread)
endif()
