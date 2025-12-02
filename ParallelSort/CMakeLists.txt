cmake_minimum_required(VERSION 3.18)
project(ParallelSort LANGUAGES CXX)

add_executable(ParallelSort_bench bench.cpp)
target_link_libraries(ParallelSort_bench PRIVATE 
    benchmark::benchmark
    OpenMP::OpenMP_CXX
)
