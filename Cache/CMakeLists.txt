cmake_minimum_required(VERSION 3.18)
project(Cache LANGUAGES CXX)

add_executable(cache_bench bench.cpp)
target_link_libraries(cache_bench PRIVATE benchmark::benchmark)