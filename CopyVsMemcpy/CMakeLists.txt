cmake_minimum_required(VERSION 3.18)
project(CopyVsMemcpy LANGUAGES CXX)

add_executable(copy_vs_memcpy bench.cpp)
target_link_libraries(copy_vs_memcpy PRIVATE benchmark::benchmark)
