cmake_minimum_required(VERSION 3.18)
project(AssemblyStudy LANGUAGES CXX)

add_executable(swap swap.cpp)
target_link_libraries(swap PRIVATE benchmark::benchmark)