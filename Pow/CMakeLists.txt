cmake_minimum_required(VERSION 3.18)
project(Pow LANGUAGES CXX)

add_executable(Pow bench.cpp)
target_link_libraries(Pow PRIVATE benchmark::benchmark)