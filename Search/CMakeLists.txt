cmake_minimum_required(VERSION 3.18)
project(Search LANGUAGES CXX)

add_executable(Search bench.cpp)
target_link_libraries(Search PRIVATE benchmark::benchmark)