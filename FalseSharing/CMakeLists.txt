cmake_minimum_required(VERSION 3.18)
project(FalseSharing LANGUAGES CXX)

add_executable(FalseSharing bench.cpp)
target_link_libraries(FalseSharing PRIVATE
  benchmark::benchmark
  OpenMP::OpenMP_CXX
)