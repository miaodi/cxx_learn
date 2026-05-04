cmake_minimum_required(VERSION 3.18)
project(CopyVsMemcpy LANGUAGES CXX)

if(CXX_LEARNING_HAS_BENCHMARK)
  add_executable(copy_vs_memcpy bench.cpp)
  target_link_libraries(copy_vs_memcpy PRIVATE benchmark::benchmark)
endif()
