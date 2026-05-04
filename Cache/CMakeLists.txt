cmake_minimum_required(VERSION 3.18)
project(Cache LANGUAGES CXX)

if(CXX_LEARNING_HAS_BENCHMARK)
  add_executable(cache_bench bench.cpp)
  target_link_libraries(cache_bench PRIVATE benchmark::benchmark)
endif()
