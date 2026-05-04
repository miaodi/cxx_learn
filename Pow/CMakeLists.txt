cmake_minimum_required(VERSION 3.18)
project(Pow LANGUAGES CXX)

if(CXX_LEARNING_HAS_BENCHMARK)
  add_executable(Pow bench.cpp)
  target_link_libraries(Pow PRIVATE benchmark::benchmark)
endif()
