cmake_minimum_required(VERSION 3.18)
project(Search LANGUAGES CXX)

if(CXX_LEARNING_HAS_BENCHMARK)
  add_executable(Search bench.cpp)
  target_link_libraries(Search PRIVATE benchmark::benchmark)
endif()
