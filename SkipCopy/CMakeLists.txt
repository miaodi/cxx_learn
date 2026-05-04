cmake_minimum_required(VERSION 3.18)
project(SkipCopy LANGUAGES CXX)

if(CXX_LEARNING_HAS_BENCHMARK)
  add_executable(SkipCopy bench.cpp)
  target_link_libraries(SkipCopy PRIVATE benchmark::benchmark)
endif()
