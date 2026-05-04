cmake_minimum_required(VERSION 3.18)
project(AssemblyStudy LANGUAGES CXX)

if(CXX_LEARNING_HAS_BENCHMARK)
  add_executable(swap swap.cpp)
  target_link_libraries(swap PRIVATE benchmark::benchmark)
endif()
