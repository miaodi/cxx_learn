cmake_minimum_required(VERSION 3.18)
project(RegisterLatency LANGUAGES CXX)

if(CXX_LEARNING_HAS_BENCHMARK)
  add_executable(register_latency_bench bench.cpp)
  target_link_libraries(register_latency_bench PRIVATE benchmark::benchmark)

  # Keep this benchmark focused on scalar dependency chains rather than SIMD gathers.
  if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    target_compile_options(register_latency_bench PRIVATE -fno-tree-vectorize)
  elseif(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    target_compile_options(register_latency_bench PRIVATE -fno-vectorize -fno-slp-vectorize)
  endif()
endif()
