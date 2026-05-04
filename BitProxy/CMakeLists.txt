if(CXX_LEARNING_HAS_BENCHMARK)
  add_executable(BitProxy_bench bench.cpp)
  target_link_libraries(BitProxy_bench PRIVATE benchmark::benchmark)
endif()
