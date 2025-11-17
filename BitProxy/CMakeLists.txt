add_executable(BitProxy_bench bench.cpp)
target_link_libraries(BitProxy_bench PRIVATE benchmark::benchmark)
