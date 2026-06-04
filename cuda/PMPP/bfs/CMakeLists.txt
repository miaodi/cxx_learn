add_library(PMPP_bfs bfs.cu bfs.h)
target_link_libraries(PMPP_bfs PUBLIC CUDA::cudart)
target_include_directories(PMPP_bfs PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_SOURCE_DIR}/cuda
)
target_compile_features(PMPP_bfs PUBLIC cxx_std_17)
set_property(TARGET PMPP_bfs PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)

if(CXX_LEARNING_HAS_BENCHMARK)
  add_executable(PMPP_bfs_bench bfs_bench.cu)
  target_link_libraries(PMPP_bfs_bench PRIVATE
    PMPP_bfs
    CUDA::cudart
    benchmark::benchmark
  )
  target_include_directories(PMPP_bfs_bench PRIVATE ${CMAKE_SOURCE_DIR}/cuda)
  target_compile_definitions(PMPP_bfs_bench PRIVATE
    PMPP_BFS_SAMPLE_DIR="${CMAKE_CURRENT_SOURCE_DIR}/samples"
  )
  target_compile_features(PMPP_bfs_bench PRIVATE cxx_std_17)
endif()
