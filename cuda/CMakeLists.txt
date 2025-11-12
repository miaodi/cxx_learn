cmake_minimum_required(VERSION 3.18)

if(NOT USE_CUDA)
  return()
endif()

project(cuda LANGUAGES CXX CUDA)

# ============================================================================
# Permutation Benchmarks
# ============================================================================

# CUDA version
add_executable(permutation_cuda permutation_cuda.cu)
target_include_directories(permutation_cuda PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(permutation_cuda PRIVATE
  benchmark::benchmark
  CUDA::cudart
  CUDA::cublas
)

# CPU version
add_executable(permutation_cpu permutation_cpu.cpp)
target_include_directories(permutation_cpu PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(permutation_cpu PRIVATE
  benchmark::benchmark
  OpenMP::OpenMP_CXX
)

# ============================================================================
# Permutation Tests
# ============================================================================

add_executable(permutation_test permutation_test.cu)
target_include_directories(permutation_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(permutation_test PRIVATE
  GTest::gtest_main
  OpenMP::OpenMP_CXX
  CUDA::cudart
)

enable_testing()
add_test(NAME permutation_test COMMAND permutation_test)

# ============================================================================
# GPU Info Utility
# ============================================================================

add_executable(gpu_info gpu_info.cu)
target_link_libraries(gpu_info PRIVATE CUDA::cudart)

# ============================================================================
# Memory coalescing benchmark
# ============================================================================

add_executable(memory_stride_bench memory_stride_bench.cu)
target_link_libraries(memory_stride_bench PRIVATE
  CUDA::cudart
  benchmark::benchmark
)
set_target_properties(memory_stride_bench PROPERTIES
  CXX_STANDARD 17
  CXX_STANDARD_REQUIRED ON
)

# ============================================================================
# Gram-Schmidt Orthogonalization
# ============================================================================

set(GRAM_SCHMIDT_SRC gram_schmidt.cu)

# Demo executable
add_executable(gram_schmidt_demo gram_schmidt_demo.cpp ${GRAM_SCHMIDT_SRC})
target_link_libraries(gram_schmidt_demo PRIVATE
  CUDA::cudart
  CUDA::cublas
)
set_target_properties(gram_schmidt_demo PROPERTIES
  CXX_STANDARD 17
  CXX_STANDARD_REQUIRED ON
)

# Test executable
add_executable(gram_schmidt_test gram_schmidt_test.cpp ${GRAM_SCHMIDT_SRC})
target_link_libraries(gram_schmidt_test PRIVATE
  CUDA::cudart
  CUDA::cublas
  GTest::gtest_main
)
set_target_properties(gram_schmidt_test PROPERTIES
  CXX_STANDARD 17
  CXX_STANDARD_REQUIRED ON
)
add_test(NAME gram_schmidt_test COMMAND gram_schmidt_test)

# Benchmark executable
add_executable(gram_schmidt_bench gram_schmidt_bench.cpp ${GRAM_SCHMIDT_SRC})
target_link_libraries(gram_schmidt_bench PRIVATE
  CUDA::cudart
  CUDA::cublas
  benchmark::benchmark
)
set_target_properties(gram_schmidt_bench PROPERTIES
  CXX_STANDARD 17
  CXX_STANDARD_REQUIRED ON
)

# Precision comparison test
add_executable(precision_comparison_test precision_comparison_test.cpp ${GRAM_SCHMIDT_SRC})
target_link_libraries(precision_comparison_test PRIVATE
  CUDA::cudart
  CUDA::cublas
)
set_target_properties(precision_comparison_test PROPERTIES
  CXX_STANDARD 17
  CXX_STANDARD_REQUIRED ON
)

# ============================================================================
# PMPP Subdirectory
# ============================================================================

add_subdirectory(PMPP)
