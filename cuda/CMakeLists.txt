cmake_minimum_required(VERSION 3.18)

if(NOT USE_CUDA)
  return()
endif()

project(cuda LANGUAGES CXX CUDA)

# ============================================================================
# Find cuBLAS manually (NVIDIA HPC SDK has it in math_libs)
# ============================================================================

find_library(CUBLAS_LIBRARY
  NAMES cublas
  PATHS
    $ENV{NVHPC_MATH_LIBS}/lib64
    $ENV{CUDA_HOME}/lib64
    $ENV{CUDA_ROOT}/lib64
    ${CMAKE_CUDA_TOOLKIT_LIBRARY_ROOT}
  NO_DEFAULT_PATH
)

if(CUBLAS_LIBRARY)
  message(STATUS "Found cuBLAS: ${CUBLAS_LIBRARY}")
  add_library(cublas_compat INTERFACE)
  target_link_libraries(cublas_compat INTERFACE ${CUBLAS_LIBRARY})
  # Create an alias so we can use CUDA::cublas or cublas_compat interchangeably
  if(NOT TARGET CUDA::cublas)
    add_library(CUDA::cublas ALIAS cublas_compat)
  endif()
else()
  message(WARNING "cuBLAS library not found. Some targets may fail to build.")
endif()

# ============================================================================
# Find NVTX3 - DISABLED (optional profiling feature)
# ============================================================================

# # Try to find NVTX3 headers (NVIDIA HPC SDK has versioned subdirectories)
# # Search in multiple possible locations including versioned paths
# find_path(NVTX3_INCLUDE_DIR
#   NAMES nvtx3/nvtx3.hpp cub/detail/nvtx3.hpp
#   PATHS
#     $ENV{CUDA_HOME}/targets/x86_64-linux/include
#     $ENV{CUDA_ROOT}/targets/x86_64-linux/include
#     ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
#   NO_DEFAULT_PATH
# )

# # If not found, search in versioned subdirectories
# if(NOT NVTX3_INCLUDE_DIR)
#   file(GLOB CUDA_VERSION_DIRS 
#     "$ENV{CUDA_HOME}/[0-9]*/targets/x86_64-linux/include"
#     "$ENV{CUDA_ROOT}/[0-9]*/targets/x86_64-linux/include"
#   )
#   find_path(NVTX3_INCLUDE_DIR
#     NAMES nvtx3/nvtx3.hpp cub/detail/nvtx3.hpp
#     PATHS ${CUDA_VERSION_DIRS}
#     NO_DEFAULT_PATH
#   )
# endif()

# if(NVTX3_INCLUDE_DIR)
#   message(STATUS "Found NVTX3 headers: ${NVTX3_INCLUDE_DIR}")
#   add_library(nvtx3 INTERFACE)
#   target_include_directories(nvtx3 INTERFACE ${NVTX3_INCLUDE_DIR})
#   # Link NVTX library (libnvToolsExt.so)
#   find_library(NVTX3_LIBRARY
#     NAMES nvToolsExt
#     PATHS
#       $ENV{CUDA_HOME}/lib64
#       $ENV{CUDA_ROOT}/lib64
#       ${CMAKE_CUDA_TOOLKIT_LIBRARY_ROOT}
#     NO_DEFAULT_PATH
#   )
#   if(NVTX3_LIBRARY)
#     target_link_libraries(nvtx3 INTERFACE ${NVTX3_LIBRARY})
#   endif()
# else()
#   message(WARNING "NVTX3 headers not found. Some profiling features may be disabled.")
# endif()

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
if(TARGET nvtx3)
  target_link_libraries(memory_stride_bench PRIVATE nvtx3)
  target_compile_definitions(memory_stride_bench PRIVATE ENABLE_NVTX)
endif()
set_target_properties(memory_stride_bench PROPERTIES
  CXX_STANDARD 17
  CXX_STANDARD_REQUIRED ON
)

# ============================================================================
# Bank conflict benchmark
# ============================================================================

add_executable(bank_conflict_bench bank_conflict_bench.cu)
target_link_libraries(bank_conflict_bench PRIVATE
  CUDA::cudart
  benchmark::benchmark
)
set_target_properties(bank_conflict_bench PROPERTIES
  CXX_STANDARD 17
  CXX_STANDARD_REQUIRED ON
)

# ============================================================================
# Sort + Unique benchmark (CUB vs Thrust)
# ============================================================================

add_executable(sort_and_unique sort_and_unique.cu)
target_link_libraries(sort_and_unique PRIVATE
  CUDA::cudart
  benchmark::benchmark
)
set_target_properties(sort_and_unique PROPERTIES
  CXX_STANDARD 17
  CXX_STANDARD_REQUIRED ON
)

# ============================================================================
# Graph Transpose Benchmark and Test
# ============================================================================

add_executable(graph_transpose_bench graph_transpose_bench.cu graph_transpose.cu)
target_include_directories(graph_transpose_bench PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(graph_transpose_bench PRIVATE
  CUDA::cudart
  benchmark::benchmark
)
set_target_properties(graph_transpose_bench PROPERTIES
  CXX_STANDARD 17
  CXX_STANDARD_REQUIRED ON
)

add_executable(graph_transpose_test graph_transpose_test.cu graph_transpose.cu)
target_include_directories(graph_transpose_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(graph_transpose_test PRIVATE
  CUDA::cudart
  GTest::gtest_main
)
set_target_properties(graph_transpose_test PROPERTIES
  CXX_STANDARD 17
  CXX_STANDARD_REQUIRED ON
)
add_test(NAME graph_transpose_test COMMAND graph_transpose_test)

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

# ============================================================================
# Scratch - Thrust Practice
# ============================================================================

add_subdirectory(scratch)
