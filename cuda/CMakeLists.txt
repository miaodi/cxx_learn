cmake_minimum_required(VERSION 3.5...3.20)
project(cuda CUDA CXX)

set(CUDA_SRC permutation_cuda.cu)
set(CUDA_DEPENDENCIES benchmark::benchmark ${CUDA_LIBRARIES})
add_executable(permutation_cuda ${CUDA_SRC})
target_compile_options(permutation_cuda PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-std=c++17>)
target_link_libraries(permutation_cuda PRIVATE ${CUDA_DEPENDENCIES} )

set(CPU_SRC permutation_cpu.cpp)
set(CPU_DEPENDENCIES benchmark::benchmark  OpenMP::OpenMP_CXX)
add_executable(permutation_cpu ${CPU_SRC})
target_link_libraries(permutation_cpu PRIVATE ${CPU_DEPENDENCIES} )