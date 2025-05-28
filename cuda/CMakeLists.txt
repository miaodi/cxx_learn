cmake_minimum_required(VERSION 3.5...3.20)
project(cuda CUDA CXX)

set(SRC permutation_bench.cu)

set(DEPENDENCIES benchmark::benchmark OpenMP::OpenMP_CXX ${CUDA_LIBRARIES})

add_executable(permutation ${SRC})
# set_target_properties(${TARGET_NAME} PROPERTIES OUTPUT_NAME ${ONE_BENCH_EXEC}) 
target_compile_options(permutation PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-std=c++17>)

target_link_libraries(permutation PRIVATE ${DEPENDENCIES} )