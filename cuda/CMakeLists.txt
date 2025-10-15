cmake_minimum_required(VERSION 3.5...3.20)

if(USE_CUDA)
    set(CMAKE_CUDA_ARCHITECTURES "native")
    project(cuda LANGUAGES CXX CUDA)
    set(CUDA_SRC permutation_cuda.cu)
    set(CUDA_DEPENDENCIES benchmark::benchmark  CUDA::cudart)
    add_executable(permutation_cuda ${CUDA_SRC})
    # target_compile_options(permutation_cuda PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-std=c++17>)
    target_link_libraries(permutation_cuda PRIVATE ${CUDA_DEPENDENCIES} )

    set(CPU_SRC permutation_cpu.cpp)
    set(CPU_DEPENDENCIES benchmark::benchmark  OpenMP::OpenMP_CXX)
    add_executable(permutation_cpu ${CPU_SRC})
    target_link_libraries(permutation_cpu PRIVATE ${CPU_DEPENDENCIES} )

    

    set(TEST_SRC permutation_test.cu)

    set(TEST_DEPENDENCIES benchmark::benchmark GTest::gtest_main OpenMP::OpenMP_CXX  CUDA::cudart)

    add_executable(permutation_test ${TEST_SRC})
    # set_target_properties(${TARGET_NAME} PROPERTIES OUTPUT_NAME ${ONE_BENCH_EXEC}) 
    target_link_libraries(permutation_test PRIVATE ${TEST_DEPENDENCIES})

    add_executable(gpu_info gpu_info.cu)
    target_link_libraries(gpu_info PRIVATE CUDA::cudart)
    add_subdirectory(PMPP)
endif(USE_CUDA)