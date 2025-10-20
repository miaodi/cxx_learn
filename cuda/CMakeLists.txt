cmake_minimum_required(VERSION 3.5...3.20)

if(USE_CUDA)
    set(CMAKE_CUDA_ARCHITECTURES "native")
    project(cuda LANGUAGES CXX CUDA)
    set(CUDA_SRC permutation_cuda.cu)
    set(CUDA_DEPENDENCIES benchmark::benchmark  CUDA::cudart CUDA::cublas )
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

    # Templated Gram-Schmidt Orthogonalization targets
    set(GRAM_SCHMIDT_SRC gram_schmidt.cu)
    set(GRAM_SCHMIDT_DEPENDENCIES CUDA::cudart CUDA::cublas)

    # Gram-Schmidt Demo (templated for float and double)
    add_executable(gram_schmidt_demo gram_schmidt_demo.cpp ${GRAM_SCHMIDT_SRC})
    target_link_libraries(gram_schmidt_demo PRIVATE ${GRAM_SCHMIDT_DEPENDENCIES})
    set_target_properties(gram_schmidt_demo PROPERTIES 
        CXX_STANDARD 17
        CXX_STANDARD_REQUIRED ON
    )

    # Gram-Schmidt Tests (templated for float and double)
    if(TARGET GTest::gtest_main)
        add_executable(gram_schmidt_test gram_schmidt_test.cpp ${GRAM_SCHMIDT_SRC})
        target_link_libraries(gram_schmidt_test PRIVATE ${GRAM_SCHMIDT_DEPENDENCIES} GTest::gtest_main)
        set_target_properties(gram_schmidt_test PROPERTIES 
            CXX_STANDARD 17
            CXX_STANDARD_REQUIRED ON
        )
    endif()

    # Gram-Schmidt Benchmarks (templated for float and double)
    if(TARGET benchmark::benchmark)
        add_executable(gram_schmidt_bench gram_schmidt_bench.cpp ${GRAM_SCHMIDT_SRC})
        target_link_libraries(gram_schmidt_bench PRIVATE ${GRAM_SCHMIDT_DEPENDENCIES} benchmark::benchmark)
        set_target_properties(gram_schmidt_bench PROPERTIES 
            CXX_STANDARD 17
            CXX_STANDARD_REQUIRED ON
        )
    endif()

    # Precision comparison test
    add_executable(precision_comparison_test precision_comparison_test.cpp ${GRAM_SCHMIDT_SRC})
    target_link_libraries(precision_comparison_test PRIVATE ${GRAM_SCHMIDT_DEPENDENCIES})
    set_target_properties(precision_comparison_test PROPERTIES 
        CXX_STANDARD 17
        CXX_STANDARD_REQUIRED ON
    )

    add_subdirectory(PMPP)
endif(USE_CUDA)