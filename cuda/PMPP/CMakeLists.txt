cmake_minimum_required(VERSION 3.5...3.20)
if(USE_CUDA)
    set(CMAKE_CUDA_ARCHITECTURES "native")
    project(PMPP LANGUAGES CXX CUDA)
    set(DEPENDENCIES benchmark::benchmark CUDA::cudart OpenMP::OpenMP_CXX GTest::gtest_main)

    # Add library with proper CUDA compilation
    add_library(PMPP vector_add.cu vector_add.h gemm.cu gemm.h convolution.cu convolution.h reduction.cu reduction.h merge.cu merge.cuh merge.cu radix_sort.cu radix_sort.cuh)
    
    target_link_libraries(PMPP PRIVATE ${DEPENDENCIES})
    target_compile_features(PMPP PRIVATE cxx_std_17)
    set_property(TARGET PMPP PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)
    
    # Include the GEMM folder for CPU implementations
    target_include_directories(PMPP PRIVATE ${CMAKE_SOURCE_DIR}/GEMM)

    # Add test executable
    add_executable(PMPP_test test.cpp merge_test.cpp radix_sort_test.cpp)
    target_link_libraries(PMPP_test PRIVATE PMPP ${DEPENDENCIES})
    target_compile_features(PMPP_test PRIVATE cxx_std_17)
    
    # Enable test discovery
    include(GoogleTest)
    gtest_discover_tests(PMPP_test)

    # Optional: Add benchmark executable if you want to run benchmarks
    add_executable(PMPP_bench bench.cpp)
    target_link_libraries(PMPP_bench PRIVATE PMPP ${DEPENDENCIES})
    target_compile_features(PMPP_bench PRIVATE cxx_std_17)
    target_include_directories(PMPP_bench PRIVATE ${CMAKE_SOURCE_DIR}/GEMM)

    # Merge-specific benchmark executable
    add_executable(PMPP_merge_bench merge_bench.cpp)
    target_link_libraries(PMPP_merge_bench PRIVATE PMPP ${DEPENDENCIES})
    target_compile_features(PMPP_merge_bench PRIVATE cxx_std_17)
    
    # Radix sort benchmark executable
    add_executable(PMPP_radix_sort_bench radix_sort_bench.cu)
    target_link_libraries(PMPP_radix_sort_bench PRIVATE PMPP ${DEPENDENCIES})
    target_compile_features(PMPP_radix_sort_bench PRIVATE cxx_std_17)
    
    # Add demo executable showing GEMM comparison usage
    add_executable(PMPP_gemm_demo gemm_demo.cpp)
    target_link_libraries(PMPP_gemm_demo PRIVATE PMPP ${DEPENDENCIES})
    target_compile_features(PMPP_gemm_demo PRIVATE cxx_std_17)
    target_include_directories(PMPP_gemm_demo PRIVATE ${CMAKE_SOURCE_DIR}/GEMM)

endif(USE_CUDA)
