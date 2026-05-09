add_library(PMPP_gemm sgemm.cu sgemm.h)
target_link_libraries(PMPP_gemm PUBLIC CUDA::cudart)
target_include_directories(PMPP_gemm PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_features(PMPP_gemm PUBLIC cxx_std_17)
set_property(TARGET PMPP_gemm PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)

if(CXX_LEARNING_HAS_GTEST)
  add_executable(PMPP_gemm_test sgemm_test.cu)
  target_link_libraries(PMPP_gemm_test PRIVATE
    PMPP_gemm
    CUDA::cudart
    GTest::gtest_main
  )
  target_compile_features(PMPP_gemm_test PRIVATE cxx_std_17)

  include(GoogleTest)
  gtest_discover_tests(PMPP_gemm_test)
endif()

if(CXX_LEARNING_HAS_BENCHMARK AND TARGET CUDA::cublas)
  add_executable(PMPP_gemm_bench sgemm_bench.cu)
  target_link_libraries(PMPP_gemm_bench PRIVATE
    PMPP_gemm
    CUDA::cublas
    CUDA::cudart
    benchmark::benchmark
  )
  target_compile_features(PMPP_gemm_bench PRIVATE cxx_std_17)
endif()

option(CXX_LEARNING_FETCH_CUTLASS
  "Download CUTLASS with FetchContent when CUTLASS_ROOT is not set"
  OFF
)
set(CXX_LEARNING_CUTLASS_GIT_TAG "v4.3.4" CACHE STRING
  "CUTLASS git tag used when CXX_LEARNING_FETCH_CUTLASS is enabled"
)

find_path(CUTLASS_INCLUDE_DIR
  NAMES cutlass/cutlass.h
  HINTS
    $ENV{CUTLASS_ROOT}/include
    $ENV{CUTLASS_DIR}/include
    $ENV{CUTLASS_ROOT}
    $ENV{CUTLASS_DIR}
)

if(NOT CUTLASS_INCLUDE_DIR AND CXX_LEARNING_FETCH_CUTLASS)
  include(FetchContent)
  FetchContent_Declare(
    cutlass
    GIT_REPOSITORY https://github.com/NVIDIA/cutlass.git
    GIT_TAG ${CXX_LEARNING_CUTLASS_GIT_TAG}
    GIT_SHALLOW TRUE
  )
  FetchContent_Populate(cutlass)
  set(CUTLASS_INCLUDE_DIR "${cutlass_SOURCE_DIR}/include" CACHE PATH
    "Path to CUTLASS headers"
    FORCE
  )
endif()

if(CUTLASS_INCLUDE_DIR)
  add_executable(PMPP_cutlass_sgemm_example cutlass_sgemm_example.cu)
  target_include_directories(PMPP_cutlass_sgemm_example SYSTEM PRIVATE
    ${CUTLASS_INCLUDE_DIR}
  )
  target_link_libraries(PMPP_cutlass_sgemm_example PRIVATE CUDA::cudart)
  target_compile_features(PMPP_cutlass_sgemm_example PRIVATE cxx_std_17)
else()
  message(STATUS
    "CUTLASS not found; PMPP_cutlass_sgemm_example will be skipped. "
    "Set CUTLASS_ROOT to a CUTLASS checkout or configure with "
    "-DCXX_LEARNING_FETCH_CUTLASS=ON to enable it.")
endif()
