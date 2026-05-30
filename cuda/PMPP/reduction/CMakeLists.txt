add_library(PMPP_reduction reduction_sum.cu reduction_sum.h)
target_link_libraries(PMPP_reduction PUBLIC CUDA::cudart)
target_include_directories(PMPP_reduction PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_features(PMPP_reduction PUBLIC cxx_std_17)
set_property(TARGET PMPP_reduction PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)

if(CXX_LEARNING_HAS_GTEST)
  add_executable(PMPP_reduction_test reduction_sum_test.cpp)
  target_link_libraries(PMPP_reduction_test PRIVATE
    PMPP_reduction
    CUDA::cudart
    GTest::gtest_main
  )
  target_compile_features(PMPP_reduction_test PRIVATE cxx_std_17)

  include(GoogleTest)
  gtest_discover_tests(PMPP_reduction_test)
endif()

if(CXX_LEARNING_HAS_BENCHMARK)
  add_executable(PMPP_reduction_bench reduction_sum_bench.cpp)
  target_link_libraries(PMPP_reduction_bench PRIVATE
    PMPP_reduction
    CUDA::cudart
    benchmark::benchmark
  )
  target_compile_features(PMPP_reduction_bench PRIVATE cxx_std_17)
endif()
