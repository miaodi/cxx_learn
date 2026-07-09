add_library(PMPP_scan scan.cu scan.h)
target_link_libraries(PMPP_scan PUBLIC CUDA::cudart)
target_include_directories(PMPP_scan PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_features(PMPP_scan PUBLIC cxx_std_17)
set_property(TARGET PMPP_scan PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)

if(CXX_LEARNING_HAS_GTEST)
  add_executable(PMPP_scan_test scan_test.cpp)
  target_link_libraries(PMPP_scan_test PRIVATE
    PMPP_scan
    CUDA::cudart
    GTest::gtest_main
  )
  target_compile_features(PMPP_scan_test PRIVATE cxx_std_17)

  include(GoogleTest)
  gtest_discover_tests(PMPP_scan_test)
endif()
