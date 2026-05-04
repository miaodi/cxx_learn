cmake_minimum_required(VERSION 3.18)
project(Atomic LANGUAGES CXX)

if(CXX_LEARNING_HAS_BENCHMARK AND CXX_LEARNING_HAS_GTEST)
  add_executable(Atomic test.cpp memory_order_test.cpp benign_data_racing_test.cpp)
  target_link_libraries(Atomic PRIVATE
    benchmark::benchmark
    GTest::gtest_main
    GTest::gmock_main
    OpenMP::OpenMP_CXX
  )

  add_test(NAME Atomic COMMAND Atomic)
endif()
