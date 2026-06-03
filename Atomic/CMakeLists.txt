cmake_minimum_required(VERSION 3.18)
project(Atomic LANGUAGES CXX)

find_package(Threads REQUIRED)

add_executable(memory_order_demo memory_order_demo.cpp)
target_link_libraries(memory_order_demo PRIVATE Threads::Threads)

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
