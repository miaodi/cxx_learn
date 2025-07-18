cmake_minimum_required(VERSION 3.5...3.20)
project(misc CXX)

add_library(mylib STATIC
  func.cpp
)

set(SRC bench.cpp)

set(DEPENDENCIES benchmark::benchmark GTest::gtest_main GTest::gmock_main  OpenMP::OpenMP_CXX mylib)

add_executable(misc_bench ${SRC})
# set_target_properties(${TARGET_NAME} PROPERTIES OUTPUT_NAME ${ONE_BENCH_EXEC}) 
target_link_libraries(misc_bench PRIVATE ${DEPENDENCIES})

set(TEST_SRC test.cpp)
add_executable(misc_tests ${TEST_SRC})
target_link_libraries(misc_tests PRIVATE ${DEPENDENCIES})