cmake_minimum_required(VERSION 3.5...3.20)
project(GEMM CXX)
set(SRC bench.cpp)

set(DEPENDENCIES benchmark::benchmark GTest::gtest_main )

add_executable(GEMM_bench ${SRC})
# set_target_properties(${TARGET_NAME} PROPERTIES OUTPUT_NAME ${ONE_BENCH_EXEC}) 
target_link_libraries(GEMM_bench PRIVATE ${DEPENDENCIES})

set(TEST_SRC test.cpp)
add_executable(GEMM_test ${TEST_SRC})
target_link_libraries(GEMM_test PRIVATE ${DEPENDENCIES})