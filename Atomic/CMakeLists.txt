cmake_minimum_required(VERSION 3.5...3.20)
project(Atomic CXX)

set(SRC test.cpp memory_order_test.cpp)

set(DEPENDENCIES benchmark::benchmark GTest::gtest_main GTest::gmock_main  OpenMP::OpenMP_CXX)

add_executable(${PROJECT_NAME} ${SRC})
# set_target_properties(${TARGET_NAME} PROPERTIES OUTPUT_NAME ${ONE_BENCH_EXEC}) 
target_link_libraries(${PROJECT_NAME} PRIVATE ${DEPENDENCIES})