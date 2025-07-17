cmake_minimum_required(VERSION 3.5...3.20)
project(CrossProductVecLen CXX)

set(ElementWiseProductBench_SRC ElementWiseProductBench.cpp)

add_library(VectorOps STATIC
  VectorOps.cpp
)

set(DEPENDENCIES benchmark::benchmark GTest::gtest_main GTest::gmock_main  OpenMP::OpenMP_CXX VectorOps)

add_executable(ElementWiseProductBench ${ElementWiseProductBench_SRC})
# set_target_properties(${TARGET_NAME} PROPERTIES OUTPUT_NAME ${ONE_BENCH_EXEC}) 
target_link_libraries(ElementWiseProductBench PRIVATE ${DEPENDENCIES})

set(CrossProductLenBench_SRC CrossProductLenBench.cpp)
add_executable(CrossProductLenBench ${CrossProductLenBench_SRC})
target_link_libraries(CrossProductLenBench PRIVATE ${DEPENDENCIES})


set(VectorOpsBench_SRC VectorOpsBench.cpp)
add_executable(VectorOpsBench ${VectorOpsBench_SRC})
target_link_libraries(VectorOpsBench PRIVATE ${DEPENDENCIES})

set(fmaBench_SRC fmaBench.cpp)
add_executable(fmaBench ${fmaBench_SRC})
target_link_libraries(fmaBench PRIVATE ${DEPENDENCIES})


set(TEST_SRC test.cpp)
add_executable(tests ${TEST_SRC})
target_link_libraries(tests PRIVATE ${DEPENDENCIES})