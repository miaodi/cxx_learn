cmake_minimum_required(VERSION 3.5...3.20)
project(CrossProductVecLen CXX)

set(ElementWiseProductBench_SRC ElementWiseProductBench.cpp)

set(DEPENDENCIES benchmark::benchmark)

add_executable(ElementWiseProductBench ${ElementWiseProductBench_SRC})
# set_target_properties(${TARGET_NAME} PROPERTIES OUTPUT_NAME ${ONE_BENCH_EXEC}) 
target_link_libraries(ElementWiseProductBench PRIVATE ${DEPENDENCIES})

set(CrossProductBench_SRC CrossProductBench.cpp)

set(DEPENDENCIES benchmark::benchmark)

add_executable(CrossProductBench ${CrossProductBench_SRC})
# set_target_properties(${TARGET_NAME} PROPERTIES OUTPUT_NAME ${ONE_BENCH_EXEC}) 
target_link_libraries(CrossProductBench PRIVATE ${DEPENDENCIES})

set(fmaBench_SRC fmaBench.cpp)
add_executable(fmaBench ${fmaBench_SRC})
target_link_libraries(fmaBench PRIVATE ${DEPENDENCIES})