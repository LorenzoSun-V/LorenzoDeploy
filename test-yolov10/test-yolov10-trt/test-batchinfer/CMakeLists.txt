cmake_minimum_required(VERSION 3.0)

# source
file(GLOB TEST_SRCS ./*.cpp)

ADD_EXECUTABLE(test-batchinfer ${TEST_SRCS})
TARGET_LINK_LIBRARIES(test-batchinfer ${OPENCV_LIBS} ${INFER_LIBS} ${UTILS_LIBS})
