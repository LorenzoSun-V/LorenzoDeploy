cmake_minimum_required(VERSION 3.0)

# source
file(GLOB TEST_SRCS ./*.cpp)

ADD_EXECUTABLE(testbatchinfer ${TEST_SRCS})
TARGET_LINK_LIBRARIES(testbatchinfer ${OPENCV_LIBS} ${INFER_LIBS} ${UTILS_LIBS})
