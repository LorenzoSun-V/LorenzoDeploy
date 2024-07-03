cmake_minimum_required(VERSION 3.0)

# source
file(GLOB TEST_SRCS ./*.cpp)

ADD_EXECUTABLE(testbatch ${TEST_SRCS})
TARGET_LINK_LIBRARIES(testbatch ${OPENCV_LIBS} ${INFER_LIBS} ${VIDEODECODE_LIBS} ${UTILS_LIBS})
