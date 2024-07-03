cmake_minimum_required(VERSION 3.0)

# source
file(GLOB TEST_SRCS ./*.cpp)

ADD_EXECUTABLE(test-rtspclient ${TEST_SRCS})
TARGET_LINK_LIBRARIES(test-rtspclient ${OPENCV_LIBS} ${INFER_LIBS} ${VIDEODECODE_LIBS} )
