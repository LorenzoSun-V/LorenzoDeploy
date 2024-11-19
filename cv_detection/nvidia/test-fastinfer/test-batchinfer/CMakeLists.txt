cmake_minimum_required(VERSION 3.0)

# source
file(GLOB TEST_SRCS ./*.cpp)

ADD_EXECUTABLE(test-batch ${TEST_SRCS})
TARGET_LINK_LIBRARIES(test-batch ${OPENCV_LIBS} ${FASTINFER_LIBS} ${FASTDEPLOY_LIBS})
