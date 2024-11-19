cmake_minimum_required(VERSION 3.0)

# source
file(GLOB TEST_SRCS ./*.cpp)

ADD_EXECUTABLE(test-video ${TEST_SRCS})
TARGET_LINK_LIBRARIES(test-video ${OPENCV_LIBS} ${RKINFER_LIBS} ${UTILS_LIBS})
