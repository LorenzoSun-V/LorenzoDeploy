cmake_minimum_required(VERSION 3.0)

# source
file(GLOB TEST_SRCS ./*.cpp)

ADD_EXECUTABLE(test-infer ${TEST_SRCS})
TARGET_LINK_LIBRARIES(test-infer ${OPENCV_LIBS} ${RKINFER_LIBS} ${UTILS_LIBS})
