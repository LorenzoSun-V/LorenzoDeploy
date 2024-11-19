cmake_minimum_required(VERSION 3.0)

# source
file(GLOB TEST_SRCS ./*.cpp)

ADD_EXECUTABLE(test-rtsp ${TEST_SRCS})
TARGET_LINK_LIBRARIES(test-rtsp ${OPENCV_LIBS} ${RKINFER_LIBS}  ${VIDEOSTREAM_LIBS} ${UTILS_LIBS})
