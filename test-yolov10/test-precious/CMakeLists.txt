cmake_minimum_required(VERSION 3.0)

# source
file(GLOB TEST_SRCS ./*.cpp)

ADD_EXECUTABLE(testprecious ${TEST_SRCS})
if(WITH_VIDEO)
    TARGET_LINK_LIBRARIES(testprecious ${OPENCV_LIBS} ${INFER_LIBS} ${VIDEODECODE_LIBS} ${UTILS_LIBS})
else()
    TARGET_LINK_LIBRARIES(testprecious ${OPENCV_LIBS} ${INFER_LIBS} ${UTILS_LIBS})
endif()
