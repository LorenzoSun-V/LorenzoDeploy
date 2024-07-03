cmake_minimum_required(VERSION 3.0)

project(onnx2engine)

# source
file(GLOB TEST_SRCS ./*.cpp)

ADD_EXECUTABLE(onnx2engine ${TEST_SRCS})
TARGET_LINK_LIBRARIES(onnx2engine ${INFER_LIBS} ${CUDA_LIBPATH})
