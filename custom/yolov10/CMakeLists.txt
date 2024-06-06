cmake_minimum_required(VERSION 3.10)

project(yolov10)

add_definitions(-std=c++17)
add_definitions(-DAPI_EXPORTS)
option(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_TYPE Debug)

# TODO(Call for PR): make cmake compatible with Windows
set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)
enable_language(CUDA)

# include and link dirs of cuda and tensorrt, you need adapt them if yours are different
# cuda
include_directories(/usr/local/cuda/include)
link_directories(/usr/local/cuda/lib64)
# tensorrt
# TODO(Call for PR): make TRT path configurable from command line
include_directories(/lorenzo/env/install/TensorRT-8.6.1.6/include/)
link_directories(/lorenzo/env/install/TensorRT-8.6.1.6/lib/)

include_directories(${PROJECT_SOURCE_DIR}/src/)
include_directories(${PROJECT_SOURCE_DIR}/plugin/)

# OpenCV
# find_package(OpenCV REQUIRED)
# include_directories(${OpenCV_INCLUDE_DIRS})
set(OPENCV_INCLUDE /usr/local/include/opencv4 )
set(OPENCV_LIBPATH /usr/local/lib ) 
set(OpenCV_LIBS opencv_highgui opencv_core opencv_imgproc opencv_imgcodecs opencv_videoio opencv_video opencv_dnn)
include_directories(${OPENCV_INCLUDE})
link_directories(${OPENCV_LIBPATH})

add_executable(main main.cc ${SRCS})
target_link_libraries(main nvinfer)
target_link_libraries(main nvonnxparser)
target_link_libraries(main cudart)
target_link_libraries(main ${OpenCV_LIBS})

add_executable(yolov10_det yolov10_det.cc ${SRCS})
target_link_libraries(yolov10_det nvinfer)
target_link_libraries(yolov10_det nvonnxparser)
target_link_libraries(yolov10_det cudart)
target_link_libraries(yolov10_det ${OpenCV_LIBS})