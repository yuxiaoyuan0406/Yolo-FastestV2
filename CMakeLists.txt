cmake_minimum_required(VERSION 3.2)

project(yolof)

set(LIBTORCH_PREFIX_PATH variable)
message(STATUS "Running command: ")
message(STATUS "    python -c \'import torch;print(torch.utils.cmake_prefix_path)\'")
execute_process(COMMAND python "-c" "\'import torch;print(torch.utils.cmake_prefix_path)\'"
    OUTPUT_VARIABLE LIBTORCH_PREFIX_PATH)
message(STATUS "Output: ")
message(STATUS "    ${LIBTORCH_PREFIX_PATH}")

set(CMAKE_PREFIX_PATH ${LIBTORCH_PREFIX_PATH})
set(CMAKE_INSTALL_PREFIX /usr/local)
set(CMAKE_BUILD_TYPE Release)
set(OPENCV_EXTRA_MODULES_PATH /home/vip/opencv_contrib/modules)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)

find_package(PythonInterp REQUIRED)
find_package(Torch REQUIRED)
find_package(OpenCV REQUIRED)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

message(STATUS "libtorch status:")
message(STATUS "    config: ${TORCH_INSTALL_PREFIX}")
message(STATUS "    libraries: ${TORCH_LIBRARIES}")
message(STATUS "    include path: ${TORCH_INCLUDE_DIRS}")

message(STATUS "OpenCV library status:")
message(STATUS "    config: ${OpenCV_DIR}")
message(STATUS "    version: ${OpenCV_VERSION}")
message(STATUS "    libraries: ${OpenCV_LIBS}")
message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")

add_executable(${PROJECT_NAME} src/main.cpp src/utils.cpp)
include_directories(./include)

target_link_libraries(${PROJECT_NAME} ${TORCH_LIBRARIES})
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})
set_property(TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD 14)
