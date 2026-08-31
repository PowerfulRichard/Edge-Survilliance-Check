cmake_minimum_required(VERSION 3.16)
project(ncnn_face_official LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs calib3d)
set(ncnn_DIR "/root/ncnn2/ncnn/build/install/lib/cmake/ncnn")

find_package(ncnn REQUIRED)

add_library(face_core STATIC
    src/scrfd.cpp
    src/mobilefacenet.cpp
    src/face_database.cpp
    src/face_pipeline.cpp
    src/cli.cpp
    src/json_output.cpp
)
target_include_directories(face_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(face_core PUBLIC ncnn ${OpenCV_LIBS})

add_executable(face_main apps/main.cpp)
target_link_libraries(face_main PRIVATE face_core)

add_executable(face_mini apps/mini.cpp)
target_link_libraries(face_mini PRIVATE face_core)

add_executable(face_debug apps/debug.cpp)
target_link_libraries(face_debug PRIVATE face_core)

install(TARGETS face_main face_mini face_debug RUNTIME DESTINATION bin)
