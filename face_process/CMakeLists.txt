cmake_minimum_required(VERSION 3.10)

project(face_app)

set(CMAKE_CXX_STANDARD 17)

find_package(OpenCV REQUIRED)

# only for test
#message(STATUS "------------------------------------------------")
#message(STATUS "OpenCV header detected: ${OpenCV_INCLUDE_DIRS}")
#message(STATUS "OpenCV Libs detected: ${OpenCV_LIBS}")
#message(STATUS "------------------------------------------------")


set(ncnn_DIR "/root/ncnn2/ncnn/build/install/lib/cmake/ncnn")
find_package(ncnn REQUIRED)

add_executable(face_app
    src/main.cpp
)

target_link_libraries(face_app
    PRIVATE
    ncnn
    ${OpenCV_LIBS}
)

# 单独编译 time.cpp 用于计时测试
add_executable(face_timer src/time.cpp)   # 如果放在 src/ 下则写 src/time.cpp
target_link_libraries(face_timer
    PRIVATE
    ncnn
    ${OpenCV_LIBS}
)
# 单独编译 精简版本（只有识别，没有注册）
add_executable(face_mini src/only.cpp)   # 如果放在 src/ 下则写 src/time.cpp
target_link_libraries(face_mini
    PRIVATE
    ncnn
    ${OpenCV_LIBS}
)
# 单独编译 debug版本（只有识别，没有注册）
add_executable(face_mini src/debug.cpp)   # 如果放在 src/ 下则写 src/time.cpp
target_link_libraries(face_debug
    PRIVATE
    ncnn
    ${OpenCV_LIBS}
)
