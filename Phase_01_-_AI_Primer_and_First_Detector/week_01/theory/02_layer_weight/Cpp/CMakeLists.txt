cmake_minimum_required(VERSION 3.15)
project(week_01_layer_weight CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_executable(layer_02 layer_02.cpp)
target_compile_definitions(layer_02 PRIVATE
    MODEL_FILE="${CMAKE_CURRENT_SOURCE_DIR}/../line_model.txt")
