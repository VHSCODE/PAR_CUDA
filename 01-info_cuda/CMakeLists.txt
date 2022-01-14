cmake_minimum_required( VERSION 3.20)

project(info_cuda LANGUAGES CXX CUDA)

set(SOURCES main.cu)

add_executable(info_cuda ${SOURCES})

target_include_directories(info_cuda PRIVATE include)
