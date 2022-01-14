cmake_minimum_required( VERSION 3.20)

project(traspuesta_cuda LANGUAGES CXX CUDA)

set(SOURCES main.cpp src/kernel.cu include/kernel.cuh include/MatrixUtils.h include/toggles.h)

add_executable(traspuesta_cuda ${SOURCES})

target_include_directories(traspuesta_cuda PRIVATE include)