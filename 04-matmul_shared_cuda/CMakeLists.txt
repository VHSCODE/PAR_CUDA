cmake_minimum_required( VERSION 3.20)

project(matmul_cuda LANGUAGES CXX CUDA)

set(SOURCES main.cpp src/kernel.cu include/kernel.cuh include/MatrixUtils.h include/toggles.h)

add_executable(matmul_cuda ${SOURCES})

target_include_directories(matmul_cuda PRIVATE include)