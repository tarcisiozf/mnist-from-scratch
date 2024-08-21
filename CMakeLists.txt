cmake_minimum_required(VERSION 3.28)
project(mnist_from_scratch CUDA)

set(CMAKE_CUDA_STANDARD 17)

add_executable(mnist_from_scratch main.cpp
        matrix.cpp
        matrix.h
        engine.cpp
        engine.h
        cuda.cu)
