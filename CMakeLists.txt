cmake_minimum_required(VERSION 3.28)
project(mnist_from_scratch)

set(CMAKE_C_STANDARD 23)

add_executable(mnist_from_scratch main.c
        matrix.c
        matrix.h
        engine.c
        engine.h
        matrix.c
        dataset.c
        dataset.h
        params.c
        params.h
        batch.c
        batch.h
        mem.c
        mem.h)
