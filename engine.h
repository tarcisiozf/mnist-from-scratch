#ifndef MNIST_FROM_SCRATCH_ENGINE_H
#define MNIST_FROM_SCRATCH_ENGINE_H

#ifdef CUDA
#include "cuda.cuh"
#else
#include "matrix.h"
#endif

#include "params.h"

Parameters* gradient_descent(const Matrix* X, const float* Y, int Y_len, float lr, int epochs);

void eval(const Matrix* X, const float* Y, int N, const Parameters* input);

#endif //MNIST_FROM_SCRATCH_ENGINE_H
