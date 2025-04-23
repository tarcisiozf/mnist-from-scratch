#ifndef MNIST_FROM_SCRATCH_ENGINE_H
#define MNIST_FROM_SCRATCH_ENGINE_H

#ifdef CUDA
#include "cuda.cuh"
#else
#include "matrix.h"
#endif

#include "params.h"

Parameters* gradient_descent(Matrix* X, double* Y, int Y_len, double lr, int epochs);

void eval(Matrix* X, double* Y, int N, Parameters* params);

#endif //MNIST_FROM_SCRATCH_ENGINE_H
