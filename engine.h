#ifndef MNIST_FROM_SCRATCH_ENGINE_H
#define MNIST_FROM_SCRATCH_ENGINE_H

#ifdef CUDA
#include "cuda.cuh"
#else
#include "matrix.h"
#endif

void init_params(Matrix** W1, Matrix** b1, Matrix** W2, Matrix** b2);

void forward(Matrix* W1, Matrix* b1, Matrix* W2, Matrix* b2, Matrix* X, Matrix** z1, Matrix** a1, Matrix** z2, Matrix** a2);

void backprop(Matrix* z1, Matrix* a1, Matrix* z2, Matrix* a2, Matrix* W2, Matrix* X, double* Y, int Y_len, Matrix** dW1, double* db1, Matrix** dW2, double* db2);

void update_params(Matrix** W1, Matrix** b1, Matrix** W2, Matrix** b2, Matrix* dW1, double db1, Matrix* dW2, double db2, double lr);

void gradient_descent(Matrix* X, double* Y, int Y_len, double lr, int epochs, Matrix** W1, Matrix** b1, Matrix** W2, Matrix** b2);

double* prediction(Matrix *a2);

void eval(Matrix* X, double* Y, int N, Matrix* W1, Matrix* b1, Matrix* W2, Matrix* b2);

#endif //MNIST_FROM_SCRATCH_ENGINE_H
