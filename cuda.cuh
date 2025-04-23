#ifndef MNIST_FROM_SCRATCH_CUDA_H
#define MNIST_FROM_SCRATCH_CUDA_H

#ifdef __cplusplus
extern "C" {
#endif

void cuda_matmul(
    int row_a, int col_a, double* data_a,
    int row_b, int col_b, double* data_b,
    int row_c, int col_c, double* data_c
);

#ifdef __cplusplus
}
#endif

#endif // MNIST_FROM_SCRATCH_CUDA_H
