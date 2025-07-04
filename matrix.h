#ifndef MNIST_FROM_SCRATCH_MATRIX_H
#define MNIST_FROM_SCRATCH_MATRIX_H

typedef struct Matrix {
    int rows;
    int cols;
    float* data;
} Matrix;

Matrix* matrix_create(int rows, int cols);

Matrix* matrix_from_shape(Matrix* m);

Matrix* matrix_from_data(int rows, int cols, float* data);

void matrix_destroy(Matrix* m);

Matrix* matrix_dot(Matrix* a, Matrix* b);

Matrix* matrix_transpose(Matrix* m);

Matrix* matrix_add(Matrix* a, Matrix* b);

Matrix* matrix_sub(Matrix* a, Matrix* b);

Matrix* matrix_mul(Matrix* a, Matrix* b);

Matrix* matrix_divf(Matrix* a, float f);

Matrix* matrix_subf(Matrix* m, float f);

Matrix* matrix_mulf(Matrix* m, float f);

float matrix_sum(Matrix* m);

void matrix_print(char* label, Matrix* m, int y, int x);

Matrix* matrix_rand(int rows, int cols);

Matrix* matrix_relu(Matrix* m);

Matrix* matrix_softmax(Matrix* m);

Matrix* matrix_one_hot(const float* Y, int len);

Matrix* matrix_cols(Matrix* m, int start, int end);

void matrix_replace(Matrix** m, Matrix* new_m);

#endif //MNIST_FROM_SCRATCH_MATRIX_H