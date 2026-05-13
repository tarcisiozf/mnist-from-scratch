#ifndef MNIST_FROM_SCRATCH_MATRIX_H
#define MNIST_FROM_SCRATCH_MATRIX_H

typedef struct Matrix {
    int rows;
    int cols;
    float* data;
} Matrix;

Matrix* matrix_create(int rows, int cols);

Matrix* matrix_from_shape(const Matrix* m);

Matrix* matrix_from_data(int rows, int cols, float* data);

void matrix_destroy(Matrix* m);

Matrix* matrix_dot(const Matrix* a, const Matrix* b);

Matrix* matrix_transpose(const Matrix* m);

Matrix* matrix_add(const Matrix* a, const Matrix* b);

Matrix* matrix_sub(const Matrix* a, const Matrix* b);

Matrix* matrix_mul(const Matrix* a, const Matrix* b);

Matrix* matrix_divf(const Matrix* a, float f);

Matrix* matrix_subf(const Matrix* m, float f);

Matrix* matrix_mulf(const Matrix* m, float f);

float matrix_sum(const Matrix* m);

void matrix_print(char* label, const Matrix* m, int y, int x);

Matrix* matrix_rand(int rows, int cols);

Matrix* matrix_relu(const Matrix* m);

Matrix* matrix_softmax(const Matrix* m);

Matrix* matrix_one_hot(const float* Y, int len);

Matrix* matrix_cols(const Matrix* m, int start, int end);

void matrix_replace(Matrix** m, Matrix* new_m);

#endif //MNIST_FROM_SCRATCH_MATRIX_H