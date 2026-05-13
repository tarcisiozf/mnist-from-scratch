#include "matrix.h"
#include "mem.h"
#include <stdlib.h>
#include <stdio.h>
#include <memory.h>
#include <math.h>

#ifdef CUDA
#include "cuda.cuh"
#endif

Matrix* matrix_create(const int rows, const int cols) {
    Matrix* m = my_malloc(sizeof(Matrix));
    m->rows = rows;
    m->cols = cols;
    m->data = (float*) my_malloc(rows * cols * sizeof(float));
    memset(m->data, 0, rows * cols * sizeof(float));
    return m;
}

Matrix* matrix_from_shape(const Matrix* m) {
    return matrix_create(m->rows, m->cols);
}

Matrix* matrix_from_data(const int rows, const int cols, float* data) {
    Matrix* m = (Matrix*) my_malloc(sizeof(Matrix));
    m->rows = rows;
    m->cols = cols;
    m->data = data;
    return m;
}

void matrix_destroy(Matrix* m) {
    if (m == NULL) {
        return;
    }
    my_free(m->data);
    my_free(m);
}

Matrix* matrix_dot(const Matrix* a, const Matrix* b) {
    if (a->cols != b->rows) {
        printf("Error: Failed to multiply shapes (%d, %d) and (%d, %d)\n", a->rows, a->cols, b->rows, b->cols);
        exit(1);
    }

    Matrix* c = matrix_create(a->rows, b->cols);
#ifdef CUDA
    cuda_matmul(
        a->rows, a->cols, a->data,
        b->rows, b->cols, b->data,
        c->rows, c->cols, c->data
    );
#else
    for (int i = 0; i < a->rows; i++) {
        for (int j = 0; j < b->cols; j++) {
            float sum = 0;
            for (int k = 0; k < a->cols; k++) {
                sum += a->data[i * a->cols + k] * b->data[k * b->cols + j];
            }
            c->data[i * c->cols + j] = sum;
        }
    }
#endif
    return c;
}

Matrix* matrix_transpose(const Matrix* m) {
    Matrix* t = matrix_create(m->cols, m->rows);
    for (int i = 0; i < m->rows; i++) {
        for (int j = 0; j < m->cols; j++) {
            t->data[j * t->cols + i] = m->data[i * m->cols + j];
        }
    }
    return t;
}

Matrix* matrix_broadcast(const Matrix* a, const Matrix* b) {
    if (a->rows == b->rows && a->cols > b->cols) {
        Matrix* c = matrix_create(b->rows, a->cols);
        for (int i = 0; i < b->rows; i++) {
            for (int j = 0; j < a->cols; j++) {
                c->data[i * c->cols + j] = b->data[i * b->cols];
            }
        }
        return c;
    }
    if (a->rows > b->rows && a->cols == b->cols) {
        Matrix* c = matrix_create(a->rows, b->cols);
        for (int i = 0; i < a->rows; i++) {
            for (int j = 0; j < b->cols; j++) {
                c->data[i * c->cols + j] = b->data[j];
            }
        }
        return c;
    }
    printf("Error: Failed to broadcast shapes (%d, %d) and (%d, %d)\n", a->rows, a->cols, b->rows, b->cols);
    exit(1);
}

Matrix* matrix_add(const Matrix* a, const Matrix* b) {
    char did_broadcast = 0;
    Matrix* m = b;
    if (a->rows != b->rows || a->cols != b->cols) {
        m = matrix_broadcast(a, b);
        did_broadcast = 1;
    }

    Matrix* c = matrix_create(a->rows, a->cols);
    for (int i = 0; i < a->rows; i++) {
        for (int j = 0; j < a->cols; j++) {
            c->data[i * c->cols + j] = a->data[i * a->cols + j] + m->data[i * m->cols + j];
        }
    }

    if (did_broadcast) {
        matrix_destroy(m);
    }

    return c;
}

Matrix* matrix_sub(const Matrix* a, const Matrix* b) {
    if (a->rows != b->rows || a->cols != b->cols) {
        printf("Error: Failed to subtract shapes (%d, %d) and (%d, %d)\n", a->rows, a->cols, b->rows, b->cols);
        exit(1);
    }

    Matrix* c = matrix_from_shape(a);
    for (int i = 0; i < a->rows; i++) {
        for (int j = 0; j < a->cols; j++) {
            c->data[i * c->cols + j] = a->data[i * a->cols + j] - b->data[i * b->cols + j];
        }
    }
    return c;
}

Matrix* matrix_mul(const Matrix* a, const Matrix* b) {
    if (a->rows != b->rows || a->cols != b->cols) {
        printf("Error: Failed to multiply shapes (%d, %d) and (%d, %d)\n", a->rows, a->cols, b->rows, b->cols);
        exit(1);
    }

    Matrix* c = matrix_from_shape(a);
    for (int i = 0; i < a->rows; i++) {
        for (int j = 0; j < a->cols; j++) {
            c->data[i * c->cols + j] = a->data[i * a->cols + j] * b->data[i * b->cols + j];
        }
    }
    return c;
}

Matrix *matrix_div(const Matrix* a, const Matrix* b) {
    char did_broadcast = 0;
    Matrix* m = b;
    if (a->rows != b->rows || a->cols != b->cols) {
        m = matrix_broadcast(a, b);
        did_broadcast = 1;
    }

    Matrix* c = matrix_from_shape(a);
    for (int i = 0; i < a->rows; i++) {
        for (int j = 0; j < a->cols; j++) {
            c->data[i * c->cols + j] = a->data[i * a->cols + j] / m->data[i * m->cols + j];
        }
    }

    if (did_broadcast) {
        matrix_destroy(m);
    }

    return c;
}

Matrix* matrix_divf(const Matrix* a, const float f) {
    Matrix* c = matrix_from_shape(a);
    for (int i = 0; i < a->rows; i++) {
        for (int j = 0; j < a->cols; j++) {
            c->data[i * c->cols + j] = a->data[i * a->cols + j] / f;
        }
    }
    return c;
}

Matrix* matrix_subf(const Matrix* m, const float f) {
    Matrix* c = matrix_from_shape(m);
    for (int i = 0; i < m->rows; i++) {
        for (int j = 0; j < m->cols; j++) {
            c->data[i * c->cols + j] = m->data[i * m->cols + j] - f;
        }
    }
    return c;
}

Matrix* matrix_mulf(const Matrix* m, const float f) {
    Matrix* c = matrix_from_shape(m);
    for (int i = 0; i < m->rows; i++) {
        for (int j = 0; j < m->cols; j++) {
            c->data[i * c->cols + j] = m->data[i * m->cols + j] * f;
        }
    }
    return c;
}

void matrix_print(char* label, const Matrix* m, const int y, const int x) {
    printf("%s\n", label);
    int rows;
    int cols;

    if (y == -1 && x == -1) {
        rows = m->rows;
        cols = m->cols;
    } else {
        rows = y;
        cols = x;
    }
    if (rows > m->rows) {
        rows = m->rows;
    }
    if (cols > m->cols) {
        cols = m->cols;
    }

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            printf("%f ", m->data[i * m->cols + j]);
        }
        printf("\n");
    }
    printf("--------------\n");
}

// Uses Kahan summation for better precision
float matrix_sum(const Matrix* m) {
    float sum = 0.0f;
    float c = 0.0f;
    for (int i = 0; i < m->rows; i++) {
        for (int j = 0; j < m->cols; j++) {
            const float y = m->data[i * m->cols + j] - c;
            const float t = sum + y;
            c = (t - sum) - y;
            sum = t;
        }
    }
    return sum;
}

Matrix* matrix_rand(const int rows, const int cols) {
    Matrix* m = matrix_create(rows, cols);
    for (int i = 0; i < rows * cols; i++) {
        m->data[i] = ((float)random() / (float)RAND_MAX) - 0.5f;
    }
    return m;
}

Matrix* matrix_relu(const Matrix* m) {
    Matrix* c = matrix_from_shape(m);
    for (int i = 0; i < m->rows; i++) {
        for (int j = 0; j < m->cols; j++) {
            const int idx = i * m->cols + j;
            const float val = m->data[idx];
            c->data[idx] = val > 0 ? val : 0;
        }
    }
    return c;
}

Matrix* matrix_softmax(const Matrix* m) {
    Matrix* c = matrix_from_shape(m);
    Matrix* sum = matrix_create(1, m->cols);

    // Subtract per-column max for numerical stability (prevents expf overflow with floats)
    float col_max[m->cols];
    for (int j = 0; j < m->cols; j++) {
        col_max[j] = -INFINITY;
        for (int i = 0; i < m->rows; i++) {
            const float v = m->data[i * m->cols + j];
            if (v > col_max[j]) col_max[j] = v;
        }
    }

    for (int i = 0; i < c->rows; i++) {
        for (int j = 0; j < c->cols; j++) {
            c->data[i * c->cols + j] = expf(m->data[i * m->cols + j] - col_max[j]);
            sum->data[j] += c->data[i * c->cols + j];
        }
    }
    Matrix* out = matrix_div(c, sum);
    matrix_destroy(c);
    matrix_destroy(sum);
    return out;
}

Matrix* matrix_one_hot(const float* Y, const int len) {
    Matrix* m = matrix_create(len, 10); // int(max)+1
    for (int i = 0; i < len; i++) {
        m->data[i * 10 + (int) Y[i]] = 1;
    }
    Matrix* out = matrix_transpose(m);
    matrix_destroy(m);
    return out;
}

Matrix* matrix_cols(const Matrix* m, const int start, const int end) {
    Matrix* c = matrix_create(m->rows, end - start);
    for (int i = 0; i < m->rows; i++) {
        for (int j = start; j < end; j++) {
            c->data[i * c->cols + j - start] = m->data[i * m->cols + j];
        }
    }
    return c;
}

void matrix_replace(Matrix** m, Matrix* new_m) {
    if (*m != NULL) {
        matrix_destroy(*m);
    }
    *m = new_m;
}