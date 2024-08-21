#include "matrix.h"
#include <stdlib.h>
#include <stdio.h>
#include <memory.h>
#include <math.h>

Matrix* matrix_create(int rows, int cols) {
    Matrix* m = (Matrix*) malloc(sizeof(Matrix));
    m->rows = rows;
    m->cols = cols;
    m->data = (double*) malloc(rows * cols * sizeof(double));
    memset(m->data, 0, rows * cols * sizeof(double));
    return m;
}

Matrix* matrix_from_shape(Matrix* m) {
    return matrix_create(m->rows, m->cols);
}

Matrix* matrix_from_data(int rows, int cols, double* data) {
    Matrix* m = (Matrix*) malloc(sizeof(Matrix));
    m->rows = rows;
    m->cols = cols;
    m->data = data;
    return m;
}

void matrix_free(Matrix* m) {
    if (m == NULL) {
        return;
    }
    free(m->data);
    free(m);
}

Matrix* matrix_dot(Matrix* a, Matrix* b) {
    if (a->cols != b->rows) {
        printf("Error: Failed to multiply shapes (%d, %d) and (%d, %d)\n", a->rows, a->cols, b->rows, b->cols);
        exit(1);
    }

    Matrix* c = matrix_create(a->rows, b->cols);
    for (int i = 0; i < a->rows; i++) {
        for (int j = 0; j < b->cols; j++) {
            double sum = 0;
            for (int k = 0; k < a->cols; k++) {
                sum += a->data[i * a->cols + k] * b->data[k * b->cols + j];
            }
            c->data[i * c->cols + j] = sum;
        }
    }
    return c;
}

Matrix* matrix_transpose(Matrix* m) {
    Matrix* t = matrix_create(m->cols, m->rows);
    for (int i = 0; i < m->rows; i++) {
        for (int j = 0; j < m->cols; j++) {
            t->data[j * t->cols + i] = m->data[i * m->cols + j];
        }
    }
    return t;
}

Matrix* matrix_broadcast(Matrix* a, Matrix* b) {
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

Matrix* matrix_add(Matrix* a, Matrix* b) {
    char did_broadcast = 0;
    if (a->rows != b->rows || a->cols != b->cols) {
        b = matrix_broadcast(a, b);
        did_broadcast = 1;
    }

    Matrix* c = matrix_create(a->rows, a->cols);
    for (int i = 0; i < a->rows; i++) {
        for (int j = 0; j < a->cols; j++) {
            c->data[i * c->cols + j] = a->data[i * a->cols + j] + b->data[i * b->cols + j];
        }
    }

    if (did_broadcast) {
        matrix_free(b);
    }

    return c;
}

Matrix* matrix_sub(Matrix* a, Matrix* b) {
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

Matrix* matrix_mul(Matrix* a, Matrix* b) {
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

Matrix *matrix_div(Matrix *a, Matrix *b) {
    char did_broadcast = 0;
    if (a->rows != b->rows || a->cols != b->cols) {
        b = matrix_broadcast(a, b);
        did_broadcast = 1;
    }

    Matrix* c = matrix_from_shape(a);
    for (int i = 0; i < a->rows; i++) {
        for (int j = 0; j < a->cols; j++) {
            c->data[i * c->cols + j] = a->data[i * a->cols + j] / b->data[i * b->cols + j];
        }
    }

    if (did_broadcast) {
        matrix_free(b);
    }

    return c;
}

Matrix* matrix_divf(Matrix* a, double f) {
    Matrix* c = matrix_from_shape(a);
    for (int i = 0; i < a->rows; i++) {
        for (int j = 0; j < a->cols; j++) {
            c->data[i * c->cols + j] = a->data[i * a->cols + j] / f;
        }
    }
    return c;
}

Matrix* matrix_subf(Matrix* m, double f) {
    Matrix* c = matrix_from_shape(m);
    for (int i = 0; i < m->rows; i++) {
        for (int j = 0; j < m->cols; j++) {
            c->data[i * c->cols + j] = m->data[i * m->cols + j] - f;
        }
    }
    return c;
}

Matrix* matrix_mulf(Matrix* m, double f) {
    Matrix* c = matrix_from_shape(m);
    for (int i = 0; i < m->rows; i++) {
        for (int j = 0; j < m->cols; j++) {
            c->data[i * c->cols + j] = m->data[i * m->cols + j] * f;
        }
    }
    return c;
}

void matrix_print(char* label, Matrix* m, int y, int x) {
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
double matrix_sum(Matrix* m) {
    double sum = 0.0;
    double c = 0.0;
    for (int i = 0; i < m->rows; i++) {
        for (int j = 0; j < m->cols; j++) {
            double y = m->data[i * m->cols + j] - c;
            double t = sum + y;
            c = (t - sum) - y;
            sum = t;
        }
    }
    return sum;
}

Matrix* matrix_rand(int rows, int cols) {
    Matrix* m = matrix_create(rows, cols);
    for (int i = 0; i < rows * cols; i++) {
        m->data[i] = (double) rand() / RAND_MAX - 0.5;
    }
    return m;
}

Matrix* matrix_relu(Matrix* m) {
    Matrix* c = matrix_from_shape(m);
    int idx;
    double val;
    for (int i = 0; i < m->rows; i++) {
        for (int j = 0; j < m->cols; j++) {
            idx = i * m->cols + j;
            val = m->data[idx];
            c->data[idx] = val > 0 ? val : 0;
        }
    }
    return c;
}

Matrix* matrix_softmax(Matrix* m) {
    Matrix* c = matrix_from_shape(m);
    Matrix* sum = matrix_create(1, m->cols);
    for (int i = 0; i < c->rows; i++) {
        for (int j = 0; j < c->cols; j++) {
            c->data[i * c->cols + j] = exp(m->data[i * m->cols + j]);
            sum->data[j] += c->data[i * c->cols + j];
        }
    }
    Matrix* out = matrix_div(c, sum);
    matrix_free(c);
    matrix_free(sum);
    return out;
}

Matrix* matrix_one_hot(const double* Y, int len) {
    Matrix* m = matrix_create(len, 10); // int(max)+1
    for (int i = 0; i < len; i++) {
        m->data[i * 10 + (int) Y[i]] = 1;
    }
    Matrix* out = matrix_transpose(m);
    matrix_free(m);
    return out;
}

Matrix* matrix_cols(Matrix* m, int start, int end) {
    Matrix* c = matrix_create(m->rows, end - start);
    for (int i = 0; i < m->rows; i++) {
        for (int j = start; j < end; j++) {
            c->data[i * c->cols + j - start] = m->data[i * m->cols + j];
        }
    }
    return c;
}