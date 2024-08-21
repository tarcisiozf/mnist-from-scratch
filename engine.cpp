#include <stdio.h>
#include <stdlib.h>
#include <memory.h>
#include "engine.h"

void init_params(Matrix** W1, Matrix** b1, Matrix** W2, Matrix** b2) {
    *W1 = matrix_rand(800, 784);
    *b1 = matrix_rand(800, 1);
    *W2 = matrix_rand(10, 800);
    *b2 = matrix_rand(10, 1);
}

void forward(Matrix* W1, Matrix* b1, Matrix* W2, Matrix* b2, Matrix* X, Matrix** z1, Matrix** a1, Matrix** z2, Matrix** a2) {
    matrix_free(*z1);
    matrix_free(*a1);
    matrix_free(*z2);
    matrix_free(*a2);

    Matrix *w1x = matrix_dot(W1, X);
    *z1 = matrix_add(w1x, b1);
    *a1 = matrix_relu(*z1);
    Matrix *w2a1 = matrix_dot(W2, *a1);
    *z2 = matrix_add(w2a1, b2);
    *a2 = matrix_softmax(*z2);

    matrix_free(w1x);
    matrix_free(w2a1);
}

Matrix* deriv_relu(Matrix* m) {
    Matrix* c = matrix_from_shape(m);
    for (int i = 0; i < m->rows * m->cols; i++) {
        c->data[i] = m->data[i] > 0 ? 1 : 0;
    }
    return c;
}

void backprop(Matrix* z1, Matrix* a1, Matrix* z2, Matrix* a2, Matrix* W2, Matrix* X, double* Y, int N, Matrix** dW1, double* db1, Matrix** dW2, double* db2) {
    matrix_free(*dW1);
    matrix_free(*dW2);

    int m = N;
    double f = 1 / (double ) m;
    Matrix* ohY = matrix_one_hot(Y, N);
    Matrix* dZ2 = matrix_sub(a2, ohY);
    Matrix *a1T = matrix_transpose(a1);
    Matrix *dZ2a1 = matrix_dot(dZ2, a1T);
    *dW2 = matrix_mulf(dZ2a1, f);
    *db2 = matrix_sum(dZ2) * f;
    Matrix *w2T = matrix_transpose(W2);
    Matrix *w2dZ2 = matrix_dot(w2T, dZ2);
    Matrix *drZ1 = deriv_relu(z1);
    Matrix* dZ1 = matrix_mul(w2dZ2, drZ1);
    Matrix *xT = matrix_transpose(X);
    Matrix *dZ1x = matrix_dot(dZ1, xT);
    *dW1 = matrix_mulf(dZ1x, f);
    *db1 = matrix_sum(dZ1) * f;

    matrix_free(ohY);
    matrix_free(dZ2);
    matrix_free(dZ1);
    matrix_free(a1T);
    matrix_free(dZ2a1);
    matrix_free(w2T);
    matrix_free(w2dZ2);
    matrix_free(drZ1);
    matrix_free(dZ1x);
    matrix_free(xT);
}

void update_params(Matrix** W1, Matrix** b1, Matrix** W2, Matrix** b2, Matrix* dW1, double db1, Matrix* dW2, double db2, double lr) {
    Matrix *dw1Lr = matrix_mulf(dW1, lr);
    Matrix* _W1 = matrix_sub(*W1, dw1Lr);
    Matrix* _b1 = matrix_subf(*b1, db1 * lr);
    Matrix *dw2Lr = matrix_mulf(dW2, lr);
    Matrix* _W2 = matrix_sub(*W2, dw2Lr);
    Matrix* _b2 = matrix_subf(*b2, db2 * lr);

    matrix_free(*W1);
    matrix_free(*b1);
    matrix_free(*W2);
    matrix_free(*b2);
    matrix_free(dw1Lr);
    matrix_free(dw2Lr);

    *W1 = _W1;
    *b1 = _b1;
    *W2 = _W2;
    *b2 = _b2;
}

double* prediction(Matrix *a2) {
    double* out = (double*) malloc(a2->cols * sizeof(double));
    for (int x = 0; x < a2->cols; x++) {
        double max = 0;
        int idx = 0;
        for (int y = 0; y < a2->rows; y++) {
            if (a2->data[y * a2->cols + x] > max) {
                max = a2->data[y * a2->cols + x];
                idx = y;
            }
        }
        out[x] = idx;
    }
    return out;
}

double accuracy(const double* predictions, const double* groundTruth, int n) {
    int correct = 0;
    for (int i = 0; i < n; i++) {
        if (predictions[i] == groundTruth[i]) {
            correct++;
        }
    }
    return ((double) correct) / ((double) n);
}

void create_mini_batch(Matrix* X, double* Y, int N, int N_batch, Matrix** X_batch, double* Y_batch) {
    matrix_free(*X_batch);
    int indices[N_batch];
    memset(indices, -1, N_batch * sizeof(int));

    for (int i = 0; i < N_batch; i++) {
        indices[i] = rand() % N;
        for (int j = 0; j < i - 1; j++) {
            if (indices[j] == indices[i]) {
                i--;
                break;
            }
        }
    }

    Matrix* c = matrix_create(X->rows, N_batch);
    for (int r = 0; r < N_batch; r++) {
        int col = indices[r];
        for (int i = 0; i < X->rows; i++) {
            c->data[i * c->cols + r] = X->data[i * X->cols + col];
        }
        Y_batch[r] = Y[col];
    }

    *X_batch = c;
}

void gradient_descent(Matrix* X, double* Y, int N, double lr, int epochs, Matrix** W1, Matrix** b1, Matrix** W2, Matrix** b2) {
    Matrix* z1 = NULL;
    Matrix* a1 = NULL;
    Matrix* z2 = NULL;
    Matrix* a2 = NULL;
    Matrix* dW1 = NULL;
    Matrix* dW2 = NULL;
    double db1 = 0;
    double db2 = 0;

    init_params(W1, b1, W2, b2);

    int N_batch = 100;
    Matrix* X_batch = NULL;
    double Y_batch[N_batch];

    for (int i = 0; i < epochs; i++) {
        create_mini_batch(X, Y, N, N_batch, &X_batch, Y_batch);

        forward(*W1, *b1, *W2, *b2, X_batch, &z1, &a1, &z2, &a2);
        backprop(z1, a1, z2, a2, *W2, X_batch, Y_batch, N_batch, &dW1, &db1, &dW2, &db2);
        update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, lr);
        if (i % 50 == 0 || i == epochs - 1) {
            printf("Epoch %d\n", i);
            printf("Accuracy: %f\n", accuracy(prediction(a2), Y_batch, N_batch));
        }
    }

    matrix_free(X_batch);
}

void eval(Matrix* X, double* Y, int N, Matrix* W1, Matrix* b1, Matrix* W2, Matrix* b2) {
    Matrix* z1 = NULL;
    Matrix* a1 = NULL;
    Matrix* z2 = NULL;
    Matrix* a2 = NULL;

    forward(W1, b1, W2, b2, X, &z1, &a1, &z2, &a2);
    printf("Eval accuracy: %f\n", accuracy(prediction(a2), Y, N));

    matrix_free(z1);
    matrix_free(a1);
    matrix_free(z2);
    matrix_free(a2);
}