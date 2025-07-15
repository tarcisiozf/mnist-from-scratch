#include <stdio.h>
#include <stdlib.h>
#include <memory.h>
#include <pthread.h>
#include <unistd.h>

#include "engine.h"
#include "batch.h"
#include "mem.h"

typedef struct BackwardParameters {
    Matrix* W1;
    double b1;
    Matrix* W2;
    double b2;
} BackwardParameters;

Parameters* init_params() {
    Parameters* params = params_create();
    params->W1 = matrix_rand(800, 784);
    params->b1 = matrix_rand(800, 1);
    params->W2 = matrix_rand(10, 800);
    params->b2 = matrix_rand(10, 1);
    return params;
}

Parameters* forward(Parameters* params, Matrix* X) {
    Parameters* z = params_create();

    Matrix *w1x = matrix_dot(params->W1, X);
    z->W1 = matrix_add(w1x, params->b1);
    z->b1 = matrix_relu(z->W1);
    Matrix *w2a1 = matrix_dot(params->W2, z->b1);
    z->W2 = matrix_add(w2a1, params->b2);
    z->b2 = matrix_softmax(z->W2);

    matrix_destroy(w1x);
    matrix_destroy(w2a1);

    return z;
}

Matrix* deriv_relu(Matrix* m) {
    Matrix* c = matrix_from_shape(m);
    for (int i = 0; i < m->rows * m->cols; i++) {
        c->data[i] = m->data[i] > 0 ? 1 : 0;
    }
    return c;
}

BackwardParameters* backprop(Parameters* forward_params, Matrix* W2, Matrix* X, double* Y, int N) {
    BackwardParameters* back_params = my_malloc(sizeof(BackwardParameters));

    const int m = N;
    const double f = 1 / (double ) m;
    Matrix* ohY = matrix_one_hot(Y, N);
    Matrix* dZ2 = matrix_sub(forward_params->b2, ohY);
    Matrix *a1T = matrix_transpose(forward_params->W1);
    Matrix *dZ2a1 = matrix_dot(dZ2, a1T);
    back_params->W2 = matrix_mulf(dZ2a1, f);
    back_params->b2 = matrix_sum(dZ2) * f;
    Matrix *w2T = matrix_transpose(W2);
    Matrix *w2dZ2 = matrix_dot(w2T, dZ2);
    Matrix *drZ1 = deriv_relu(forward_params->W1);
    Matrix* dZ1 = matrix_mul(w2dZ2, drZ1);
    Matrix *xT = matrix_transpose(X);
    Matrix *dZ1x = matrix_dot(dZ1, xT);
    back_params->W1 = matrix_mulf(dZ1x, f);
    back_params->b1 = matrix_sum(dZ1) * f;

    matrix_destroy(ohY);
    matrix_destroy(dZ2);
    matrix_destroy(dZ1);
    matrix_destroy(a1T);
    matrix_destroy(dZ2a1);
    matrix_destroy(w2T);
    matrix_destroy(w2dZ2);
    matrix_destroy(drZ1);
    matrix_destroy(dZ1x);
    matrix_destroy(xT);

    return back_params;
}

void update_params(Parameters* params, BackwardParameters* back_params, double lr) {
    Matrix *dw1Lr = matrix_mulf(back_params->W1, lr);
    Matrix* W1 = matrix_sub(params->W1, dw1Lr);
    Matrix* b1 = matrix_subf(params->b1, back_params->b1 * lr);
    Matrix *dw2Lr = matrix_mulf(back_params->W2, lr);
    Matrix* W2 = matrix_sub(params->W2, dw2Lr);
    Matrix* b2 = matrix_subf(params->b2, back_params->b2 * lr);

    matrix_destroy(params->W1);
    matrix_destroy(params->b1);
    matrix_destroy(params->W2);
    matrix_destroy(params->b2);
    matrix_destroy(dw1Lr);
    matrix_destroy(dw2Lr);

    params->W1 = W1;
    params->b1 = b1;
    params->W2 = W2;
    params->b2 = b2;
}

double* prediction(Matrix *a2) {
    double* out = (double*) my_malloc(a2->cols * sizeof(double));
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

double accuracy(const double* predictions, const double* groundTruth, const int n) {
    int correct = 0;
    for (int i = 0; i < n; i++) {
        if (predictions[i] == groundTruth[i]) {
            correct++;
        }
    }
    return ((double) correct) / ((double) n);
}

void backward_parameters_destroy(BackwardParameters* params) {
    matrix_destroy(params->W1);
    matrix_destroy(params->W2);
    my_free(params);
}

typedef struct Foo {
    Matrix* X;
    double* Y;
    int N;
    double lr;
    int epochs;
} Foo;

void* foo(void* arg) {
    Foo* foo = arg;
    Parameters* params = init_params();

    int batch_size = 100;

    int iters = foo->epochs / 10;
    for (int i = 0; i < iters; i++) {
        Batch* batch = create_mini_batch(foo->X, foo->Y, foo->N, batch_size);

        Parameters* forward_params = forward(params, batch->X);
        BackwardParameters* backward_params = backprop(forward_params, params->W2, batch->X, batch->Y, batch->size);
        update_params(params, backward_params, foo->lr); // allreduce
        if (i % 50 == 0 || i == iters - 1) {
            printf("Epoch %d\n", i);
            double* pred = prediction(forward_params->b2);
            printf("Accuracy: %f\n", accuracy(pred, batch->Y, batch->size));
            my_free(pred);
        }

        backward_parameters_destroy(backward_params);
        params_destroy(forward_params);
        batch_destroy(batch);
    }

    return params;
}

Parameters* gradient_descent(Matrix* X, double* Y, int N, double lr, int epochs) {
    long cpus = sysconf(_SC_NPROCESSORS_ONLN);
    printf("Detected %ld CPUs\n", cpus);

    int nt = (int)cpus;
    pthread_t threads[nt];

    Foo arg = {
        .X = X,
        .Y = Y,
        .N = N,
        .lr = lr,
        .epochs = epochs
    };

    for (int i = 0; i < nt; i++) {
        int errcode = pthread_create(&threads[i], NULL, foo, &arg);
        if (errcode != 0) {
            fprintf(stderr, "Error creating thread %d: %s\n", i, strerror(errcode));
            exit(EXIT_FAILURE);
        }
    }

    void* params = NULL;
    for (int i = 0; i < nt; i++) {
        int errcode = pthread_join(threads[i], params);
        if (errcode != 0) {
            fprintf(stderr, "Error joining thread %d: %s\n", i, strerror(errcode));
            exit(EXIT_FAILURE);
        }
        printf("Thread %d finished successfully\n", i);
    }

    printf("here\n");
    return params;
}

void eval(Matrix* X, double* Y, int N, Parameters* input) {
    Parameters* params = forward(input, X);
    double* pred = prediction(params->b2);
    printf("Eval accuracy: %f\n", accuracy(pred, Y, N));
    my_free(pred);
    params_destroy(params);
}