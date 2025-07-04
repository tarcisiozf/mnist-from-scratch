#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>
#include "engine.h"
#include "dataset.h"
#include "mem.h"

#define LEARNING_RATE 0.1
#define EPOCHS 1000

int main(void) {
    srandom(time(NULL));

    Dataset* dataset = read_dataset("./dataset.bin");

    Matrix* T = matrix_transpose(dataset->X);
    matrix_replace(&dataset->X, matrix_divf(T, 255));
    matrix_destroy(T);

    int N_test = 1000;
    Matrix* X_test = matrix_cols(dataset->X, 0, N_test);
    float* Y_test = dataset->Y;

    int N_train = dataset->N - N_test;
    Matrix* X_train = matrix_cols(dataset->X, N_test, dataset->N);
    float* Y_train = &dataset->Y[N_test];

    Parameters* params = gradient_descent(X_train, Y_train, N_train, LEARNING_RATE, EPOCHS);
    eval(X_test, Y_test, N_test, params);

    params_destroy(params);
    matrix_destroy(X_train);
    matrix_destroy(X_test);
    dataset_destroy(dataset);

    print_memory_usage();

    return 0;
}
