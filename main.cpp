#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>
#include "engine.h"

#define NUM_PIXELS 784

void readDataset(const char *filename, Matrix** X, double** Y, int* N) {
    FILE *file = fopen(filename, "rb");
    if (!file) {
        perror("Failed to open file");
        exit(EXIT_FAILURE);
    }

    uint32_t numRecords;
    fread(&numRecords, sizeof(uint32_t), 1, file);

    *N = numRecords;

    double* labels = (double*) malloc(numRecords * sizeof(double));

    uint32_t* pixels = (uint32_t*) malloc(numRecords * NUM_PIXELS * sizeof(uint32_t));
    double* pixelsf = (double*) malloc(numRecords * NUM_PIXELS * sizeof(double));
    uint32_t p;
    uint32_t label;

    for (uint32_t i = 0; i < numRecords; i++) {
        fread(&label, sizeof(uint32_t), 1, file);

        labels[i] = label;

        for (int j = 0; j < NUM_PIXELS; j++) {
            fread(&p, sizeof(uint32_t), 1, file);
            pixels[i*NUM_PIXELS + j] = p;
        }
    }

    for (int i = 0; i < numRecords * NUM_PIXELS; i++) {
        pixelsf[i] = (double) pixels[i];
    }
    free(pixels);

    *X = matrix_from_data(numRecords, NUM_PIXELS, pixelsf);
    *Y = labels;

    fclose(file);
}

int main(void) {
    Matrix* X;
    double* Y;
    int N;

    srand(time(NULL));
    readDataset("./dataset.bin", &X, &Y, &N);

    X = matrix_divf(matrix_transpose(X), 255);

    int N_test = 1000;
    Matrix* X_test = matrix_cols(X, 0, N_test);
    double* Y_test = Y;

    int N_train = N - N_test;
    Matrix* X_train = matrix_cols(X, N_test, N);
    double* Y_train = &Y[N_test];

    Matrix* W1;
    Matrix* b1;
    Matrix* W2;
    Matrix* b2;

    gradient_descent(X_train, Y_train, N_train, 0.1, 1000, &W1, &b1, &W2, &b2);

    eval(X_test, Y_test, N_test, W1, b1, W2, b2);

    return 0;
}
