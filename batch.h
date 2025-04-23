#ifndef BATCH_H
#define BATCH_H
#include "matrix.h"

typedef struct Batch {
    Matrix* X;
    double* Y;
    int size;
} Batch;

Batch* create_mini_batch(Matrix* X, double* Y, int N, int batch_size);

void batch_destroy(Batch* batch);

#endif //BATCH_H
