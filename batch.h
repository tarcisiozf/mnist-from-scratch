#ifndef BATCH_H
#define BATCH_H
#include "matrix.h"

typedef struct Batch {
    Matrix* X;
    float* Y;
    int size;
} Batch;

Batch* create_mini_batch(const Matrix* X, const float* Y, int N, int batch_size);

void batch_destroy(Batch* batch);

#endif //BATCH_H
