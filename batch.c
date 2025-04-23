#include <string.h>
#include <stdlib.h>

#include "batch.h"
#include "mem.h"

Batch* create_mini_batch(Matrix* X, double* Y, int N, int batch_size) {
    int indices[batch_size];
    memset(indices, -1, batch_size * sizeof(int));

    for (int i = 0; i < batch_size; i++) {
        indices[i] = random() % N;
        for (int j = 0; j < i - 1; j++) {
            if (indices[j] == indices[i]) {
                i--;
                break;
            }
        }
    }

    Batch* batch = (Batch*) my_malloc(sizeof(Batch));
    batch->size = batch_size;
    batch->X = matrix_create(X->rows, batch_size);
    batch->Y = (double*) my_malloc(batch_size * sizeof(double));

    for (int r = 0; r < batch_size; r++) {
        int col = indices[r];
        for (int i = 0; i < X->rows; i++) {
            batch->X->data[i * batch->X->cols + r] = X->data[i * X->cols + col];
        }
        batch->Y[r] = Y[col];
    }

    return batch;
}

void batch_destroy(Batch* batch) {
    matrix_destroy(batch->X);
    my_free(batch->Y);
    my_free(batch);
}