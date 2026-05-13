#ifndef DATASET_H
#define DATASET_H
#include "matrix.h"

typedef struct Dataset {
    Matrix* X;
    float* Y;
    uint32_t N;
} Dataset;

Dataset* read_dataset(const char *filename);

void dataset_destroy(Dataset* dataset);

#endif //DATASET_H
