//
// Created by Tarcisio Zotelli Ferraz on 23/04/25.
//

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include "dataset.h"
#include "mem.h"
#include "mmap.h"

#define DIM 14

Dataset* read_dataset(const char *filename) {
    size_t size;
    const u8* data = mopen(filename, &size);
    size_t offset = 0;

    const uint32_t numRecords = *(uint32_t*)(data + offset);
    offset += sizeof(uint32_t);

    const float* labels = (float*)(data + offset);
    offset += numRecords * sizeof(float);

    const float* vectors = (float*)(data + offset);

    Dataset* dataset = my_malloc(sizeof(Dataset));
    dataset->N = numRecords;
    dataset->X = matrix_from_data(numRecords, DIM, vectors);
    dataset->Y = labels;

    return dataset;
}

void dataset_destroy(Dataset* dataset) {}