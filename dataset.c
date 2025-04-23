//
// Created by Tarcisio Zotelli Ferraz on 23/04/25.
//

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include "dataset.h"
#include "mem.h"

#define NUM_PIXELS 784

Dataset* read_dataset(const char *filename) {
    FILE *file = fopen(filename, "rb");
    if (!file) {
        perror("Failed to open file");
        exit(EXIT_FAILURE);
    }

    Dataset* dataset = (Dataset*) my_malloc(sizeof(Dataset));

    uint32_t numRecords;
    fread(&numRecords, sizeof(uint32_t), 1, file);

    dataset->N = numRecords;

    double* labels = (double*) my_malloc(numRecords * sizeof(double));

    uint32_t* pixels = (uint32_t*) my_malloc(numRecords * NUM_PIXELS * sizeof(uint32_t));
    double* pixelsf = (double*) my_malloc(numRecords * NUM_PIXELS * sizeof(double));
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
    my_free(pixels);

    dataset->X = matrix_from_data(numRecords, NUM_PIXELS, pixelsf);
    dataset->Y = labels;

    fclose(file);

    return dataset;
}

void dataset_destroy(Dataset* dataset) {
    matrix_destroy(dataset->X);
    my_free(dataset->Y);
    my_free(dataset);
}
