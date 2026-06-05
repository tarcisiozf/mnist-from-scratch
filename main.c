#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>

#include "engine.h"
#include "dataset.h"
#include "evaldas.h"
#include "mem.h"

#define LEARNING_RATE 0.003
#define EPOCHS 40000

void write_matrix(FILE* f, const Matrix* m) {
    const uint32_t rows = m->rows;
    const uint32_t cols = m->cols;
    fwrite(&rows, sizeof(uint32_t), 1, f);
    fwrite(&cols, sizeof(uint32_t), 1, f);
    fwrite(m->data, sizeof(float), rows*cols, f);
}

void save_params(const char* str, const Parameters* params) {
    printf("num weights: %d\n",
        params->W1->rows*params->W1->cols+
        params->b1->rows*params->b1->cols+
        params->W2->rows*params->W2->cols+
        params->b2->rows*params->b2->cols);

    FILE* f = fopen(str, "wb");
    if (f == nullptr) {
        printf("Error opening file\n");
        exit(-1);
    }

    write_matrix(f, params->W1);
    write_matrix(f, params->b1);
    write_matrix(f, params->W2);
    write_matrix(f, params->b2);
    fflush(f);
    fclose(f);
}

int main(void) {
    srandom(time(NULL));

    const Dataset* train_dataset = read_dataset("output.bin");
    const Parameters* params = gradient_descent(
        matrix_transpose(train_dataset->X),
        train_dataset->Y,
        train_dataset->N,
        LEARNING_RATE,
        EPOCHS
    );

    save_params("weights.bin", params);

    return 0;
}
