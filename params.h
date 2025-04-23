#ifndef MODEL_H
#define MODEL_H
#include "matrix.h"

typedef struct Parameters {
    Matrix* W1;
    Matrix* b1;
    Matrix* W2;
    Matrix* b2;
} Parameters;

Parameters* params_create();

void params_destroy(Parameters* params);

#endif //MODEL_H
