#include "params.h"
#include "mem.h"
#include <stdlib.h>

Parameters* params_create() {
    Parameters* p = my_malloc(sizeof(Parameters));
    p->W1 = NULL;
    p->b1 = NULL;
    p->W2 = NULL;
    p->b2 = NULL;
    return p;
}

void params_destroy(Parameters* params) {
    if (params == NULL) {
        return;
    }
    matrix_destroy(params->W1);
    matrix_destroy(params->b1);
    matrix_destroy(params->W2);
    matrix_destroy(params->b2);
    my_free(params);
}