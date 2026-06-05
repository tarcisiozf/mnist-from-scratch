#include <stdio.h>
#include "engine.h"

#include <math.h>

#include "batch.h"
#include "mem.h"

typedef struct BackwardParameters {
    Matrix* W1;
    Matrix* b1;
    Matrix* W2;
    Matrix* b2;
} BackwardParameters;

#define INPUT_LAYER 14
#define HIDDEN_LAYER_1 32
// Target scores come from a top_k=5 neighbor vote, so the legal output
// support is {0/5, 1/5, 2/5, 3/5, 4/5, 5/5}. Modeling as 6-class
// classification eliminates the bimodal mode-collapse the regressor hit.
#define OUTPUT_LAYER 6
#define SCORE_STEP 0.2f

static Parameters* init_params() {
    Parameters* params = params_create();
    params->W1 = matrix_rand_he(HIDDEN_LAYER_1, INPUT_LAYER, INPUT_LAYER);
    params->b1 = matrix_create(HIDDEN_LAYER_1, 1);
    params->W2 = matrix_rand_he(OUTPUT_LAYER, HIDDEN_LAYER_1, HIDDEN_LAYER_1);
    params->b2 = matrix_create(OUTPUT_LAYER, 1);
    return params;
}

Parameters* forward(const Parameters* params, const Matrix* X) {
    Parameters* z = params_create();

    Matrix *w1x = matrix_dot(params->W1, X);
    z->W1 = matrix_add(w1x, params->b1);
    z->b1 = matrix_relu(z->W1);
    Matrix *w2a1 = matrix_dot(params->W2, z->b1);
    z->W2 = matrix_add(w2a1, params->b2);
    z->b2 = matrix_softmax(z->W2);

    matrix_destroy(w1x);
    matrix_destroy(w2a1);

    return z;
}

static Matrix* deriv_relu(const Matrix* m) {
    Matrix* c = matrix_from_shape(m);
    for (int i = 0; i < m->rows * m->cols; i++) {
        c->data[i] = m->data[i] > 0 ? 1 : 0;
    }
    return c;
}

static BackwardParameters* backprop(const Parameters* forward_params, const Matrix* W2, const Matrix* X, const float* Y, const int N) {
    BackwardParameters* back_params = my_malloc(sizeof(BackwardParameters));

    const int m = N;
    const float f = 1.0f / (float)m;
    // One-hot encode the discrete score: cls = round(Y * 5), Y ∈ {0, 0.2, …, 1.0}.
    const int n_classes = forward_params->b2->rows;
    Matrix* yMat = matrix_create(n_classes, N);
    for (int i = 0; i < N; i++) {
        int cls = (int)(Y[i] * 5.0f + 0.5f);
        if (cls < 0) cls = 0;
        if (cls >= n_classes) cls = n_classes - 1;
        yMat->data[cls * N + i] = 1.0f;
    }
    Matrix* dZ2 = matrix_sub(forward_params->b2, yMat);
    Matrix *a1T = matrix_transpose(forward_params->b1);
    Matrix *dZ2a1 = matrix_dot(dZ2, a1T);
    back_params->W2 = matrix_mulf(dZ2a1, f);
    Matrix* db2_sum = matrix_row_sum(dZ2);
    back_params->b2 = matrix_mulf(db2_sum, f);
    Matrix *w2T = matrix_transpose(W2);
    Matrix *w2dZ2 = matrix_dot(w2T, dZ2);
    Matrix *drZ1 = deriv_relu(forward_params->W1);
    Matrix* dZ1 = matrix_mul(w2dZ2, drZ1);
    Matrix *xT = matrix_transpose(X);
    Matrix *dZ1x = matrix_dot(dZ1, xT);
    back_params->W1 = matrix_mulf(dZ1x, f);
    Matrix* db1_sum = matrix_row_sum(dZ1);
    back_params->b1 = matrix_mulf(db1_sum, f);

    matrix_destroy(yMat);
    matrix_destroy(dZ2);
    matrix_destroy(dZ1);
    matrix_destroy(a1T);
    matrix_destroy(dZ2a1);
    matrix_destroy(w2T);
    matrix_destroy(w2dZ2);
    matrix_destroy(drZ1);
    matrix_destroy(dZ1x);
    matrix_destroy(xT);
    matrix_destroy(db2_sum);
    matrix_destroy(db1_sum);

    return back_params;
}

typedef struct AdamState {
    Matrix* m_W1;
    Matrix* m_b1;
    Matrix* m_W2;
    Matrix* m_b2;
    Matrix* v_W1;
    Matrix* v_b1;
    Matrix* v_W2;
    Matrix* v_b2;
    int t;
} AdamState;

static AdamState* adam_init(const Parameters* params) {
    AdamState* a = my_malloc(sizeof(AdamState));
    a->m_W1 = matrix_create(params->W1->rows, params->W1->cols);
    a->m_b1 = matrix_create(params->b1->rows, params->b1->cols);
    a->m_W2 = matrix_create(params->W2->rows, params->W2->cols);
    a->m_b2 = matrix_create(params->b2->rows, params->b2->cols);
    a->v_W1 = matrix_create(params->W1->rows, params->W1->cols);
    a->v_b1 = matrix_create(params->b1->rows, params->b1->cols);
    a->v_W2 = matrix_create(params->W2->rows, params->W2->cols);
    a->v_b2 = matrix_create(params->b2->rows, params->b2->cols);
    a->t = 0;
    return a;
}

static void adam_destroy(AdamState* a) {
    matrix_destroy(a->m_W1);
    matrix_destroy(a->m_b1);
    matrix_destroy(a->m_W2);
    matrix_destroy(a->m_b2);
    matrix_destroy(a->v_W1);
    matrix_destroy(a->v_b1);
    matrix_destroy(a->v_W2);
    matrix_destroy(a->v_b2);
    my_free(a);
}

static inline void apply_adam_step(Matrix* p, Matrix* m, Matrix* v, const Matrix* g,
                                   const float b1, const float b2,
                                   const float bc1, const float bc2,
                                   const float lr, const float eps) {
    const int n = p->rows * p->cols;
    for (int i = 0; i < n; i++) {
        const float gi = g->data[i];
        m->data[i] = b1 * m->data[i] + (1.0f - b1) * gi;
        v->data[i] = b2 * v->data[i] + (1.0f - b2) * gi * gi;
        const float m_hat = m->data[i] / bc1;
        const float v_hat = v->data[i] / bc2;
        p->data[i] -= lr * m_hat / (sqrtf(v_hat) + eps);
    }
}

static void update_params(Parameters* params, const BackwardParameters* back_params, AdamState* a, const float lr) {
    const float b1 = 0.9f;
    const float b2 = 0.999f;
    const float eps = 1e-8f;
    a->t += 1;
    const float bc1 = 1.0f - powf(b1, (float)a->t);
    const float bc2 = 1.0f - powf(b2, (float)a->t);
    apply_adam_step(params->W1, a->m_W1, a->v_W1, back_params->W1, b1, b2, bc1, bc2, lr, eps);
    apply_adam_step(params->b1, a->m_b1, a->v_b1, back_params->b1, b1, b2, bc1, bc2, lr, eps);
    apply_adam_step(params->W2, a->m_W2, a->v_W2, back_params->W2, b1, b2, bc1, bc2, lr, eps);
    apply_adam_step(params->b2, a->m_b2, a->v_b2, back_params->b2, b1, b2, bc1, bc2, lr, eps);
}

float* prediction(const Matrix *a2) {
    float* out = my_malloc(a2->cols * sizeof(float));
    for (int x = 0; x < a2->cols; x++) {
        float max = -INFINITY;
        int idx = 0;
        for (int y = 0; y < a2->rows; y++) {
            const float v = a2->data[y * a2->cols + x];
            if (v > max) { max = v; idx = y; }
        }
        out[x] = (float)idx * SCORE_STEP;
    }
    return out;
}

float mse(const float* predictions, const float* groundTruth, const int n) {
    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        const float d = predictions[i] - groundTruth[i];
        sum += d * d;
    }
    return sum / (float)n;
}

void backward_parameters_destroy(BackwardParameters* params) {
    matrix_destroy(params->W1);
    matrix_destroy(params->W2);
    matrix_destroy(params->b1);
    matrix_destroy(params->b2);
    my_free(params);
}

Parameters* gradient_descent(const Matrix* X, const float* Y, const int N, const float lr, const int epochs) {
    Parameters* params = init_params();
    AdamState* a = adam_init(params);

    const int batch_size = 1024;

    for (int i = 0; i < epochs; i++) {
        Batch* batch = create_mini_batch(X, Y, N, batch_size);

        // Learning rate decay
        float current_lr = lr / (1.0f + 0.001f * (float)i);

        Parameters* forward_params = forward(params, batch->X);
        BackwardParameters* backward_params = backprop(forward_params, params->W2, batch->X, batch->Y, batch->size);
        update_params(params, backward_params, a, current_lr);
        if (i % 50 == 0 || i == epochs - 1) {
            printf("Epoch %d\n", i);
            float* pred = prediction(forward_params->b2);
            printf("MSE: %f\n", mse(pred, batch->Y, batch->size));
            my_free(pred);
        }

        backward_parameters_destroy(backward_params);
        params_destroy(forward_params);
        batch_destroy(batch);
    }

    adam_destroy(a);
    return params;
}

void eval(const Matrix* X, const float* Y, const int N, const Parameters* input) {
    Parameters* params = forward(input, X);
    float* pred = prediction(params->b2);
    printf("Eval MSE: %f\n", mse(pred, Y, N));
    my_free(pred);
    params_destroy(params);
}