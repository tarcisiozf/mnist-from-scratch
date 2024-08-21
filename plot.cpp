#include <stdio.h>
#include <stdlib.h>
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

void plot_image(const char* filename, double* matrix) {
    unsigned char *img = (unsigned char *)malloc(28 * 28 * sizeof(unsigned char));

    // Find the min and max values in the matrix for normalization
    double min = matrix[0];
    double max = matrix[0];
    double val;
    for (int i = 0; i < 28; i++) {
        for (int j = 0; j < 28; j++) {
            val = matrix[i * 28 + j];
            if (val < min) {
                min = val;
            }
            if (val > max) {
                max = val;
            }
        }
    }

    // Normalize the matrix and fill the image buffer
    for (int i = 0; i < 28; i++) {
        for (int j = 0; j < 28; j++) {
            float normalized = (matrix[i * 28 + j] - min) / (max - min) * 255.0f;
            img[i * 28 + j] = (unsigned char)normalized;
        }
    }

    if (!stbi_write_png(filename, 28, 28, 1, img, 28)) {
        fprintf(stderr, "Failed to write image.\n");
        free(img);
    }

    free(img);
}