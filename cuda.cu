__global__ void matrixMultiplyKernel(double *A, double *B, double *C, int N, int M, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < K) {
        double value = 0;

        for (int i = 0; i < M; ++i) {
            value += A[row * M + i] * B[i * K + col];
        }

        C[row * K + col] = value;
    }
}

extern "C" void cuda_matmul(
    int row_a, int col_a, double* data_a, 
    int row_b, int col_b, double* data_b, 
    int row_c, int col_c, double* data_c
) {
    double *d_A, *d_B, *d_C;

    size_t size_A = row_a * col_a * sizeof(double);
    size_t size_B = row_b * col_b * sizeof(double);
    size_t size_C = row_a * col_b * sizeof(double);

    cudaMalloc((void**)&d_A, size_A);
    cudaMalloc((void**)&d_B, size_B);
    cudaMalloc((void**)&d_C, size_C);

    cudaMemcpy(d_A, data_a, size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, data_b, size_B, cudaMemcpyHostToDevice);

    dim3 dimBlock(16, 16);
    dim3 dimGrid((col_b + dimBlock.x - 1) / dimBlock.x, (row_a + dimBlock.y - 1) / dimBlock.y);

    matrixMultiplyKernel<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, row_a, col_a, col_b);

    cudaMemcpy(data_c, d_C, size_C, cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}
