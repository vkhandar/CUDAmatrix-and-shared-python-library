#include <cuda_runtime.h>
#include <stdio.h>

#define TILE_WIDTH 16
__global__ void matrixMultiplyTiled(float *A, float *B, float *C, int N) {
    __shared__ float ds_A[TILE_WIDTH][TILE_WIDTH];
    __shared__ float ds_B[TILE_WIDTH][TILE_WIDTH];

    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;
    int Row = by * TILE_WIDTH + ty;
    int Col = bx * TILE_WIDTH + tx;

    float Pvalue = 0.0f;

    for (int m = 0; m < (N + TILE_WIDTH - 1) / TILE_WIDTH; m++) {
        ds_A[ty][tx] = (Row < N && m*TILE_WIDTH + tx < N)
            ? A[Row * N + m*TILE_WIDTH + tx] : 0.0f;

        ds_B[ty][tx] = (Col < N && m*TILE_WIDTH + ty < N)
            ? B[(m*TILE_WIDTH + ty) * N + Col] : 0.0f;

        __syncthreads();

        for (int k = 0; k < TILE_WIDTH; k++)
            Pvalue += ds_A[ty][k] * ds_B[k][tx];

        __syncthreads();
    }

    if (Row < N && Col < N)
        C[Row * N + Col] = Pvalue;
}

extern "C" __declspec(dllexport)
void gpu_matrix_multiply(float *h_A, float *h_B, float *h_C, int N) {
    size_t size = N * N * sizeof(float);

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    dim3 threads(TILE_WIDTH, TILE_WIDTH);
    dim3 blocks((N + TILE_WIDTH - 1) / TILE_WIDTH,
                (N + TILE_WIDTH - 1) / TILE_WIDTH);

    matrixMultiplyTiled<<<blocks, threads>>>(d_A, d_B, d_C, N);
    cudaDeviceSynchronize();

    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

