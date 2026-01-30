#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <chrono>

#define TILE_WIDTH 16

// ------------------- CPU Matrix Multiply -------------------
void matrixMultiplyCPU(float *A, float *B, float *C, int N) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < N; k++) {
                sum += A[i * N + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

// ------------------- Naïve CUDA Kernel -------------------
__global__ void matrixMultiplyNaive(float *A, float *B, float *C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < N && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < N; k++) {
            sum += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

// ------------------- Tiled CUDA Kernel -------------------
__global__ void matrixMultiplyTiled(float *A, float *B, float *C, int N) {
    __shared__ float ds_A[TILE_WIDTH][TILE_WIDTH];
    __shared__ float ds_B[TILE_WIDTH][TILE_WIDTH];

    int bx = blockIdx.x; int by = blockIdx.y;
    int tx = threadIdx.x; int ty = threadIdx.y;
    int Row = by * TILE_WIDTH + ty;
    int Col = bx * TILE_WIDTH + tx;

    float Pvalue = 0.0;

    for (int m = 0; m < (N + TILE_WIDTH - 1) / TILE_WIDTH; ++m) {
        if (Row < N && (m*TILE_WIDTH+tx) < N)
            ds_A[ty][tx] = A[Row * N + m * TILE_WIDTH + tx];
        else
            ds_A[ty][tx] = 0.0f;

        if (Col < N && (m*TILE_WIDTH+ty) < N)
            ds_B[ty][tx] = B[(m*TILE_WIDTH + ty) * N + Col];
        else
            ds_B[ty][tx] = 0.0f;

        __syncthreads();

        for (int k = 0; k < TILE_WIDTH; ++k)
            Pvalue += ds_A[ty][k] * ds_B[k][tx];

        __syncthreads();
    }

    if (Row < N && Col < N)
        C[Row * N + Col] = Pvalue;
}

// ------------------- Benchmark Function -------------------
void runBenchmark(int N) {
    size_t size = N * N * sizeof(float);

    float *h_A = (float*)malloc(size);
    float *h_B = (float*)malloc(size);
    float *h_C = (float*)malloc(size);
    float *h_ref = (float*)malloc(size);

    for (int i = 0; i < N*N; i++) {
        h_A[i] = 1.0f;
        h_B[i] = 1.0f;
    }

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    dim3 threads(TILE_WIDTH, TILE_WIDTH);
    dim3 blocks((N + TILE_WIDTH - 1) / TILE_WIDTH, (N + TILE_WIDTH - 1) / TILE_WIDTH);

    // --- CPU timing ---
    auto start_cpu = std::chrono::high_resolution_clock::now();
    matrixMultiplyCPU(h_A, h_B, h_ref, N);
    auto stop_cpu = std::chrono::high_resolution_clock::now();
    double cpu_time = std::chrono::duration<double>(stop_cpu - start_cpu).count();

    // --- Naïve GPU ---
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    matrixMultiplyNaive<<<blocks, threads>>>(d_A, d_B, d_C, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float naive_time = 0;
    cudaEventElapsedTime(&naive_time, start, stop);

    // --- Tiled GPU ---
    cudaEventRecord(start);
    matrixMultiplyTiled<<<blocks, threads>>>(d_A, d_B, d_C, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float tiled_time = 0;
    cudaEventElapsedTime(&tiled_time, start, stop);

    // --- Print CSV row ---
    printf("%d,%f,%f,%f,%f,%f\n", 
           N, cpu_time, naive_time/1000.0, tiled_time/1000.0,
           cpu_time/(naive_time/1000.0), cpu_time/(tiled_time/1000.0));

    free(h_A); free(h_B); free(h_C); free(h_ref);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
}

// ------------------- Main -------------------
int main() {
    // CSV Header
    printf("N,CPU(s),Naive_GPU(s),Tiled_GPU(s),Speedup_Naive,Speedup_Tiled\n");

    int sizes[] = {256, 512, 1024, 2048};
    for (int i = 0; i < 5; i++) {
        runBenchmark(sizes[i]);
    }

    return 0;
}
