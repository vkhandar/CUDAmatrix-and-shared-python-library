
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <chrono>

#define TILE_WIDTH 16

// Simple CUDA error-check macros
#define CHECK_CUDA(call)                                                         \
    do {                                                                         \
        cudaError_t err = (call);                                                \
        if (err != cudaSuccess) {                                                \
            fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,        \
                    cudaGetErrorString(err));                                    \
            exit(EXIT_FAILURE);                                                  \
        }                                                                        \
    } while (0)

#define CHECK_CUBLAS(call)                                                       \
    do {                                                                         \
        cublasStatus_t st = (call);                                              \
        if (st != CUBLAS_STATUS_SUCCESS) {                                       \
            fprintf(stderr, "cuBLAS error %d at %s:%d\n", st, __FILE__, __LINE__); \
            exit(EXIT_FAILURE);                                                  \
        }                                                                        \
    } while (0)

// ---------------- CPU reference) ----------------
void matrixMultiplyCPU(float *A, float *B, float *C, int N) {
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j) {
            float s = 0.0f;
            for (int k = 0; k < N; ++k) s += A[i * N + k] * B[k * N + j];
            C[i * N + j] = s;
        }
}

// ---------------- Naive GPU kernel  ----------------
__global__ void matrixMultiplyNaive(float *A, float *B, float *C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < N && col < N) {
        float sum = 0.f;
        for (int k = 0; k < N; ++k) sum += A[row * N + k] * B[k * N + col];
        C[row * N + col] = sum;
    }
}

// ---------------- Tiled GPU kernel ----------------
__global__ void matrixMultiplyTiled(float *A, float *B, float *C, int N) {
    __shared__ float ds_A[TILE_WIDTH][TILE_WIDTH];
    __shared__ float ds_B[TILE_WIDTH][TILE_WIDTH];

    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;
    int Row = by * TILE_WIDTH + ty;
    int Col = bx * TILE_WIDTH + tx;

    float Pvalue = 0.0f;
    int numTiles = (N + TILE_WIDTH - 1) / TILE_WIDTH;

    for (int m = 0; m < numTiles; ++m) {
        int a_col = m * TILE_WIDTH + tx;
        int b_row = m * TILE_WIDTH + ty;

        if (Row < N && a_col < N) ds_A[ty][tx] = A[Row * N + a_col];
        else ds_A[ty][tx] = 0.0f;

        if (b_row < N && Col < N) ds_B[ty][tx] = B[b_row * N + Col];
        else ds_B[ty][tx] = 0.0f;

        __syncthreads();

        for (int k = 0; k < TILE_WIDTH; ++k) Pvalue += ds_A[ty][k] * ds_B[k][tx];
        __syncthreads();
    }

    if (Row < N && Col < N) C[Row * N + Col] = Pvalue;
}

// ----------------  (cuBLAS) ----------------
void runBenchmark(int N, cublasHandle_t handle, int max_cpu_N) {
    size_t bytes = (size_t)N * N * sizeof(float);

    // allocate host
    float *h_A = (float*)malloc(bytes);
    float *h_B = (float*)malloc(bytes);
    float *h_C = (float*)malloc(bytes);    // GPU result
    float *h_ref = (float*)malloc(bytes);  // CPU reference (optional)

    if (!h_A || !h_B || !h_C || !h_ref) {
        fprintf(stderr, "Host allocation failed for N=%d\n", N);
        exit(EXIT_FAILURE);
    }

    for (size_t i = 0; i < (size_t)N * N; ++i) { h_A[i] = 1.0f; h_B[i] = 1.0f; }

    // device alloc
    float *d_A, *d_B, *d_C;
    CHECK_CUDA(cudaMalloc(&d_A, bytes));
    CHECK_CUDA(cudaMalloc(&d_B, bytes));
    CHECK_CUDA(cudaMalloc(&d_C, bytes));

    // copy
    CHECK_CUDA(cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemset(d_C, 0, bytes));

    dim3 threads(TILE_WIDTH, TILE_WIDTH);
    dim3 blocks((N + TILE_WIDTH - 1) / TILE_WIDTH, (N + TILE_WIDTH - 1) / TILE_WIDTH);

    // CPU time 
    double cpu_time = -1.0;
    if (N <= max_cpu_N) {
        auto t0 = std::chrono::high_resolution_clock::now();
        matrixMultiplyCPU(h_A, h_B, h_ref, N);
        auto t1 = std::chrono::high_resolution_clock::now();
        cpu_time = std::chrono::duration<double>(t1 - t0).count();
    }

    // Naive GPU timing (single launch)
    cudaEvent_t s, e;
    CHECK_CUDA(cudaEventCreate(&s));
    CHECK_CUDA(cudaEventCreate(&e));




    CHECK_CUDA(cudaEventRecord(s));
    matrixMultiplyNaive<<<blocks, threads>>>(d_A, d_B, d_C, N);
    CHECK_CUDA(cudaEventRecord(e));
    CHECK_CUDA(cudaEventSynchronize(e));
    float naive_ms = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&naive_ms, s, e));



    // Tiled GPU timing
    CHECK_CUDA(cudaEventRecord(s));
    matrixMultiplyTiled<<<blocks, threads>>>(d_A, d_B, d_C, N);
    CHECK_CUDA(cudaEventRecord(e));
    CHECK_CUDA(cudaEventSynchronize(e));
    float tiled_ms = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&tiled_ms, s, e));

    // cuBLAS timing 
	const float alpha = 1.0f;
    const float beta = 0.0f;


    CHECK_CUBLAS(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha, d_B, N, d_A, N, &beta, d_C, N));
    CHECK_CUDA(cudaDeviceSynchronize());

    int repeats = 3;
    float cublas_ms_total = 0.0f;
    for (int r = 0; r < repeats; ++r) {
        CHECK_CUDA(cudaEventRecord(s));
        CHECK_CUBLAS(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha, d_B, N, d_A, N, &beta, d_C, N));
        CHECK_CUDA(cudaEventRecord(e));
        CHECK_CUDA(cudaEventSynchronize(e));
        float iter_ms = 0.0f;
        CHECK_CUDA(cudaEventElapsedTime(&iter_ms, s, e));
        cublas_ms_total += iter_ms;
    }
    float cublas_ms = cublas_ms_total / repeats;

    // Print CSV row ms to s
    double naive_s = naive_ms / 1000.0;
    double tiled_s = tiled_ms / 1000.0;
    double cublas_s = cublas_ms / 1000.0;

    if (cpu_time < 0) {
        // CPU skipped
        printf("%d,NA,%.6f,%.6f,%.6f,NA,NA,NA\n", N, naive_s, tiled_s, cublas_s);
    } else {
        printf("%d,%.6f,%.6f,%.6f,%.6f,%.2f,%.2f,%.2f\n", N, cpu_time, naive_s, tiled_s, cublas_s, cpu_time / naive_s, cpu_time / tiled_s, cpu_time / cublas_s);
    }

    // cleanup
    CHECK_CUDA(cudaEventDestroy(s));
    CHECK_CUDA(cudaEventDestroy(e));
    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_B));
    CHECK_CUDA(cudaFree(d_C));
    free(h_A); free(h_B); free(h_C); free(h_ref);
}

int main(int argc, char** argv) {
    CHECK_CUDA(cudaFree(0));
    int max_cpu_N = 1024; 
    printf("N,CPU(s),Naive_GPU(s),Tiled_GPU(s),cuBLAS_GPU(s),Speedup_Naive,Speedup_Tiled,Speedup_cuBLAS\n");

    int sizes[] = {256, 512, 1024, 2048};
    int ns = sizeof(sizes) / sizeof(sizes[0]);

    cublasHandle_t handle;
    CHECK_CUBLAS(cublasCreate(&handle));
    printf("Noooooo!");

    for (int i = 0; i < ns; ++i) {
        runBenchmark(sizes[i], handle, max_cpu_N);
    }

    CHECK_CUBLAS(cublasDestroy(handle));
    return 0;
}
