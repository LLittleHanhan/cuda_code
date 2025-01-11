#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <stdlib.h>
#define abs(x) ((x) >= 0.0 ? (x) : -(x))

void initmatrix(float* s, int M, int N) {
    srand((unsigned)time(NULL));
    for (int i = 0; i < M * N; i++) {
        // s[i * N + j] = float(rand() % 100) / 100;
        s[i] = i;
    }
    return;
}

void check_result(float* a, float* b, float* c, int M, int K, int N) {
    float error = 0;
    float temp = 0;
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            temp = 0;
            for (int k = 0; k < K; k++) {
                temp += a[i * K + k] * b[k * N + j];
            }
            if (abs(temp - c[i * N + j]) > error) {
                error = abs(temp - c[i * N + j]);
            }
            // error += abs(temp - c[i * N + j]);
        }
    }
    printf("error is %.8f\n", error);
    return;
}

/*
argv[1] = M
argv[2] = N
argv[3] = K
*/
int main(int argc, char** argv) {
    int dev = 3;
    cudaSetDevice(dev);

    const int M = 1024;
    const int K = 1024;
    const int N = 1024;

    float* a = new float[M * K];
    float* b = new float[K * N];
    float* c = new float[M * N];
    initmatrix(a, M, K);
    initmatrix(b, K, N);

    float *da, *db, *dc;
    cudaMalloc(&da, sizeof(float) * M * K);
    cudaMalloc(&db, sizeof(float) * K * N);
    cudaMalloc(&dc, sizeof(float) * M * N);
    cudaMemcpy(da, a, sizeof(float) * M * K, cudaMemcpyHostToDevice);
    cudaMemcpy(db, b, sizeof(float) * K * N, cudaMemcpyHostToDevice);

    cublasHandle_t handle;
    cublasCreate(&handle);
    float alpha = 1;
    float beta = 0;
    cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_T, M, N, K, &alpha, da, K, db, N, &beta, dc, M);
    cudaMemcpy(c, dc, sizeof(float) * M * N, cudaMemcpyDeviceToHost);
    cublasDestroy(handle);
    //check_result(a, b, c, M, K, N);

    cudaFree(da);
    cudaFree(db);
    cudaFree(dc);
    delete[] a;
    delete[] b;
    delete[] c;
    return 0;
}