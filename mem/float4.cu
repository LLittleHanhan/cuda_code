#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <stdlib.h>

void init(float* a, int N) {
    for (int i = 0; i < N; i++) {
        a[i] = 1;
    }
}
__global__ void readglobal(float* da, float* db, int N) {
    int x = threadIdx.x;
    reinterpret_cast<float2*>(db)[x] = reinterpret_cast<float2*>(da)[x];
}

int main(int argc, char** argv) {
    int dev = 0;
    cudaSetDevice(dev);
    const int N = 128;
    float* a = new float[N];
    float *da, *db;
    init(a, N);
    cudaMalloc(&da, sizeof(float) * N);
    cudaMalloc(&db, sizeof(float) * N);
    cudaMemcpy(da, a, sizeof(float) * N, cudaMemcpyHostToDevice);

    readglobal<<<1, 32>>>(da, db, N);

    cudaMemcpy(a, da, sizeof(float) * N, cudaMemcpyDeviceToHost);
}