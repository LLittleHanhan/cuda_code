#include <cuda_runtime.h>
#include <stdio.h>

void init(int* a, int N) {
    for (int i = 0; i < N; i++) {
        a[i] = 1;
    }
}

__global__ void test(int* da, int* dr, int N) {
    int tid = (blockDim.x * blockIdx.x + threadIdx.x) * 16;
    for (; tid < N; tid += blockDim.x * gridDim.x * 16)
        int a = da[tid];
}

int main() {
    const int N = 1024 * 1024 * 1024;

    int* a = new int[N];
    int* da;
    init(a, N);
    cudaMalloc(&da, sizeof(int) * N);
    cudaMemcpy(da, a, sizeof(int) * N, cudaMemcpyHostToDevice);

    int* r = new int[N];
    int* dr;
    cudaMalloc(&dr, sizeof(int) * N);

    test<<<1024, 1024>>>(da, dr, N);
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(err));
    }
}