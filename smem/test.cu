#include <cuda_runtime.h>
#include <stdio.h>
#include <cuda_fp16.h>

void init(int* a, int N) {
    for (int i = 0; i < N; i++) {
        a[i] = 1;
    }
}

__global__ void test(int* da) {
    __shared__ int2 s[32];
    int i = threadIdx.x;
    int2 a = s[i];
    da[2 * i] = *reinterpret_cast<int*>(&a);
    da[2 * i + 1] = *(reinterpret_cast<int*>(&a)+1);
}

int main() {
    const int N = 1024;

    int* a = new int[N];
    int* da;
    init(a, N);
    cudaMalloc(&da, sizeof(int) * N);
    cudaMemcpy(da, a, sizeof(int) * N, cudaMemcpyHostToDevice);
    test<<<1, 32>>>(da);
    cudaDeviceSynchronize();
    for(int i =0;i<32;i++){
        // printf("%d ",((i ^ 0x2F) * 3) & 0x7f);
        // printf("%d ",i * 4 + (i / 16));
    }
    printf("\n");
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(err));
    }
}