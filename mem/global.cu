#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <stdlib.h>
#define CHECK(call)\
{\
  const cudaError_t error=call;\
  if(error!=cudaSuccess)\
  {\
      printf("ERROR: %s:%d,",__FILE__,__LINE__);\
      printf("code:%d,reason:%s\n",error,cudaGetErrorString(error));\
      exit(1);\
  }\
}
void init(int* a, int N) {
    for (int i = 0; i < N; i++) {
        a[i] = 1;
    }
}
__global__ void readglobal(int* da, int N) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    da[y * N + x] += 1;
    // da[x * N + y] += 1;
}

int main(int argc,char** argv) {
    int dev = 0;
    cudaSetDevice(dev);
    const int N = 1024;
    int* a = new int[N * N];
    int* da;
    init(a, N * N);
    cudaMalloc(&da, sizeof(int) * N * N);
    cudaMemcpy(da, a, sizeof(int) * N * N, cudaMemcpyHostToDevice);

    int tx = atoi(argv[1]);
    int ty = atoi(argv[2]);
    dim3 block(tx, ty);
    dim3 grid(N/tx, N/ty);

    cudaEvent_t start, stop;
    float time;
    int iter = 100;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    for (int i = 0; i < iter; i++) {
        readglobal<<<grid, block>>>(da, N);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    printf("time is %.3f\n",time/iter);
    printf("global memery max bw = %.3f GB/s\n", float(N * N * 2) * 1e-9 / (time / (iter * 1000)));

    cudaMemcpy(a, da, sizeof(int) * N * N, cudaMemcpyDeviceToHost);
}