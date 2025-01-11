/*
shuffle指令
*/
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdlib.h>
#include <iostream>
using namespace std;
const int m = 2;
const int N = 256 * 640 * 2;
const int thread_num_per_block = 256;
const int block_num = N / (thread_num_per_block * m);

void initval(float s[], int n) {
    srand((unsigned)time(NULL));
    for (int i = 0; i < n; i++) {
        s[i] = float(rand() % 10) / 100000;
        // s[i] = 1.0;
    }
}

__global__ void reduce(float* input, float* output) {
    __shared__ float array[thread_num_per_block / 32];

    int tid = threadIdx.x;
    int id = blockIdx.x * blockDim.x * m + threadIdx.x;
    float sum = 0;
    for (int i = 0; i < m; i++)
        sum += input[id + blockDim.x * i];
    // warp shuffle

    int warpId = threadIdx.x / 32;
    int laneId = threadIdx.x % 32;

    sum += __shfl_down_sync(0Xffffffff, sum, 16);
    sum += __shfl_down_sync(0Xffffffff, sum, 8);
    sum += __shfl_down_sync(0Xffffffff, sum, 4);
    sum += __shfl_down_sync(0Xffffffff, sum, 2);
    sum += __shfl_down_sync(0Xffffffff, sum, 1);
    if (laneId == 0) {
        array[warpId] = sum;
    }
    __syncthreads();

    if (warpId == 0) {
        sum = (tid < thread_num_per_block / 32) ? array[tid] : 0;
        sum += __shfl_down_sync(0Xffffffff, sum, 16);
        sum += __shfl_down_sync(0Xffffffff, sum, 8);
        sum += __shfl_down_sync(0Xffffffff, sum, 4);
        sum += __shfl_down_sync(0Xffffffff, sum, 2);
        sum += __shfl_down_sync(0Xffffffff, sum, 1);
    }
    if (tid == 0) {
        output[blockIdx.x] = sum;
    }
    return;
}

int main() {
    int dev = 3;
    cudaSetDevice(dev);

    float* a = new float[N];
    float* res = new float[block_num];
    initval(a, N);

    float *input, *output;
    cudaMalloc(&input, sizeof(float) * N);
    cudaMalloc(&output, sizeof(float) * block_num);
    cudaMemcpy(input, a, sizeof(float) * N, cudaMemcpyHostToDevice);
    dim3 block(thread_num_per_block);
    dim3 grid(block_num);
    reduce<<<grid, block>>>(input, output);

    int iter = 5;
    float msecTotal;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    for (int i = 0; i < iter; i++) {
        reduce<<<grid, block>>>(input, output);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&msecTotal, start, stop);

    cudaMemcpy(res, output, sizeof(float) * block_num, cudaMemcpyDeviceToHost);

    cout << "time:" << msecTotal / iter << "ms" << endl;

    // float sum = 0.0;
    // for (int i = 0; i < block_num; i++)
    //     sum += res[i];
    // cout << sum << endl;

    // float hsum = 0.0;
    // for (int i = 0; i < N; i++) {
    //     hsum += a[i];
    // }
    // cout << hsum << endl;
    cudaFree(input);
    cudaFree(output);
    delete[] a;
    delete[] res;
}