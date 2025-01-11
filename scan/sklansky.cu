#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdlib.h>
#include <iostream>
using namespace std;

const int N = 256;
const int thread_num_per_block = 256;
const int block_num = N / (thread_num_per_block);

void initval(float s[], int n) {
    srand((unsigned)time(NULL));
    for (int i = 0; i < n; i++) {
        // s[i] = float(rand() % 10) / 100000;
        s[i] = 0.001;
    }
}

__global__ void scan(float* input) {
    __shared__ float array[thread_num_per_block];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;
    array[tid] = input[idx];
    __syncthreads();

    for (int i = 1; i <= thread_num_per_block / 2; i *= 2) {
        if(tid<thread_num_per_block/2){
            array[tid + i + tid / i * i] += array[i - 1 + tid / i * 2 * i];
        }
        __syncthreads();
    }
    input[idx] = array[tid];
    return;
}

int main() {
    int dev = 3;
    cudaSetDevice(dev);

    float* a = new float[N];
    float* res = new float[N];
    initval(a, N);

    float* input;
    cudaMalloc(&input, sizeof(float) * N);
    cudaMemcpy(input, a, sizeof(float) * N, cudaMemcpyHostToDevice);
    dim3 block(thread_num_per_block);
    dim3 grid(block_num);

    int iter = 1;
    float msecTotal;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    for (int i = 0; i < iter; i++) {
        scan<<<grid, block>>>(input);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&msecTotal, start, stop);

    cudaMemcpy(res, input, sizeof(float) * N, cudaMemcpyDeviceToHost);

    cout << "sklansky time:" << msecTotal / iter << "ms" << endl;

    // for (int i = 0; i < thread_num_per_block; i++) {
    //     cout << res[i] << " "<<endl;
    // }
    cudaFree(input);
    delete[] a;
    delete[] res;
}