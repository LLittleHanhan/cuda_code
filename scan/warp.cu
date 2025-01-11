#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdlib.h>
#include <cub/cub.cuh>
#include <iostream>

using namespace std;

const int warpsize = 32;
const int thread_num_per_block = 128;
const int block_num = 1;
const int N = thread_num_per_block * block_num;

void initval(float s[], int n) {
    srand((unsigned)time(NULL));
    for (int i = 0; i < n; i++) {
        // s[i] = float(rand() % 10) / 100000;
        s[i] = 1;
    }
}

__global__ void baseline(float* input) {
    // Specialize WarpScan for type int
    typedef cub::WarpScan<int> WarpScan;

    // Allocate WarpScan shared memory for 4 warps
    __shared__ typename WarpScan::TempStorage temp_storage[4];
    // Obtain one input item per thread
    int tid = threadIdx.x;
    int thread_data = input[blockIdx.x * blockDim.x + tid];

    // Compute inclusive warp-wide prefix sums
    int warp_id = threadIdx.x / 32;
    WarpScan(temp_storage[warp_id]).InclusiveSum(thread_data, thread_data);

    input[blockIdx.x * blockDim.x + tid] = thread_data;
    return;
}

__global__ void scan(float* input) {
    int tid = threadIdx.x;
    int warpLane = tid % warpsize;
    float thread_data = input[blockIdx.x * blockDim.x + tid];

    float temp;

    temp = __shfl_up_sync(0xffffffff, thread_data, 1);
    if (warpLane >= 1)
        thread_data += temp;
    temp = __shfl_up_sync(0xffffffff, thread_data, 2);
    if (warpLane >= 2)
        thread_data += temp;
    temp = __shfl_up_sync(0xffffffff, thread_data, 4);
    if (warpLane >= 4)
        thread_data += temp;
    temp = __shfl_up_sync(0xffffffff, thread_data, 8);
    if (warpLane >= 8)
        thread_data += temp;
    temp = __shfl_up_sync(0xffffffff, thread_data, 16);
    if (warpLane >= 16)
        thread_data += temp;

    input[blockIdx.x * blockDim.x + tid] = thread_data;
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
        baseline<<<grid, block>>>(input);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&msecTotal, start, stop);

    cudaMemcpy(res, input, sizeof(float) * N, cudaMemcpyDeviceToHost);

    cout << "warp time:" << msecTotal / iter << "ms" << endl;

    // for (int i = 0; i < N; i++) {
    //     cout << res[i] << endl;
    // }
    cudaFree(input);
    delete[] a;
    delete[] res;
}