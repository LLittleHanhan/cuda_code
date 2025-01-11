#include <cuda_runtime.h>
#include <stdio.h>

constexpr int warp = 32;
constexpr int len = 256 * 4 * 128;
constexpr int repeat = 4;
constexpr int threads = 256;
constexpr int blocks = len / (threads * repeat);
constexpr int warps = threads / warp;

__global__ void reduceSum(float* input, float* output) {
    int bid = blockIdx.x;
    int tid = threadIdx.x;
    int warpId = tid / warp;
    int warplane = tid % warp;

    __shared__ float s[warps];
    float temp[4];
    *reinterpret_cast<float4*>(&temp) = *reinterpret_cast<float4*>(&input[bid * threads * repeat + tid * repeat]);
    float a = 0;
#pragma unroll
    for (int i = 0; i < repeat; i++) {
        a += temp[i];
    }

    a += __shfl_down_sync(0xffffffff, a, 16);
    a += __shfl_down_sync(0xffffffff, a, 8);
    a += __shfl_down_sync(0xffffffff, a, 4);
    a += __shfl_down_sync(0xffffffff, a, 2);
    a += __shfl_down_sync(0xffffffff, a, 1);

    if (warplane == 0) {
        s[warpId] = a;
    }
    __syncthreads();
    if (warpId == 0 && warplane < warps) {
        if (warplane < warps)
            a = s[warplane];
        else
            a = 0;
#pragma unroll
        for (int i = warps; i >= 1; i /= 2) {
            a += __shfl_down_sync(0xffffffff, a, i);
        }
        if (warplane == 0) {
            output[blockIdx.x] = a;
        }
    }
}

void init(float* a, int len) {
    for (int i = 0; i < len; i++) {
        a[i] = 1;
    }
}

int main() {
    float* input = new float[len];
    float* output = new float[blocks];
    init(input, len);
    init(output, blocks);

    float* dinput;
    float* doutput;
    cudaMalloc(&dinput, sizeof(float) * len);
    cudaMalloc(&doutput, sizeof(float) * blocks);
    cudaMemcpy(dinput, input, sizeof(float) * len, cudaMemcpyHostToDevice);

    reduceSum<<<blocks, threads>>>(dinput, doutput);

    cudaMemcpy(output, doutput, sizeof(float) * blocks, cudaMemcpyDeviceToHost);
    // for (int i = 0; i < blocks; i++) {
    //     printf("%f ", output[i]);
    // }
    // printf("\n");
}