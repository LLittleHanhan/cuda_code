#include <cuda_runtime.h>
#include <iostream>
#include "Profile.h"
using namespace std;

const int32_t thread_num = 256;
const int32_t block_num = 800;
const int32_t iter = 1;
double Profiler::time = 0;
__global__ void relu_kernel(float* input, float* output) {
    int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    output[idx] = input[idx] < 0 ? 0 : input[idx];
}

int main() {
    float* input;
    float* output;
    int32_t elem_cnt = thread_num * block_num;
    cudaMalloc(&input, sizeof(float) * elem_cnt);
    cudaMalloc(&output, sizeof(float) * elem_cnt);
    {
        Profiler p("test");
        for(int i =0;i<iter;i++){
            relu_kernel<<<block_num, thread_num>>>(input, output);
        }
    }
    printf("time is %.3f ms\n",Profiler::time / iter);
    cudaDeviceSynchronize();
    float* host = new float[elem_cnt];
    cudaMemcpy(host, output, sizeof(float) * elem_cnt, cudaMemcpyDeviceToHost);
    cudaFree(input);
    cudaFree(output);
    return 0;
}