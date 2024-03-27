#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cstdio>
#include <cstring>
#include <iostream>
__global__ void relu_kernel(float* input, float* output){
  int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  output[idx] = input[idx] < 0 ? 0 : input[idx];
}

int main(){
  float* input;
  float* output;
  int32_t elem_cnt = 3*224*224;
  
  cudaMalloc(&input, sizeof(float)*elem_cnt);
  cudaMalloc(&output, sizeof(float)*elem_cnt);
  int32_t thread_num = 256;
  int32_t grid_size = (elem_cnt + thread_num -1) / thread_num;
  relu_kernel<<<grid_size, thread_num>>>(input, output); 
  
  cudaDeviceSynchronize();
  cudaFree(input);
  cudaFree(output);
  return 0;
}