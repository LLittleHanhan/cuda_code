#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cstdio>
#include <cstring>
#include <iostream>
using namespace std;

#define CHECK(call){\
    const cudaError_t error = call;\
    if(error != cudaSuccess){\
        cout<<cudaGetErrorString(error);\
        exit(1);\
    }\
}
__shared__ int hello[32][33];
__global__ void readrow(int* out){
    int x = threadIdx.x;
    int y = threadIdx.y;
    int idx = threadIdx.y*blockDim.x+threadIdx.x;
    hello[x][y] = idx;
    __syncthreads();
    out[idx]=hello[x][y];
}

int main() {
    int dev = 0;
    cudaSetDevice(dev);
    
    cudaSharedMemConfig conf;
    cudaDeviceGetSharedMemConfig(&conf);
    cout << conf <<endl;//1,four
    //CHECK(cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte));
    //cudaDeviceGetSharedMemConfig(&conf);
    //cout << conf <<endl;//

    dim3 block(32,32);
    dim3 grid(1);

    int* out;
    cudaMalloc(&out,sizeof(int)*32*32);
    readrow<<<grid,block>>>(out);
}