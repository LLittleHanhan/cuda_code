#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cstdio>
#include <cstring>
#include <iostream>
using namespace std;

__device__ int ga[64];



__global__ void test() {
   ga[threadIdx.x]+=1;
}

int main(int argc, char** argv) {
    int dev = 0;
    cudaSetDevice(dev);
    cout<<sizeof(char);
    test<<<1, 24>>>();
}