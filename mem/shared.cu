#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <stdlib.h>

void init(float* a, int N) {
    for (int i = 0; i < N; i++) {
        a[i] = 1;
    }
}
__global__ void readshared() {
    extern __shared__ float array[];
    int idx = threadIdx.x;
    for(int i=0; i<8 ; i++){
        array[i*32+idx] = i;
    }
   
}

int main(int argc, char** argv) {
    int dev = 3;
    cudaSetDevice(dev);
    readshared<<<1, 32, 8* 32 * sizeof(float)>>>();
}