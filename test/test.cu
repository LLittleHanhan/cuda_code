#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <stdlib.h>
using namespace std;

__global__ void baseline(int* a) {
    long long start, stop;
    asm volatile(
        "{\n\t"
        "mov.u64 %0, %%clock64;\n\t"
        "mov.s32 %1, 1;\n\t"
        "mov.u64 %2, %%clock64;\n\t"
        "}"
        : "=l"(start), "=r"(*a), "=l"(stop));

    printf("%lld\n", stop - start);
}

int main() {
    int dev = 3;
    cudaSetDevice(dev);
    int thread_num_per_block = 1;
    int block_num = 1;
    int ha = 0;
    int* da;
    cudaMalloc(&da, sizeof(int));
    cudaMemcpy(da, &ha, sizeof(int), cudaMemcpyHostToDevice);
    dim3 block(thread_num_per_block);
    dim3 grid(block_num);
    baseline<<<grid, block>>>(da);
    //cudaDeviceSynchronize();
    cudaMemcpy(&ha, da, sizeof(int), cudaMemcpyDeviceToHost);
    printf("%d", ha);
}