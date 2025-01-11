#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <stdlib.h>
using namespace std;

__global__ void kernel_IADD3(int* data) {
    int a[2], b[2], c[2], d[2], e[2], f[2];
    a[0] = data[0];
    a[1] = data[1];
    b[0] = data[2];
    b[1] = data[3];
    c[0] = data[4];
    c[1] = data[5];
    d[0] = data[6];
    d[1] = data[7];
    e[0] = data[8];
    e[1] = data[9];
    f[0] = data[10];
    f[1] = data[11];

    for (int i = 0; i < 200000; i++) {
        for (int j = 0; j < 2; j++) {
            a[j] = a[j] * b[(j + 1) % 2] + c[(j + 2) % 2];
            d[j] = d[j] + e[(j + 1) % 2] + f[(j + 2) % 2];
        }
        for (int j = 0; j < 2; j++) {
            b[j] = b[j] * c[(j + 1) % 2] + a[(j + 2) % 2];
            e[j] = e[j] + f[(j + 1) % 2] + d[(j + 2) % 2];
        }
        for (int j = 0; j < 2; j++) {
            c[j] = c[j] * a[(j + 1) % 2] + b[(j + 2) % 2];
            f[j] = f[j] + d[(j + 1) % 2] + e[(j + 2) % 2];
        }
    }
    data[0] = a[0];
    data[1] = a[1];
    data[2] = b[0];
    data[3] = b[1];
    data[4] = c[0];
    data[5] = c[1];
    data[6] = d[0];
    data[7] = d[1];
    data[8] = e[0];
    data[9] = e[1];
    data[10] = f[0];
    data[11] = f[1];
}

int main() {
    int dev = 1;
    cudaSetDevice(dev);
    int thread_num_per_block = 32;
    int block_num = 1;
    int data[12]{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
    int* d_data;
    cudaMalloc(&d_data, sizeof(int) * 12);
    cudaMemcpy(d_data, &data, sizeof(int) * 12, cudaMemcpyHostToDevice);
    dim3 block(thread_num_per_block);
    dim3 grid(block_num);
    kernel_IADD3<<<grid, block>>>(d_data);
    cudaDeviceSynchronize();
    cudaMemcpy(&data, d_data, sizeof(int) * 12, cudaMemcpyDeviceToHost);
    for (int i = 0; i < 12; i++) {
        printf("%d ", data[i]);
    }
}