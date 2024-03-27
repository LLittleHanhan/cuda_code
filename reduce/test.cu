#include <stdio.h>
#include "cuda_runtime.h"

__global__ void kernel(double *a)
{
    double v = a[threadIdx.x];

    unsigned mask = 0x00000000; // 0xffffffff;//0x000000ff;
    unsigned int offset = 4;
    v = __shfl_down_sync(mask, v, offset, 32);

    a[threadIdx.x] = v;
}

void main()
{
    double *a, *a_d;
    a = (double *)calloc(32, sizeof(double));
    cudaMalloc((void **)&a_d, 32 * sizeof(double));
    for (int i = 0; i < 32; i++)
    {
        a[i] = i / 4;
    }
    cudaMemcpy(a_d, a, 32 * sizeof(double), cudaMemcpyHostToDevice);

    for (int i = 0; i < 32; i++)
    {
        printf("%2.0f ", a[i]);
    }
    printf("\n");

    kernel<<<1, 32>>>(a_d);

    cudaMemcpy(a, a_d, 32 * sizeof(double), cudaMemcpyDeviceToHost);
    for (int i = 0; i < 32; i++)
    {
        printf("%2.0f ", a[i]);
    }
}