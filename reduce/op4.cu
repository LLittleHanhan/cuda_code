/*
展开最后一个线程束
*/
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <stdlib.h>
using namespace std;

const int N = 256 * 1024 * 128;
const int m = 2;
const int thread_num_per_block = 256;
const int block_num = N / (thread_num_per_block * m);

void initval(float s[], int n)
{
    srand((unsigned)time(NULL));
    for (int i = 0; i < n; i++)
    {
        s[i] = float(rand() % 10) / 100000;
        // s[i] = 1.0;
    }
}

__global__ void reduce(float *input, float *output)
{
    __shared__ float array[thread_num_per_block];
    int tid = threadIdx.x;
    int id = blockIdx.x * blockDim.x * m + threadIdx.x;
    array[tid] = input[id] + input[id + blockDim.x];
    __syncthreads();

    for (int s = blockDim.x / 2; s > 32; s /= 2)
    {
        if (tid < s)
        {
            array[tid] += array[tid + s];
        }
        __syncthreads();
    }
    if (tid < 32)
    {
        array[tid] += array[tid + 32];
        array[tid] += array[tid + 16];
        array[tid] += array[tid + 8];
        array[tid] += array[tid + 4];
        array[tid] += array[tid + 2];
        array[tid] += array[tid + 1];
    }
    if (tid == 0)
    {
        output[blockIdx.x] = array[0];
    }
}

int main(int argc, char **argv)
{
    int dev = 0;
    cudaSetDevice(dev);

    float *a = new float[N];
    float *res = new float[block_num];
    initval(a, N);

    float *input, *output;
    cudaMalloc(&input, sizeof(float) * N);
    cudaMalloc(&output, sizeof(float) * block_num);
    cudaMemcpy(input, a, sizeof(float) * N, cudaMemcpyHostToDevice);

    dim3 block(thread_num_per_block);
    dim3 grid(block_num);

    reduce<<<grid, block>>>(input, output);

    cudaMemcpy(res, output, sizeof(float) * block_num, cudaMemcpyDeviceToHost);

    float sum = 0.0;
    for (int i = 0; i < block_num; i++)
        sum += res[i];
    cout << sum << endl;

    float hsum = 0.0;
    for (int i = 0; i < N; i++)
    {
        hsum += a[i];
    }
    cout << hsum << endl;

    cudaFree(input);
    cudaFree(output);
    delete[] a;
    delete[] res;
}