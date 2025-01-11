#include <cuda_runtime.h>

__global__ ChildKernel(void* data) {
    printf("in child thread is %d", threadIdx.x);
}

__global__ ParentKernel(void* data, cudaStream_t& stream2) {
    ChildKernel<<<1, 32, 0, stream2>>>(data);
    printf("in parent thread is %d", threadIdx.x)
}

int main() {
    cudaStream_t stream1;
    cudaStreamCreate(&stream1);

    cudaStream_t stream2;
    cudaStreamCreate(&stream2);
    void* data;
    ParentKernel<<<1, 32, 0, stream1>>>(data, stream2);
}