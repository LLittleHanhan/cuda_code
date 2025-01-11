#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <iostream>
using namespace std;

__global__ void testT() {
    __shared__ half smem[8 * 8];
    int tid = threadIdx.x;
    if (tid == 0){
        for (int i = 0; i < 8 * 8; i++) {
            smem[i] = __float2half(i);
        }
        for (int i = 0; i < 8 * 8; i++) {
            if(i%8==0)
                printf("\n");
            printf("%f ",__half2float(smem[i]));
        }
        printf("\n");
    }
    uint32_t dst[1];
  
    const int start = (tid % 8) * 8;
    asm volatile("ldmatrix.sync.aligned.x1.trans.m8n8.b16 {%0}, [%1];\n"
                : "=r"(dst[0])
                : "l"(&smem[start]));
    
    half* pdst = reinterpret_cast<half*>(dst); 
    printf("%d %f %f\n", threadIdx.x, __half2float(*pdst),__half2float(*(pdst+1)));
}

__global__ void testN() {
    __shared__ half smem[8 * 8];
    int tid = threadIdx.x;
    if (tid == 0){
        for (int i = 0; i < 8 * 8; i++) {
            smem[i] = __float2half(i);
        }
        for (int i = 0; i < 8 * 8; i++) {
            if(i%8==0)
                printf("\n");
            printf("%f ",__half2float(smem[i]));
        }
        printf("\n");
    }
    uint32_t dst[1];
  
    const int start = (tid % 8) * 8;
    asm volatile("ldmatrix.sync.aligned.x1.m8n8.b16 {%0}, [%1];\n"
                : "=r"(dst[0])
                : "l"(&smem[start]));
    
    half* pdst = reinterpret_cast<half*>(dst); 
    printf("%d %f %f\n", threadIdx.x, __half2float(*pdst),__half2float(*(pdst+1)));
}


int main() {
    testT<<<1, 32>>>();
    testN<<<1,32>>>();
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Error: %s\n", cudaGetErrorString(err));
    }

    return 0;
}
