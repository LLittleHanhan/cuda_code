#include <cuda_runtime.h>
#include <stdio.h>


__device__ int flag = 1;
__global__ void test() {
    int tid = threadIdx.x;
    if(tid < 16){
        if(tid == 0){
            printf("tid 0-15 start\n");
        }
        // __syncthreads();
        if(tid == 0){
            printf("tid 0-15 middle\n");
        }
        
        // while(flag == 1);
        
        if(tid == 0){
            printf("tid 0-15 end\n");
        }
    }
    // else if (tid < 32){
    //     if(tid == 16){
    //         printf("tid 16-31 start\n");
    //     }
    //     // __syncthreads();
    //     if(tid == 16){
    //         printf("tid 16-31 end\n");
    //     }
    // }

    else if (tid < 64){
        if(tid == 32){
            printf("tid 32-63 start\n");
        }
        __syncthreads();
        if(tid == 32){
            printf("tid 32-63 end\n");
        }
    }
    
    else if (tid < 96){
        if(tid  == 64){
            printf("tid 64-95 start\n");
        }
        // __syncthreads();
        flag = 0; 
        if(tid == 64){
            printf("tid 64-95 end\n");
        }

    }
    __syncthreads();
    if(tid == 0 || tid == 32 || tid == 64){
        printf("all end\n");
    }
    
}

int main() {
    test<<<1, 96>>>();
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(err));
    }
}