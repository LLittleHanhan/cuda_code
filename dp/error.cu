#include <cuda_runtime.h>
#include <stdio.h>

// 子内核
__global__ void childKernel() {
    printf("in child\n");
    int* invalid_ptr = nullptr;
    *invalid_ptr = 0;
}

__global__ void check1(){
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Error launching child kernel: %s\n", cudaGetErrorString(err));
        return;
    }
    // else
        // printf("success in kernrl\n");

}

__global__ void check2(){
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Error launching child kernel: %s\n", cudaGetErrorString(err));
        return;
    }
    // else
        // printf("success in kernrl\n");

}


__global__ void parentKernel(int depth1,int depth2,int depth3,int depth4) {
    printf("Parent kernel depth %d\n",depth1);
    if(depth1 == 10000){
        printf("end\n");
        return;
    }
    // int a[10]{depth+1};
    // for(int i =0;i<10;i++){
    //     printf("%d",a[i]);
    // }
    parentKernel<<<1, 1>>>(depth1 +1,depth2+2,depth3+3,depth4+4);
    
    // check1<<<1,1,0,cudaStreamTailLaunch>>>();
    // check2<<<1,1,0,cudaStreamTailLaunch>>>();
}

int main() {
    size_t size = 0;
    cudaDeviceGetLimit(&size,cudaLimitStackSize);
    cudaDeviceSetLimit(16,cudaLimitStackSize);
    printf("%d\n",size);

    parentKernel<<<1, 1>>>(1,1,1,1);
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Error during parent kernel execution: %s\n", cudaGetErrorString(err));
    }
    else 
        printf("sucess in host\n");
    return 0;
}
