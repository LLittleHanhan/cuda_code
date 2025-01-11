#include<iostream>
using namespace std;
const int N = 1 << 16;

__global__ void kernel(float *x, int n) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    for (int i = tid; i < n; i += blockDim.x * gridDim.x) {
        x[i] = sqrt(pow(3.14159,i));
    }
}

int main() {
    cudaStream_t s1,s2;
    cudaStreamCreate(&s1);
    cudaStreamCreate(&s2);
    cudaEvent_t e1,e2;
    cudaEventCreate(&e1);
    cudaEventCreate(&e2);
    
    float *data;
    data = new float[N];
    
    float *d_data;
    cudaMalloc(&d_data,N*sizeof(float));
    cudaMemcpy(d_data,data,sizeof(float)*N,cudaMemcpyHostToDevice);
 
    for(int i=0;i<10;i++){
        kernel<<<1, 64, 0, s1>>>(d_data, N);
      
            cudaEventRecord(e1,s1);
        cudaStreamWaitEvent(s2,e1);
        kernel<<<1, 64, 0, s2>>>(d_data, N/2);
    }
    
    

    cudaMemcpy(data,d_data,sizeof(float)*N,cudaMemcpyDeviceToHost);
    for(int i=0;i<10;i++){
        std::cout<<data[i];
    }
    return 0;
}