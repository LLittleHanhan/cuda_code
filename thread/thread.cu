#include<iostream>
#include<thread>
#include <nvtx3/nvToolsExt.h>
using namespace std;
const int N = 1 << 16;

__global__ void kernel(float *x, int n) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    for (int i = tid; i < n; i += blockDim.x * gridDim.x) {
        x[i] = sqrt(pow(3.14159,i));
    }
}

void func1(float* data,float* temp){
    auto program_start = std::chrono::high_resolution_clock::now();
    nvtxRangePushA("thread1 first ");
    for(int i = 0;i<N;i++){
        temp[i] += data[i];
        data[i] *= temp[i];
    }
    auto program_end = std::chrono::high_resolution_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::microseconds>(program_end - program_start).count();
    nvtxRangePop();
    cout<<"thread1 first "<<time<<endl;

    program_start = std::chrono::high_resolution_clock::now();
    nvtxRangePushA("thread1 second ");
    for(int i = 0;i<N;i++){
        temp[i] += data[i];
    }
    nvtxRangePop();
    program_end = std::chrono::high_resolution_clock::now();
    time = std::chrono::duration_cast<std::chrono::microseconds>(program_end - program_start).count();
    cout<<"thread1 second "<<time<<endl;
}

int main() {
    cudaStream_t cur,t1,t2;
    cudaStreamCreate(&cur);
    cudaStreamCreate(&t1);
    cudaStreamCreate(&t2);
    
    float *data,*temp;
    float *d_data;
    data = new float[N];
    temp = new float[N];
    for(int i=0; i<N; i++){
        data[i] = i;
        temp[i] = i;
    }
    
    // cudaMalloc(&d_data,N*sizeof(float));
    // cudaMemcpy(d_data,data,sizeof(float)*N,cudaMemcpyHostToDevice);
    // kernel<<<1, 64, 0, cur>>>(d_data, N);
    // cudaMemcpy(data,d_data,sizeof(float)*N,cudaMemcpyDeviceToHost);
    nvtxRangePushA("main thread first");
    auto program_start = std::chrono::high_resolution_clock::now();
    for(int i = 0;i<N;i++){
        temp[i] += data[i];
    }
    auto program_end = std::chrono::high_resolution_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::microseconds>(program_end - program_start).count();
    nvtxRangePop();
    cout<<"main thread first "<<time<<endl;
    
    thread t(func1,data,temp);
    t.join();
    
    program_start = std::chrono::high_resolution_clock::now();
    nvtxRangePushA("main thread second");
    for(int i = 0;i<N; i++){
        temp[i] += data[i];
    }
    nvtxRangePop();
    program_end = std::chrono::high_resolution_clock::now();
    time = std::chrono::duration_cast<std::chrono::microseconds>(program_end - program_start).count();
    cout<<"main thread second "<<time<<endl;
    
    for(int i=0; i<10; i++){
        std::cout<<temp[i]<<endl;
    }
    return 0;
}