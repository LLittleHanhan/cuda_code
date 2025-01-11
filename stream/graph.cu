#include<iostream>
const int N = 1 << 10;

__global__ void kernel(float *x, int n) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    for (int i = tid; i < n; i += blockDim.x * gridDim.x) {
        x[i] = sqrt(pow(3.14159,i));
    }
}

int main() {
    float *data;
    float *d_data;
    cudaMallocHost(&data, N * sizeof(float));
    for(int i=0;i<10;i++){
        data[i] = i;
    }
    cudaMalloc(&d_data,N*sizeof(float));
    cudaMemcpy(d_data,data,sizeof(float)*N,cudaMemcpyHostToDevice);
    cudaStream_t cur;
    cudaStreamCreate(&cur);
    for (int i = 0; i < 1; i++) {
        kernel<<<1, 64, 0, cur>>>(d_data, N);
    }
    cudaMemcpy(data,d_data,sizeof(float)*N,cudaMemcpyDeviceToHost);
    for(int i=0;i<10;i++){
        std::cout<<data[i];
    }
    return 0;
}

// int main() {
//     cudaGraph_t graph;
//     cudaGraphExec_t instance;


//     float *data;
//     cudaMalloc(&data, N * sizeof(float));
//     cudaStream_t cur;
//     cudaStreamCreate(&cur);

//     cudaStreamBeginCapture(cur,cudaStreamCaptureModeGlobal);
//     for (int i = 0; i < 10; i++) {
//         kernel<<<1, 64, 0, cur>>>(data, N);
//     }
//     cudaStreamEndCapture(cur,&graph);
//     cudaGraphInstantiate(&instance,graph,0);

//     cudaGraphLaunch(instance,cur);
//     return 0;
// }
