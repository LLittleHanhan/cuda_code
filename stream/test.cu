const int N = 1 << 20;

__global__ void kernel(float *x, int n) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    for (int i = tid; i < n; i += blockDim.x * gridDim.x) {
        x[i] = sqrt(pow(3.14159,i));
    }
}

int main() {
    cudaEvent_t test;
    cudaEventCreate(&test);
    const int num_streams = 8;

    cudaStream_t streams[num_streams];
    float *data[num_streams];
    kernel<<<1, 1>>>(0, 0);
    for (int i = 0; i < num_streams; i++) {
        cudaStreamCreateWithFlags(&streams[i],cudaStreamNonBlocking);
 
        cudaMalloc(&data[i], N * sizeof(float));
        
        if(i==7){
            cudaStreamWaitEvent(streams[7],test);
        }
        // launch one worker kernel per stream
        kernel<<<1, 64, 0, streams[i]>>>(data[i], N);
        
        if(i==0){
            cudaEventRecord(test,streams[0]);
        }

        // launch a dummy kernel on the default stream
        kernel<<<1, 1>>>(0, 0);
    }

    cudaDeviceReset();

    return 0;
}