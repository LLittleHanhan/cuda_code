#include <cuda_runtime.h>
#include <stdio.h>
// cuda check
#define CUDA_CHECK()                                                                                   \
    do {                                                                                               \
        cudaError_t err = cudaGetLastError();                                                          \
        if (err != cudaSuccess) {                                                                      \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(err);                                                                                 \
        }                                                                                              \
    } while (0)

constexpr int warp = 32;
constexpr int len = 256 * 4 * 128 * 1024 * 8;
constexpr int repeat = 4;
constexpr int threads = 256;
constexpr int blocks = len / (threads * repeat);
constexpr int warps = threads / warp;

__global__ void reduceSum(float* input, float* output) {
    int bid = blockIdx.x;
    int tid = threadIdx.x;
    int warpId = tid / warp;
    int warplane = tid % warp;

    __shared__ float s[warps];
    __shared__ float si[threads * 2];
    int flag = 0;
    float sum = 0;
    unsigned int asl = __cvta_generic_to_shared(si + flag * threads + tid);
    asm volatile("cp.async.ca.shared.global.L2::128B [%0], [%1], %2;\n" ::"r"(asl),
                "l"(input + bid * threads * repeat + 0 * threads + tid),
                "n"(sizeof(float)));
    asm volatile("cp.async.commit_group;\n");
    #pragma unroll
    for(int i = 0; i < repeat; i++){
        flag = !flag;
        if(i + 1 < repeat){
            asl = __cvta_generic_to_shared(si + flag * threads + tid);
            asm volatile("cp.async.ca.shared.global.L2::128B [%0], [%1], %2;\n" ::"r"(asl),
                        "l"(input + bid * threads * repeat + (i + 1) * threads + tid),
                        "n"(sizeof(float)));
            asm volatile("cp.async.commit_group;\n");
            asm volatile("cp.async.wait_group %0;\n" ::"n"(1));
        }
        else
            asm volatile("cp.async.wait_group %0;\n" ::"n"(0));
        __syncthreads();
        float a = si[tid];
        a += __shfl_down_sync(0xffffffff, a, 16);
        a += __shfl_down_sync(0xffffffff, a, 8);
        a += __shfl_down_sync(0xffffffff, a, 4);
        a += __shfl_down_sync(0xffffffff, a, 2);
        a += __shfl_down_sync(0xffffffff, a, 1);

        if (warplane == 0) {
            s[warpId] = a;
        }

        __syncthreads();

        if (warpId == 0) {
            if (warplane < warps)
                a = s[warplane];
            else
                a = 0;
            #pragma unroll
            for (int i = warps; i >= 1; i /= 2) {
                a += __shfl_down_sync(0xffffffff, a, i);
            }
            if (warplane == 0) {
                sum += a;
            }
        }
        
    }
    if (warpId == 0 && warplane == 0) {
        output[blockIdx.x] = sum;     
    }
}

void init(float* a, int len) {
    for (int i = 0; i < len; i++) {
        a[i] = 1;
    }
}

int main() {
    float* input = new float[len];
    float* output = new float[blocks];
    init(input, len);
    init(output, blocks);

    float* dinput;
    float* doutput;
    cudaMalloc(&dinput, sizeof(float) * len);
    cudaMalloc(&doutput, sizeof(float) * blocks);
    cudaMemcpy(dinput, input, sizeof(float) * len, cudaMemcpyHostToDevice);

    reduceSum<<<blocks, threads>>>(dinput, doutput);
    
    CUDA_CHECK();
    
    cudaMemcpy(output, doutput, sizeof(float) * blocks, cudaMemcpyDeviceToHost);
    // for (int i = 0; i < blocks; i++) {
    //     printf("%f ", output[i]);
    // }
    // printf("\n");
}