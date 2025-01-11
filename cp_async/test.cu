#include <cuda_runtime.h>
#include <stdio.h>

__global__ void test(int* a) {
    int tid = threadIdx.x;

    __shared__ int smem[32];

    unsigned int asl = __cvta_generic_to_shared(smem + threadIdx.x);
    int src_size = 4;
    asm volatile("cp.async.ca.shared.global.L2::128B [%0], [%1], %2, %3;\n" ::"r"(asl),
                 "l"(a + tid),
                 "n"(sizeof(int)),
                 "r"(src_size));

    asm volatile("cp.async.ca.shared.global.L2::128B [%0], [%1], %2, %3;\n" ::"r"(asl),
                 "l"(a + 32 + tid),
                 "n"(sizeof(int)),
                 "r"(src_size));

    asm volatile("cp.async.ca.shared.global.L2::128B [%0], [%1], %2, %3;\n" ::"r"(asl),
                 "l"(a + 64 + tid),
                 "n"(sizeof(int)),
                 "r"(src_size));

    asm volatile("cp.async.ca.shared.global.L2::128B [%0], [%1], %2, %3;\n" ::"r"(asl),
                 "l"(a + 96 + tid),
                 "n"(sizeof(int)),
                 "r"(src_size));

    const int N = 2;
    asm volatile("cp.async.wait_group %0;\n" ::"n"(N));

    if (tid == 0) {
        for (int i = 0; i < 32; i++) {
            printf("%d ", smem[i]);
        }
    }
}
int main() {
    int* a = new int[128];
    for (int i = 0; i < 128; i++) {
        a[i] = i;
    }
    int* da;
    cudaMalloc(&da, sizeof(int) * 128);
    cudaMemcpy(da, a, sizeof(int) * 128, cudaMemcpyHostToDevice);
    test<<<1, 32>>>(da);
    cudaDeviceSynchronize();
}