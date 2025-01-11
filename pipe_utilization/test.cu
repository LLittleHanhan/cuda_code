#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <iostream>
using namespace std;

__global__ void test(float* dr1,float* dr2,float* dr3) {
    float r1[4];
    r1[0] = dr1[0];
    r1[1] = dr1[1];
    r1[2] = dr1[2];
    r1[3] = dr1[3];
    
    float r2[4];
    r2[0] = dr2[0];
    r2[1] = dr2[1];
    r2[2] = dr2[2];
    r2[3] = dr2[3];

    float r3[4];
    r3[0] = dr3[0];
    r3[1] = dr3[1];
    r3[2] = dr3[2];
    r3[3] = dr3[3];
    
    uint32_t a[4]{0,1,2,3};
    uint32_t b[2]{0, 1};
#pragma unroll
    for (int i = 0; i < 1000; i++) {
        asm volatile(
            "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
            "{%0,  %1,  %2,  %3},"
            "{%4,  %5,  %6,  %7},"
            "{%8,  %9},"
            "{%10, %11, %12, %13};\n"
            : "=f"(r1[0]), "=f"(r1[1]), "=f"(r1[2]), "=f"(r1[3])
            : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]), "f"(r1[0]), "f"(r1[1]), "f"(r1[2]), "f"(r1[3]));

        asm volatile(
            "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
            "{%0,  %1,  %2,  %3},"
            "{%4,  %5,  %6,  %7},"
            "{%8,  %9},"
            "{%10, %11, %12, %13};\n"
            : "=f"(r2[0]), "=f"(r2[1]), "=f"(r2[2]), "=f"(r2[3])
            : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]), "f"(r2[0]), "f"(r2[1]), "f"(r2[2]), "f"(r2[3]));

        asm volatile(
            "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
            "{%0,  %1,  %2,  %3},"
            "{%4,  %5,  %6,  %7},"
            "{%8,  %9},"
            "{%10, %11, %12, %13};\n"
            : "=f"(r3[0]), "=f"(r3[1]), "=f"(r3[2]), "=f"(r3[3])
            : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]), "f"(r3[0]), "f"(r3[1]), "f"(r3[2]), "f"(r3[3]));
    }

    dr1[0] = r1[0];
    dr1[1] = r1[1];
    dr1[2] = r1[2];
    dr1[3] = r1[3];
    
    dr2[0] = r2[0];
    dr2[1] = r2[1];
    dr2[2] = r2[2];
    dr2[3] = r2[3];

    dr3[0] = r3[0];
    dr3[1] = r3[1];
    dr3[2] = r3[2];
    dr3[3] = r3[3];
}

int main() {
    float r1[4]{0, 1, 2, 3};
    float* dr1;
    cudaMalloc(&dr1, sizeof(float) * 4);
    cudaMemcpy(dr1, r1, sizeof(float) * 4, cudaMemcpyHostToDevice);

    float r2[4]{0, 1, 2, 3};
    float* dr2;
    cudaMalloc(&dr2, sizeof(float) * 4);
    cudaMemcpy(dr2, r2, sizeof(float) * 4, cudaMemcpyHostToDevice);

    float r3[4]{0, 1, 2, 3};
    float* dr3;
    cudaMalloc(&dr3, sizeof(float) * 4);
    cudaMemcpy(dr3, r3, sizeof(float) * 4, cudaMemcpyHostToDevice);
    
    test<<<128, 128>>>(dr1,dr2,dr3);
    
    cudaMemcpy(r1, dr1, sizeof(float) * 4, cudaMemcpyDeviceToHost);
    cudaMemcpy(r2, dr2, sizeof(float) * 4, cudaMemcpyDeviceToHost);
    cudaMemcpy(r3, dr3, sizeof(float) * 4, cudaMemcpyDeviceToHost);
   
    cudaDeviceSynchronize();
}