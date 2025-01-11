#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <iostream>
using namespace std;

const int n = 32;
const int dn = 64;

// sm_70 884 f16
__global__ void test(half* a, half* b, float* c) {
    int tid = threadIdx.x;
    if ((tid > 3 && tid < 16) || tid > 19)
        return;
    half ta[4];
    half tb[4];
    float tc[8]{0};
    for (int i = 0; i < 4; i++) {
        if (tid < 16)
            ta[i] = a[(tid % 4) * 4 + i];
        else
            ta[i] = a[(tid % 4 + 4) * 4 + i];
    }
    for (int i = 0; i < 4; i++) {
        if (tid < 16)
            tb[i] = b[(tid % 4) * 8 + i];
        else
            tb[i] = b[(tid % 4) * 8 + 4 + i];
    }
    uint32_t* ta_as_uint32 = reinterpret_cast<uint32_t*>(ta);
    uint32_t* tb_as_uint32 = reinterpret_cast<uint32_t*>(tb);

    asm volatile(
        "mma.sync.aligned.m8n8k4.row.row.f32.f16.f16.f32"
        "{%0,  %1,  %2,  %3,  %4,  %5,  %6,  %7},"
        "{%8,  %9},"
        "{%10, %11},"
        "{%0, %1, %2, %3, %4, %5, %6, %7};"
        : "+f"(tc[0]), "+f"(tc[1]), "+f"(tc[2]), "+f"(tc[3]),
          "+f"(tc[4]), "+f"(tc[5]), "+f"(tc[6]), "+f"(tc[7])
        : "r"(ta_as_uint32[0]), "r"(ta_as_uint32[1]),
          "r"(tb_as_uint32[0]), "r"(tb_as_uint32[1]));

    for (int i = 0; i < 8; i++) {
        if (tid < 16)
            c[((tid & 0b1) + (i & 0b10)) * 8 + (i & 0b100) + (tid & 0b10) + (i & 0b1)] = tc[i];
        else
            c[((tid & 0b1) + (i & 0b10) + 4) * 8 + (i & 0b100) + (tid & 0b10) + (i & 0b1)] = tc[i];
    }
}

int main() {
    half* a = new half[n];
    for (int i = 0; i < n; i++) {
        a[i] = i / 10;
    }
    half* da = nullptr;
    cudaMalloc(&da, sizeof(half) * n);
    cudaMemcpy(da, a, sizeof(half) * n, cudaMemcpyHostToDevice);

    half* b = new half[n];
    for (int i = 0; i < n; i++) {
        b[i] = i / 10;
    }
    half* db = nullptr;
    cudaMalloc(&db, sizeof(half) * n);
    cudaMemcpy(db, b, sizeof(half) * n, cudaMemcpyHostToDevice);

    float* c = new float[dn];
    for (int i = 0; i < dn; i++) {
        c[i] = 0;
    }
    float* dc = nullptr;
    cudaMalloc(&dc, sizeof(float) * dn);
    cudaMemcpy(dc, c, sizeof(float) * dn, cudaMemcpyHostToDevice);

    test<<<1, 32>>>(da, db, dc);

    cudaMemcpy(c, dc, sizeof(float) * dn, cudaMemcpyDeviceToHost);

    for (int i = 0; i < dn; i++) {
        printf("%f", c[i]);
    }
}