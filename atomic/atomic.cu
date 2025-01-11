#include <iostream>

using namespace std;

__global__ void test(int* x) {
    atomicAdd(x, 1);
    printf("tid %d x %d", threadIdx.x, x);
}

int main() {
    int* x;
    cudaMalloc(&x, sizeof(int));
    test<<<2, 32>>>(x);
}