#include <cuda_runtime.h>
#include <iostream>
#include "Profile.h"
using namespace std;

int main() {
    int batch_size = 1 << 20;
    for (; batch_size <= 1 << 30; batch_size *= 2) {
        cout << "batch size is " << batch_size / 1 << 20 << " MB" << endl;
        float* a = new float[batch_size];
        float* da;
        cudaMalloc(&da, sizeof(float) * batch_size);
        {
            Profiler p("");
            cudaMemcpy(da, a, sizeof(float) * batch_size, cudaMemcpyHostToDevice);
        }
        delete[] a;
        cudaFree(da);
    }
}