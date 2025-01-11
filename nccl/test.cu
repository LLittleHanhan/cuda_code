#include <cuda_runtime.h>
#include <nccl.h>
#include <iostream>

#define NCCL_CHECK(cmd)                                                        \
    {                                                                          \
        ncclResult_t r = cmd;                                                  \
        if (r != ncclSuccess) {                                                \
            std::cerr << "NCCL error: " << ncclGetErrorString(r) << std::endl; \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    }

int main() {
    int num_devices;
    cudaGetDeviceCount(&num_devices);

    // Initialize NCCL
    ncclComm_t comms[num_devices];
    cudaStream_t streams[num_devices];

    // Allocate buffer for data on each device
    float* data[num_devices];
    for (int i = 0; i < num_devices; i++) {
        cudaSetDevice(i);
        cudaMalloc(&data[i], sizeof(float) * 10);
        cudaStreamCreate(&streams[i]);
    }

    // Initialize NCCL communicators for each device
    int* devs = new int[num_devices];
    for (int i = 0; i < num_devices; i++) {
        devs[i] = i;
    }
    NCCL_CHECK(ncclCommInitAll(comms, num_devices, devs));

    // Set the data (just an example)
    for (int i = 0; i < num_devices; i++) {
        cudaSetDevice(i);
        float value = (float)i;
        cudaMemcpyAsync(data[i], &value, sizeof(float), cudaMemcpyHostToDevice, streams[i]);
    }

    // Perform All-Reduce
    for (int i = 0; i < num_devices; i++) {
        ncclReduce((const void*)data[i], (void*)data[i], 10, ncclFloat, ncclSum, comms[i], streams[i]);
    }

    // Wait for all streams to finish
    for (int i = 0; i < num_devices; i++) {
        cudaSetDevice(i);
        cudaStreamSynchronize(streams[i]);
    }

    // Clean up
    for (int i = 0; i < num_devices; i++) {
        cudaFree(data[i]);
        float value;
        cudaMemcpyAsync(&value, data[i], sizeof(float), cudaMemcpyDeviceToHost, streams[i]);
        cout << value << endl;
        cudaStreamDestroy(streams[i]);
        ncclCommDestroy(comms[i]);
    }

    delete[] devs;
    return 0;
}
