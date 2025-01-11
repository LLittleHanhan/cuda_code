#include <iostream>

int main() {
    cudaGraph_t graph;
    cudaGraphExec_t instance;

    float* data;
    cudaMalloc(&data, N * sizeof(float));
    cudaStream_t cur;
    cudaStreamCreate(&cur);

    cudaStreamBeginCapture(cur, cudaStreamCaptureModeGlobal);
    for (int i = 0; i < 10; i++) {
        kernel<<<1, 64, 0, cur>>>(data, N);
    }
    cudaStreamEndCapture(cur, &graph);

    cudaGraphInstantiate(&instance, graph, 0);
    cudaGraphLaunch(instance, cur);
    return 0;
}
