#pragma once
#include <cuda_runtime.h>
#include <nvtx3/nvToolsExt.h>
#include <chrono>
#include <cstdio>

class Profiler {
   private:
    std::chrono::system_clock::time_point start;
    const char* tag;

   public:
    static double time;
    Profiler(const char* _tag)
        : tag(_tag) {
        cudaDeviceSynchronize();
        start = std::chrono::high_resolution_clock::now();
        nvtxRangePushA(tag);
    }

    ~Profiler() {
        cudaDeviceSynchronize();
        nvtxRangePop();
        time = static_cast<std::chrono::duration<double, std::milli>>(std::chrono::high_resolution_clock::now() - start).count();
        // printf("%s : %.3fms\n", tag, time);
    }
};