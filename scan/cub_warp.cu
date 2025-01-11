#include <cub/cub.cuh>

__global__ void ExampleKernel(...)
{
    // Specialize WarpScan for type int
    typedef cub::WarpScan<int> WarpScan;

    // Allocate WarpScan shared memory for 4 warps
    __shared__ typename WarpScan::TempStorage temp_storage[4];

    // Obtain one input item per thread
    int thread_data = ...

    // Compute inclusive warp-wide prefix sums
    int warp_id = threadIdx.x / 32;
    WarpScan(temp_storage[warp_id]).InclusiveSum(thread_data, thread_data);
}
