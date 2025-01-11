#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <stdlib.h>
#define abs(x) ((x) >= 0.0 ? (x) : -(x))
#define FETCH_FLOAT4(pointer) (reinterpret_cast<float4*>(pointer)[0])

void initmatrix(float* s, int M, int N) {
    for (int i = 0; i < M * N; i++) {
        s[i] = i % 13;
        // s[i] = 1.0;
    }
    return;
}

void check_result(float* a, float* b, float* c, int M, int K, int N) {
    float error = 0;
    float temp = 0;
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            temp = 0;
            for (int k = 0; k < K; k++) {
                temp += a[i * K + k] * b[k * N + j];
                // k=64
                //  if(i==34 && j==11){
                //      printf("%.8f %.8f %.8f\n",a[i * K + k],b[k * N + j],temp);
                //  }
            }
            if (abs(temp - c[i * N + j]) > error) {
                error = abs(temp - c[i * N + j]);
            }
        }
    }
    printf("max error is %.8f\n", error);
    return;
}
template <const int BM, const int BK, const int BN, const int TM, const int TN>
__global__ void sgemm(float* __restrict__ da, float* __restrict__ db, float* __restrict__ dc, int M, int K, int N) {
    __shared__ float das[2][BK][BM];
    __shared__ float dbs[2][BK][BN];
    float tile[TM][TN]{0};
    float dar[2][TM];
    float dbr[2][TN];
    int shared_flag = 0;
    int register_flag = 0;
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int tid = ty * blockDim.x + tx;
    const int thread_per_block = (BM / TM) * (BN / TN);
   
    // prefetch
    float* block_start_address_da = da + by * BM * K;
    float* block_start_address_db = db + bx * BN;
    // float4，4 float per thread
    const int row_float4_nums_da = BK / 4;
    const int row_float4_nums_db = BN / 4;
    const int row_float4_da = tid / row_float4_nums_da;
    const int col_float4_da = tid % row_float4_nums_da * 4;
    const int row_float4_db = tid / row_float4_nums_db;
    const int col_float4_db = tid % row_float4_nums_db * 4;

    const int row_stride_da = thread_per_block / row_float4_nums_da;
    const int row_stride_db = thread_per_block / row_float4_nums_db;

    const int nums_per_thread_da = BM * BK / thread_per_block;
    const int nums_per_thread_db = BK * BN / thread_per_block;
    float da_reg[nums_per_thread_da];
    float db_reg[nums_per_thread_db];
// da
#pragma unroll
    for (int i = 0; i < BM; i += row_stride_da) {
        // if (i + row_float4_da < BM) {
        FETCH_FLOAT4(&da_reg[0]) = FETCH_FLOAT4(block_start_address_da + (i + row_float4_da) * K + col_float4_da);
        das[shared_flag][col_float4_da][i + row_float4_da] = da_reg[0];
        das[shared_flag][col_float4_da + 1][i + row_float4_da] = da_reg[1];
        das[shared_flag][col_float4_da + 2][i + row_float4_da] = da_reg[2];
        das[shared_flag][col_float4_da + 3][i + row_float4_da] = da_reg[3];
        //}
    }
    
// db
#pragma unroll
    for (int i = 0; i < BK; i += row_stride_db) {
        // if (i + row_float4_db < BK) {
        FETCH_FLOAT4(&dbs[shared_flag][i + row_float4_db][col_float4_db]) = FETCH_FLOAT4(block_start_address_db + (i + row_float4_db) * N + col_float4_db);
        //}
    }
    __syncthreads();
// register
#pragma unroll
    for (int i = 0; i < TM; i += 4) {
        FETCH_FLOAT4(&dar[register_flag][i]) = FETCH_FLOAT4(&das[shared_flag][0][ty * TM + i]);
    }
#pragma unroll
    for (int i = 0; i < TN; i += 4) {
        FETCH_FLOAT4(&dbr[register_flag][i]) = FETCH_FLOAT4(&dbs[shared_flag][0][tx * TN + i]);
    }

// round
#pragma unroll
    for (int big_round = 0; big_round < K / BK; big_round++) {
        shared_flag = !shared_flag;
        if (big_round != K / BK - 1) {
            block_start_address_da += BK;
            block_start_address_db += BK * N;
// fetch global memery to shared memery for next big round
//  da
#pragma unroll
            for (int i = 0; i < BM; i += row_stride_da) {
                // if (i + row_float4_da < BM) {
                FETCH_FLOAT4(&da_reg[i * 4]) = FETCH_FLOAT4(block_start_address_da + (i + row_float4_da) * K + col_float4_da);
                //}
            }
// db
#pragma unroll
            for (int i = 0; i < BK; i += row_stride_db) {
                // if (i + row_float4_db < BK) {
                FETCH_FLOAT4(&db_reg[i * 4]) = FETCH_FLOAT4(block_start_address_db + (i + row_float4_db) * N + col_float4_db);
                //}
            }
        }
#pragma unroll
        for (int small_round = 0; small_round < BK - 1; small_round++) {
            register_flag = !register_flag;
#pragma unroll
            for (int i = 0; i < TM; i += 4) {
                FETCH_FLOAT4(&dar[register_flag][i]) = FETCH_FLOAT4(&das[!shared_flag][small_round + 1][ty * TM + i]);
            }
#pragma unroll
            for (int i = 0; i < TN; i += 4) {
                FETCH_FLOAT4(&dbr[register_flag][i]) = FETCH_FLOAT4(&dbs[!shared_flag][small_round + 1][tx * TN + i]);
            }
// compute last register data
#pragma unroll
            for (int x = 0; x < TM; x++) {
#pragma unroll
                for (int y = 0; y < TN; y++) {
                    tile[x][y] += dar[!register_flag][x] * dbr[!register_flag][y];
                }
            }
        }
    

// cpy register to shared memery
//  da
#pragma unroll
        for (int i = 0; i < BM; i += row_stride_da) {
            // if (i + row_float4_da < BM) {
            das[shared_flag][col_float4_da][i + row_float4_da] = da_reg[i * 4];
            das[shared_flag][col_float4_da + 1][i + row_float4_da] = da_reg[i * 4 + 1];
            das[shared_flag][col_float4_da + 2][i + row_float4_da] = da_reg[i * 4 + 2];
            das[shared_flag][col_float4_da + 3][i + row_float4_da] = da_reg[i * 4 + 3];
            //}
        }
// db
#pragma unroll
        for (int i = 0; i < BK; i += row_stride_db) {
            // if (i + row_float4_db < BK) {
            FETCH_FLOAT4(&dbs[shared_flag][i + row_float4_db][col_float4_db]) = FETCH_FLOAT4(&db_reg[i * 4]);
            //}
        }
        __syncthreads();
        register_flag = !register_flag;
        // fetch fist line shared memery to register for next big round
        if (big_round != K / BK - 1) {
#pragma unroll
            for (int i = 0; i < TM; i += 4) {
                FETCH_FLOAT4(&dar[register_flag][i]) = FETCH_FLOAT4(&das[shared_flag][0][ty * TM + i]);
            }
#pragma unroll
            for (int i = 0; i < TN; i += 4) {
                FETCH_FLOAT4(&dbr[register_flag][i]) = FETCH_FLOAT4(&dbs[shared_flag][0][tx * TN + i]);
            }
        }
#pragma unroll
        for (int x = 0; x < TM; x++) {
#pragma unroll
            for (int y = 0; y < TN; y++) {
                tile[x][y] += dar[!register_flag][x] * dbr[!register_flag][y];
            }
        }
    }
    // copy tile to global mem
    float* thread_start_address_dc = dc + (by * BM + ty * TM) * N + bx * BN + tx * TN;
#pragma unroll
    for (int x = 0; x < TM; x++) {
#pragma unroll
        for (int y = 0; y < TN; y += 4) {
            FETCH_FLOAT4(thread_start_address_dc + x * N + y) = FETCH_FLOAT4(&tile[x][y]);
        }
    }
}
/*
argv[1] = M
argv[2] = N
argv[3] = K
*/
int main(int argc, char** argv) {
    int dev = 0;
    cudaSetDevice(dev);
    const int M = 1024;
    const int K = 1024;
    const int N = 1024;

    const int BM = 128;
    const int BK = 8;
    const int BN = 128;
    const int TM = 8;
    const int TN = 8;
    double flopsPerMatrixMul = 2.0 * M * N * K;

    float* a = new float[M * K];
    float* b = new float[K * N];
    float* c = new float[M * N];
    initmatrix(a, M, K);
    initmatrix(b, K, N);
    float *da, *db, *dc;
    cudaMalloc(&da, sizeof(float) * M * K);
    cudaMalloc(&db, sizeof(float) * K * N);
    cudaMalloc(&dc, sizeof(float) * M * N);
    cudaMemcpy(da, a, sizeof(float) * M * K, cudaMemcpyHostToDevice);
    cudaMemcpy(db, b, sizeof(float) * K * N, cudaMemcpyHostToDevice);

    dim3 block(BN / TN, BM / TM);
    dim3 grid(N / BN, M / BM);

    int iter = 1000;
    float msecTotal;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    for (int i = 0; i < iter; i++) {
        sgemm<BM, BK, BN, TM, TN><<<grid, block>>>(da, db, dc, M, K, N);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&msecTotal, start, stop);
    double msecPerMatrixMul = msecTotal / iter;
    double gigaFLOPS = (flopsPerMatrixMul * 1.0e-9f) / (msecPerMatrixMul / 1000.0f);
    printf("My gemm Performance= %.2f GFLOPS, Time= %.3f msec, Size= %.0f Ops,\n", gigaFLOPS, msecPerMatrixMul, flopsPerMatrixMul);

    cudaMemcpy(c, dc, sizeof(float) * M * N, cudaMemcpyDeviceToHost);
    //check_result(a, b, c, M, K, N);

    cublasHandle_t handle;
    cublasCreate(&handle);
    float alpha = 1;
    float beta = 0;
    cudaEventRecord(start);
    for (int i = 0; i < iter; i++) {
        cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_T, M, N, K, &alpha, da, K, db, N, &beta, dc, M);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&msecTotal, start, stop);
    msecPerMatrixMul = msecTotal / iter;
    gigaFLOPS = (flopsPerMatrixMul * 1.0e-9f) / (msecPerMatrixMul / 1000.0f);
    printf("cublas Performance= %.2f GFLOPS, Time= %.3f msec, Size= %.0f Ops,\n", gigaFLOPS, msecPerMatrixMul, flopsPerMatrixMul);
    cublasDestroy(handle);

    cudaFree(da);
    cudaFree(db);
    cudaFree(dc);
    delete[] a;
    delete[] b;
    delete[] c;
    return 0;
}