#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <stdlib.h>
#define abs(x) ((x) >= 0.0 ? (x) : -(x))

void initmatrix(float* s, int M, int N) {
    srand((unsigned)time(NULL));
    for (int i = 0; i < M * N; i++) {
        // s[i * N + j] = float(rand() % 100) / 100;
        s[i] = 1.0;
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
            }
            if (abs(temp - c[i * N + j]) > error) {
                error = abs(temp - c[i * N + j]);
            }
            // error += abs(temp - c[i * N + j]);
        }
    }
    printf("error is %.8f\n", error);
    return;
}
template <const int BM, const int BK, const int BN, const int TM, const int TN>
__global__ void sgemm(float* da, float* db, float* dc, int M, int K, int N) {
    __shared__ float das[BM][BK];
    __shared__ float dbs[BK][BN];
    float tile[TM][TN]{0};

    float dar[TM];
    float dbr[TN];

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int tid = tx * blockDim.y + ty;
    int total_thread_number = blockDim.x * blockDim.y;

    float* block_da_start = da + K * BM * bx;
    float* block_db_start = db + BN * by;

    // A:BM*BK B:BK*BN
    // big round
    // shared memary
    float* fetch_da_start = block_da_start;
    float* fetch_db_start = block_db_start;
    for (int big_round = 0; big_round < K / BK; big_round++) {
        // da=>das
        int numbers_per_thread_da = (BM * BK) / total_thread_number;
        int count = tid * numbers_per_thread_da;
        for (int data_i = 0; data_i < numbers_per_thread_da; data_i++) {
            das[count / BK][count % BK] = *(fetch_da_start + (count / BK) * K + count % BK);
            count++;
        }
        // db=>dbs
        int numbers_per_thread_db = (BK * BN) / total_thread_number;
        count = tid * numbers_per_thread_db;
        for (int data_i = 0; data_i < numbers_per_thread_db; data_i++) {
            dbs[count / BN][count % BN] = *(fetch_db_start + (count / BN) * N + count % BN);
            count++;
        }
        fetch_da_start += BK;
        fetch_db_start += (BK * N);

        __syncthreads();
        // small round
        // register
        for (int small_round = 0; small_round < BK; small_round++) {
            for (int data_i = 0; data_i < TM; data_i++) {
                dar[data_i] = das[data_i + tx * TM][small_round];
            }
            for (int data_i = 0; data_i < TN; data_i++) {
                dbr[data_i] = dbs[small_round][data_i + ty * TN];
            }
            for (int x = 0; x < TM; x++) {
                for (int y = 0; y < TN; y++) {
                    tile[x][y] += dar[x] * dbr[y];
                }
            }
        }
        __syncthreads();
    }
    // write back;
    for (int x = 0; x < TM; x++) {
        for (int y = 0; y < TN; y++) {
            dc[(bx * BM + tx * TM + x) * N + by * BN + ty * TN + y] = tile[x][y];
        }
    }
    return;
}
/*
argv[1] = M
argv[2] = N
argv[3] = K
*/
int main(int argc, char** argv) {
    int dev = 3;
    cudaSetDevice(dev);
    // TM*TN*BK>=BM or BN
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

    dim3 block(BM / TM, BN / TN);
    dim3 grid(M / BM, N / BN);

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
    // check_result(a, b, c, M, K, N);

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