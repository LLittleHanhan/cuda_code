#include <cuda_runtime.h>
#include <stdio.h>
#include <cassert>
#include <cute/tensor.hpp>
#include <random>

using namespace cute;

// #define DEBUG

#ifdef DEBUG
#define DEBUG_BLOCK(expr) \
    do {                  \
        expr              \
    } while (0)
#else
#define DEBUG_BLOCK(...) \
    do {                 \
    } while (0)
#endif

// cuda check
#define CUDA_CHECK()                                                                                   \
    do {                                                                                               \
        cudaError_t err = cudaGetLastError();                                                          \
        if (err != cudaSuccess) {                                                                      \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(err);                                                                                 \
        }                                                                                              \
    } while (0)

#define PRINT(var)         \
    do {                   \
        printf(#var ": "); \
        print(var);        \
        printf("\n");      \
    } while (0)

// tools
void Init(half* a, int m, int n, int seed) {
    std::mt19937 generator(seed);
    std::uniform_int_distribution<int> distribution(1, 100);
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            a[i * n + j] = __float2half(distribution(generator) * 1.0 / 100);
            // a[i * n + j] = __float2half(i * n + j * 1.0 / 100);
        }
    }
}

__host__ __device__ void print(half* a, int m, int n) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            printf("%.3f ", __half2float(a[i * n + j]));
        }
        printf("\n");
    }
}

template <int _DIM, int _BR, int _BC>
struct FA_CONFIG {
    static constexpr int DIM = _DIM;
    static constexpr int BR = _BR;
    static constexpr int BC = _BC;

    // abstract mma 64 * 16 * 16
    using MMA = decltype(make_tiled_mma(
        SM80_16x8x16_F32F16F16F32_TN{},
        Layout<Shape<_4, _1, _1>>{},
        Tile<_64, Layout<Shape<_16>, Stride<_1>>, Layout<Shape<_16>, Stride<_1>>>{}));

    static constexpr int THREADS = size(MMA{});

    // smem layout
    using SmemLayoutQ = Layout<Shape<Int<BR>, Int<DIM>>, Stride<Int<DIM>, Int<1>>>;
    using SmemLayoutK = Layout<Shape<Int<BC>, Int<DIM>>, Stride<Int<DIM>, Int<1>>>;
    using SmemLayoutV = SmemLayoutK;
    using SmemLayoutVT = Layout<Shape<Int<DIM>, Int<BC>>, Stride<Int<1>, Int<DIM>>>;
    using SmemLayoutO = Layout<Shape<Int<BR>, Int<DIM>>, Stride<Int<DIM>, Int<1>>>;

    // abstract copy
    // G2S
    using G2SCopyQ = decltype(make_tiled_copy(Copy_Atom<SM80_CP_ASYNC_CACHEGLOBAL<uint128_t>, half>{},
                                              Layout<Shape<_32, _4>, Stride<_4, _1>>{},
                                              Layout<Shape<_1, _8>>{}));
    using G2SCopyK = G2SCopyQ;
    using G2SCopyV = G2SCopyQ;

    // S2R
    using S2RCopyQ = decltype(make_tiled_copy_A(Copy_Atom<SM75_U32x4_LDSM_N, half>{}, MMA{}));
    using S2RCopyK = decltype(make_tiled_copy_B(Copy_Atom<SM75_U32x4_LDSM_N, half>{}, MMA{}));
    using S2RCopyVT = decltype(make_tiled_copy_B(Copy_Atom<SM75_U16x8_LDSM_T, half>{}, MMA{}));

    // R2S
    using R2SCopyO = decltype(make_tiled_copy_C(Copy_Atom<UniversalCopy<half>, half>{}, MMA{}));
    // S2G
    using S2GCopyO = decltype(make_tiled_copy(Copy_Atom<UniversalCopy<uint128_t>, half>{},
                                              Layout<Shape<_32, _4>, Stride<_4, _1>>{},
                                              Layout<Shape<_1, _8>>{}));
};
// SM75_U16x8_LDSM_T
template <class _FA_CONFIG>
__global__ void flash_attention_kernel(int seq, half* q, half* k, half* v, half* r, float scale) {
    using FA_CONFIG = _FA_CONFIG;

    using MMA = typename FA_CONFIG::MMA;

    using SmemLayoutQ = typename FA_CONFIG::SmemLayoutQ;
    using SmemLayoutK = typename FA_CONFIG::SmemLayoutK;
    using SmemLayoutV = typename FA_CONFIG::SmemLayoutV;
    using SmemLayoutVT = typename FA_CONFIG::SmemLayoutVT;
    using SmemLayoutO = typename FA_CONFIG::SmemLayoutO;

    using G2SCopyQ = typename FA_CONFIG::G2SCopyQ;
    using G2SCopyK = typename FA_CONFIG::G2SCopyK;
    using S2RCopyQ = typename FA_CONFIG::S2RCopyQ;
    using S2RCopyK = typename FA_CONFIG::S2RCopyK;

    using G2SCopyV = typename FA_CONFIG::G2SCopyV;
    using S2RCopyVT = typename FA_CONFIG::S2RCopyVT;

    using R2SCopyO = typename FA_CONFIG::R2SCopyO;
    using S2GCopyO = typename FA_CONFIG::S2GCopyO;

    constexpr static int dim = FA_CONFIG::DIM;
    constexpr static int br = FA_CONFIG::BR;
    constexpr static int bc = FA_CONFIG::BC;

    int bidx = blockIdx.x;
    int bidy = blockIdx.y;
    int tid = threadIdx.x;

    /////////// global mem
    auto Q = make_tensor(make_gmem_ptr(q + bidy * seq * dim), make_layout(make_shape(seq, dim), make_stride(dim, _1{})));
    auto K = make_tensor(make_gmem_ptr(k + bidy * seq * dim), make_layout(make_shape(seq, dim), make_stride(dim, _1{})));
    auto V = make_tensor(make_gmem_ptr(v + bidy * seq * dim), make_layout(make_shape(seq, dim), make_stride(dim, _1{})));
    auto O = make_tensor(make_gmem_ptr(r + bidy * seq * dim), make_layout(make_shape(seq, dim), make_stride(dim, _1{})));

    auto gQ = local_tile(Q, make_tile(Int<br>{}, Int<dim>{}), make_coord(bidx, _));
    auto gK = local_tile(K, make_tile(Int<bc>{}, Int<dim>{}), make_coord(_, _));
    auto gV = local_tile(V, make_tile(Int<bc>{}, Int<dim>{}), make_coord(_, _));
    auto gO = local_tile(O, make_tile(Int<br>{}, Int<dim>{}), make_coord(bidx, _));

    DEBUG_BLOCK(
        if (bid == 0 && tid == 0) {
            PRINT(gQ);
            PRINT(gK);
            PRINT(gV);
            PRINT(gO);
            printf("\n");
        });

    /////////// smem
    __shared__ half sQ_data[br * dim];
    __shared__ half sK_data[bc * dim];
    __shared__ half sV_data[bc * dim];

    auto sQ = make_tensor(make_smem_ptr(sQ_data), SmemLayoutQ{});
    auto sK = make_tensor(make_smem_ptr(sK_data), SmemLayoutK{});
    auto sV = make_tensor(make_smem_ptr(sV_data), SmemLayoutV{});
    auto sVT = make_tensor(make_smem_ptr(sV_data), SmemLayoutVT{});
    auto sO = make_tensor(make_smem_ptr(sQ_data), SmemLayoutO{});

    DEBUG_BLOCK(
        if (bid == 0 && tid == 0) {
            PRINT(sQ);
            PRINT(sK);
            PRINT(sV);
            PRINT(sVT);
            PRINT(sO);
            printf("\n");
        });

    /////////// tiled mma
    MMA mma;
    // gemm1
    auto thr_mma = mma.get_slice(tid);
    auto tPrQ = thr_mma.partition_fragment_A(sQ);
    auto tPrK = thr_mma.partition_fragment_B(sK);
    auto tPrP = partition_fragment_C(mma, Shape<Int<br>, Int<bc>>{});
    // gemm2
    auto tOrVT = thr_mma.partition_fragment_B(sVT);
    auto tOrO = partition_fragment_C(mma, Shape<Int<br>, Int<bc>>{});
    DEBUG_BLOCK(
        if (bid == 0 && tid == 0) {
            PRINT(tPrQ);
            PRINT(tPrK);
            PRINT(tPrP);
            PRINT(tOrVT);
            PRINT(tOrO);
            printf("\n");
        });

    /////////// tiled gs
    // gemm1
    G2SCopyQ g2s_copy_Q;
    auto g2s_thr_copy_Q = g2s_copy_Q.get_slice(tid);
    auto tQgQ = g2s_thr_copy_Q.partition_S(gQ(_, _, 0));
    auto tQsQ = g2s_thr_copy_Q.partition_D(sQ);
    G2SCopyK g2s_copy_K;
    auto g2s_thr_copy_K = g2s_copy_K.get_slice(tid);
    auto tKgK = g2s_thr_copy_K.partition_S(gK(_, _, _, 0));
    auto tKsK = g2s_thr_copy_K.partition_D(sK);
    // gemm2
    G2SCopyV g2s_copy_V;
    auto g2s_thr_copy_V = g2s_copy_V.get_slice(tid);
    auto tVgV = g2s_thr_copy_V.partition_S(gV(_, _, _, 0));
    auto tVsV = g2s_thr_copy_V.partition_D(sV);

    DEBUG_BLOCK(
        if (bid == 0 && tid == 0) {
            PRINT(tQgQ);
            PRINT(tQsQ);
            PRINT(tKgK);
            PRINT(tKsK);
            PRINT(tVgV);
            PRINT(tVsV);
            printf("\n");
        });

    /////////// tiled s2r
    // gemm1
    S2RCopyQ s2r_copy_Q;
    auto s2r_thr_copy_Q = s2r_copy_Q.get_slice(tid);
    auto tPsQ = s2r_thr_copy_Q.partition_S(sQ);
    auto tPrQ_copy = s2r_thr_copy_Q.retile_D(tPrQ);
    S2RCopyK s2r_copy_K;
    auto s2r_thr_copy_K = s2r_copy_K.get_slice(tid);
    auto tPsK = s2r_thr_copy_K.partition_S(sK);
    auto tPrK_copy = s2r_thr_copy_K.retile_D(tPrK);
    // gemm2
    S2RCopyVT s2r_copy_VT;
    auto s2r_thr_copy_VT = s2r_copy_VT.get_slice(tid);
    auto tOsVT = s2r_thr_copy_VT.partition_S(sVT);
    auto tOrVT_copy = s2r_thr_copy_VT.retile_D(tOrVT);

    DEBUG_BLOCK(
        if (bid == 0 && tid == 0) {
            PRINT(tPsQ);
            PRINT(tPrQ_copy);
            PRINT(tPsK);
            PRINT(tPrK_copy);
            PRINT(tOsVT);
            PRINT(tOrVT_copy);
            printf("\n");
        });

    /////////// max sum
    auto mv = make_tensor<float>(Shape<Int<size<0, 0>(tPrP) * size<1>(tPrP)>>{});
    auto sumv = make_tensor_like(mv);
    auto tiler = make_tile(make_layout(make_shape(size<0, 1>(tPrP)), make_stride(size<0, 0>(tPrP))),
                           make_layout(size<1>(tPrP)));
    auto tPrP_sm = zipped_divide(tPrP, tiler);
    auto tOrO_sm = zipped_divide(tOrO, tiler);
    DEBUG_BLOCK(
        if (bid == 0 && tid == 0) {
            printf("mv: ");
            print_tensor(mv);
            printf("\n");
            printf("sumv: ");
            print_tensor(sumv);
            printf("\n");
            printf("tPrP_sm: ");
            print(tPrP_sm);
            printf("\n");
            printf("tOrO_sm: ");
            print(tOrO_sm);
            printf("\n");
        });

    /////////// begin
    copy(g2s_copy_Q, tQgQ, tQsQ);

    for (int n = 0; n < seq / bc; n++) {
        clear(tPrP);
        copy(g2s_copy_K, tKgK(_, _, _, n), tKsK);
        cp_async_fence();
        cp_async_wait<0>();
        __syncthreads();
        DEBUG_BLOCK(
            if (bid == 0 && tid == 0) {
                printf("smem q\n");
                print(sQ_data, br, dim);
                printf("smem k\n");
                print(sK_data, bc, dim);
                printf("\n");
            });

        copy(s2r_copy_Q, tPsQ, tPrQ_copy);
        copy(s2r_copy_K, tPsK, tPrK_copy);

        DEBUG_BLOCK(
            if (bid == 0 && tid == 0) {
                printf("tPrQ: ");
                print_tensor(tPrQ);
                printf("\n");
                printf("tPrK: ");
                print_tensor(tPrK);
                printf("\n");
            });

        gemm(mma, tPrP, tPrQ, tPrK, tPrP);

        // scale
        for (int i = 0; i < tPrP.size(); i++) {
            tPrP(i) *= scale;
        }

        DEBUG_BLOCK(
            if (bid == 0 && tid == 0) {
                printf("tPrP: ");
                print_tensor(tPrP);
                print_tensor(tPrP_sm);
                printf("\n");
            });

        // 1. compute new max
        // 2. scale old sum and old o
        // 3. compute new p
        // 4. compute new sum
        for (int i = 0; i < size<0>(tPrP_sm); i++) {
            float om = mv(i);
            float nm = om;
            float sum = 0;
            // compute new max
            for (int j = 0; j < size<1>(tPrP_sm); j++)
                nm = max(nm, tPrP_sm(i, j));
            nm = max(nm, __shfl_xor_sync(0xffffffff, nm, 0x2));
            nm = max(nm, __shfl_xor_sync(0xffffffff, nm, 0x1));
            // scale old sum and old o
            float scale = exp2f(om - nm);
            // compute new sum
            for (int j = 0; j < size<1>(tPrP_sm); j++) {
                tOrO_sm(i, j) *= scale;
                tPrP_sm(i, j) = exp2f(tPrP_sm(i, j) - nm);
                sum += tPrP_sm(i, j);
            }
            sum += __shfl_xor_sync(0xffffffff, sum, 0x2);
            sum += __shfl_xor_sync(0xffffffff, sum, 0x1);
            sumv(i) = sumv(i) * scale + sum;
            mv(i) = nm;
        }

        DEBUG_BLOCK(
            if (bid == 0 && tid == 0) {
                printf("mv: ");
                print_tensor(mv);
                printf("sumv: ");
                print_tensor(sumv);
            });

        // gemm2
        copy(g2s_copy_V, tVgV(_, _, _, n), tVsV);
        cp_async_fence();
        cp_async_wait<0>();
        __syncthreads();

        DEBUG_BLOCK(
            if (bid == 0 && tid == 0) {
                // printf("smem v\n");
                // print(sV_data,bc,dim);
                printf("\n");
            });

        copy(s2r_copy_VT, tOsVT, tOrVT_copy);
        // f32 -> fp16
        auto tOrP_fp32 = make_tensor(tPrP.data(), tPrQ.shape());
        auto tOrP = make_tensor<half_t>(tOrP_fp32.shape());
        for (int i = 0; i < size(tOrP); i++) {
            tOrP(i) = __float2half(tOrP_fp32(i));
        }

        DEBUG_BLOCK(
            if (bid == 0 && tid == 0) {
                printf("tOrP: ");
                print_tensor(tOrP);
                printf("\n");
                printf("tOrVT: ");
                print_tensor(tOrVT);
                printf("\n");
            });
        gemm(mma, tOrO, tOrP, tOrVT, tOrO);

        DEBUG_BLOCK(
            if (bid == 0 && tid == 0) {
                printf("tOrO: ");
                print_tensor(tOrO);
                printf("\n");
            });
    }
    // normlize
    for (int i = 0; i < size<0>(tOrO_sm); i++) {
        for (int j = 0; j < size<1>(tOrO_sm); j++)
            tOrO_sm(i, j) = tOrO_sm(i, j) * __frcp_rn(sumv(i));
    }

    // rO -> sO -> gO
    auto tOrO_fp16 = make_tensor<half_t>(tOrO.shape());
    for (int i = 0; i < size(tOrO); i++) {
        tOrO_fp16(i) = __float2half(tOrO(i));
    }
    DEBUG_BLOCK(
        if (bid == 0 && tid == 0) {
            printf("tOrO_fp16: ");
            print_tensor(tOrO_fp16);
            printf("\n");
        });
    // write back
    R2SCopyO r2s_copy_O;
    auto r2s_thr_copy_O = r2s_copy_O.get_slice(tid);
    auto tOrO_r2s = r2s_thr_copy_O.retile_S(tOrO_fp16);
    auto tOsO_r2s = r2s_thr_copy_O.partition_D(sO);

    S2GCopyO s2g_copy_O;
    auto s2g_thr_copy_O = s2g_copy_O.get_slice(tid);
    auto tOsO_s2g = s2g_thr_copy_O.partition_S(sO);
    auto tOgO = s2g_thr_copy_O.partition_D(gO(_, _, 0));
    DEBUG_BLOCK(
        if (bid == 0 && tid == 0) {
            PRINT(tOrO_r2s);
            PRINT(tOsO_r2s);
            PRINT(tOsO_s2g);
            PRINT(tOgO);
            printf("\n");
        });
    copy(r2s_copy_O, tOrO_r2s, tOsO_r2s);
    __syncthreads();
    DEBUG_BLOCK(
        if (bid == 0 && tid == 0) {
            printf("smem o\n");
            print(sQ_data, br, bc);
            printf("\n");
        });
    copy(s2g_copy_O, tOsO_s2g, tOgO);
}

template <class _FA_CONFIG>
void flash_attention() {
    using FA_CONFIG = _FA_CONFIG;
    int br = FA_CONFIG::BR;
    int bc = FA_CONFIG::BC;

    constexpr int dim = FA_CONFIG::DIM;
    int head_num = 12;
    
    int seq = 1024;
    int batch_size = 256;
    dim3 grid(seq/br,batch_size * head_num);
    dim3 block(FA_CONFIG::THREADS);

    half* q = new half[seq * dim * batch_size * head_num];
    half* k = new half[seq * dim * batch_size * head_num];
    half* v = new half[seq * dim * batch_size * head_num];
    half* r = new half[seq * dim * batch_size * head_num];
    Init(q, seq, dim * batch_size * head_num, 0);
    Init(k, seq, dim * batch_size * head_num, 1);
    Init(v, seq, dim * batch_size * head_num, 2);

    // print(q, seq, dim);

    half *dq, *dk, *dv, *dr;
    cudaMalloc(&dq, sizeof(half) * seq * dim * batch_size * head_num);
    cudaMalloc(&dk, sizeof(half) * seq * dim * batch_size * head_num);
    cudaMalloc(&dv, sizeof(half) * seq * dim * batch_size * head_num);
    cudaMalloc(&dr, sizeof(half) * seq * dim * batch_size * head_num);

    cudaMemcpy(dq, q, sizeof(half) * seq * dim * batch_size * head_num, cudaMemcpyHostToDevice);
    cudaMemcpy(dk, k, sizeof(half) * seq * dim * batch_size * head_num, cudaMemcpyHostToDevice);
    cudaMemcpy(dv, v, sizeof(half) * seq * dim * batch_size * head_num, cudaMemcpyHostToDevice);

    
    int smem_size = sizeof(half) * (br * dim + 2 * bc * dim);
    printf("blocks is x%d y%d, threads is %d, smem size is %d\n", grid.x,grid.y, block.x, smem_size);
    float scale = 1.0 / sqrt(dim) * M_LOG2E;
    
    flash_attention_kernel<FA_CONFIG><<<grid, block>>>(seq, dq, dk, dv, dr, scale);
    CUDA_CHECK();
    cudaMemcpy(r, dr, sizeof(half) * seq * dim * batch_size * head_num, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    // print(r, seq, dim);
}

int main() {
    flash_attention<FA_CONFIG<64, 64, 64>>();
    return 0;
}