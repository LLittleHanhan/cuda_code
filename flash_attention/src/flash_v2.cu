#include <cuda_runtime.h>
#include <stdio.h>
#include <torch/extension.h>
#include <cassert>
#include <cute/tensor.hpp>
#include <random>

using namespace cute;

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

    // smem layout swizzle use for copy
    using SmemLayoutAtom = decltype(composition(
        Swizzle<3, 3, 3>{},
        Layout<Shape<Int<8>, Int<64>>,
                Stride<Int<64>, Int<1>>>{}));
    using SmemLayoutQ = decltype(tile_to_shape(
        SmemLayoutAtom{},
        Shape<Int<BR>, Int<DIM>>{}));
    using SmemLayoutK = decltype(tile_to_shape(
        SmemLayoutAtom{},
        Shape<Int<BC>, Int<DIM>>{}));
    using SmemLayoutV = SmemLayoutK; // use for g-s
    using SmemLayoutO = SmemLayoutQ;
    
    using SmemLayoutAtomT = decltype(composition(
        Swizzle<3, 3, 3>{},
        Layout<Shape<Int<64>, Int<8>>,
                Stride<Int<1>, Int<64>>>{}));
    using SmemLayoutVT = decltype(tile_to_shape( // use for s->r
        SmemLayoutAtomT{},
        Shape<Int<DIM>, Int<BC>>{}));
    
    // not swizzle for mma
    using SmemLayoutQ_NOTSWIZZLE = Layout<Shape<Int<BR>, Int<DIM>>, Stride<Int<DIM>, Int<1>>>;
    using SmemLayoutK_NOTSWIZZLE = Layout<Shape<Int<BC>, Int<DIM>>, Stride<Int<DIM>, Int<1>>>;
    using SmemLayoutVT_NOTSWIZZLE = Layout<Shape<Int<DIM>, Int<BC>>, Stride<Int<1>, Int<DIM>>>;
    using SmemLayoutO_NOTSWIZZLE = Layout<Shape<Int<BR>, Int<DIM>>, Stride<Int<DIM>, Int<1>>>;

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

template <class _FA_CONFIG>
__global__ void flash_attention_kernel(int seq, half* q, half* k, half* v, half* r, float scale) {
    using FA_CONFIG = _FA_CONFIG;

    using MMA = typename FA_CONFIG::MMA;

    using SmemLayoutQ = typename FA_CONFIG::SmemLayoutQ;
    using SmemLayoutK = typename FA_CONFIG::SmemLayoutK;
    using SmemLayoutV = typename FA_CONFIG::SmemLayoutV;
    using SmemLayoutVT = typename FA_CONFIG::SmemLayoutVT;
    using SmemLayoutO = typename FA_CONFIG::SmemLayoutO;

    using SmemLayoutQ_NOTSWIZZLE = typename FA_CONFIG::SmemLayoutQ_NOTSWIZZLE;
    using SmemLayoutK_NOTSWIZZLE = typename FA_CONFIG::SmemLayoutK_NOTSWIZZLE;
    using SmemLayoutVT_NOTSWIZZLE = typename FA_CONFIG::SmemLayoutVT_NOTSWIZZLE;
    using SmemLayoutO_NOTSWIZZLE = typename FA_CONFIG::SmemLayoutO_NOTSWIZZLE;
    
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
    int bid = bidx + bidy;
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


    /////////// smem
    __shared__ half sQ_data[br * dim];
    __shared__ half sK_data[bc * dim];
    __shared__ half sV_data[bc * dim];

    auto sQ = make_tensor(make_smem_ptr(sQ_data), SmemLayoutQ{});
    auto sK = make_tensor(make_smem_ptr(sK_data), SmemLayoutK{});
    auto sV = make_tensor(make_smem_ptr(sV_data), SmemLayoutV{});
    auto sVT = make_tensor(make_smem_ptr(sV_data), SmemLayoutVT{});
    auto sO = make_tensor(make_smem_ptr(sQ_data), SmemLayoutO{});
    
    auto sQ_NOTSWIZZLE = make_tensor(make_smem_ptr(sQ_data), SmemLayoutQ_NOTSWIZZLE{});
    auto sK_NOTSWIZZLE = make_tensor(make_smem_ptr(sK_data), SmemLayoutK_NOTSWIZZLE{});
    auto sVT_NOTSWIZZLE = make_tensor(make_smem_ptr(sV_data), SmemLayoutVT_NOTSWIZZLE{});
    auto sO_NOTSWIZZLE = make_tensor(make_smem_ptr(sQ_data), SmemLayoutO_NOTSWIZZLE{});

    /////////// tiled mma
    MMA mma;
    // gemm1
    auto thr_mma = mma.get_slice(tid);
    auto tPrQ = thr_mma.partition_fragment_A(sQ_NOTSWIZZLE);
    auto tPrK = thr_mma.partition_fragment_B(sK_NOTSWIZZLE);
    auto tPrP = partition_fragment_C(mma, Shape<Int<br>, Int<bc>>{});
    // gemm2
    auto tOrVT = thr_mma.partition_fragment_B(sVT_NOTSWIZZLE);
    auto tOrO = partition_fragment_C(mma, Shape<Int<br>, Int<bc>>{});
   
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


    /////////// max sum
    auto mv = make_tensor<float>(Shape<Int<size<0, 0>(tPrP) * size<1>(tPrP)>>{});
    auto sumv = make_tensor_like(mv);
    auto tiler = make_tile(make_layout(make_shape(size<0, 1>(tPrP)), make_stride(size<0, 0>(tPrP))),
                           make_layout(size<1>(tPrP)));
    auto tPrP_sm = zipped_divide(tPrP, tiler);
    auto tOrO_sm = zipped_divide(tOrO, tiler);
   

    /////////// begin
    copy(g2s_copy_Q, tQgQ, tQsQ);
    cp_async_fence();
    cp_async_wait<0>();
    __syncthreads();
    // scale here to reduce computation
    half2 scale_half2 = {__float2half_rn(scale), __float2half_rn(scale)};
    auto tPrQ_half2 = recast<half2>(tPrQ_copy);
   

    // for (int i = 0; i < size<2>(tPrQ_half2); i++) {
    //     copy(s2r_copy_Q, tPsQ(_, _, i), tPrQ_copy(_, _, i));
    //     for (int j = 0; j < size<0>(tPrQ_half2); j++) {
    //         tPrQ_half2(j, 0, i) = __hmul2_rn(scale_half2, tPrQ_half2(j, 0, i));
    //     }
    // }

    copy(s2r_copy_Q, tPsQ, tPrQ_copy);
    for (int i = 0; i < size(tPrQ_half2); i++) {
        tPrQ_half2(i) = __hmul2_rn(scale_half2, tPrQ_half2(i));
        
    }

    for (int n = 0; n < seq / bc; n++) {
        clear(tPrP);
        copy(g2s_copy_K, tKgK(_, _, _, n), tKsK);
        cp_async_fence();
        cp_async_wait<0>();
        __syncthreads();
       
        copy(s2r_copy_K, tPsK, tPrK_copy);

        

        gemm(mma, tPrP, tPrQ, tPrK, tPrP);

        

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

        

        // gemm2
        copy(g2s_copy_V, tVgV(_, _, _, n), tVsV);
        cp_async_fence();
        cp_async_wait<0>();
        __syncthreads();

       

        copy(s2r_copy_VT, tOsVT, tOrVT_copy);
        // f32 -> fp16
        auto tOrP_fp32 = make_tensor(tPrP.data(), tPrQ.shape());
        auto tOrP = make_tensor<half_t>(tOrP_fp32.shape());
        for (int i = 0; i < size(tOrP); i++) {
            tOrP(i) = __float2half(tOrP_fp32(i));
        }

       
        gemm(mma, tOrO, tOrP, tOrVT, tOrO);

     
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
   
    // write back
    R2SCopyO r2s_copy_O;
    auto r2s_thr_copy_O = r2s_copy_O.get_slice(tid);
    auto tOrO_r2s = r2s_thr_copy_O.retile_S(tOrO_fp16);
    auto tOsO_r2s = r2s_thr_copy_O.partition_D(sO);

    S2GCopyO s2g_copy_O;
    auto s2g_thr_copy_O = s2g_copy_O.get_slice(tid);
    auto tOsO_s2g = s2g_thr_copy_O.partition_S(sO);
    auto tOgO = s2g_thr_copy_O.partition_D(gO(_, _, 0));
   
    copy(r2s_copy_O, tOrO_r2s, tOsO_r2s);
    __syncthreads();
   
    copy(s2g_copy_O, tOsO_s2g, tOgO);
}



torch::Tensor flash(torch::Tensor q, torch::Tensor k, torch::Tensor v) {
    int batch_size = q.size(0);
    int head_num = q.size(1);
    int seq = q.size(2);
    constexpr int dim = 64;
    auto out = torch::empty_like(q);
    float scale = 1.0 / sqrt(dim) * M_LOG2E;

    using CONFIG = FA_CONFIG<64, 64, 64>;
    int br = CONFIG::BR;
    int bc = CONFIG::BC;
    dim3 grid(seq / CONFIG::BR, batch_size * head_num);
    dim3 block(CONFIG::THREADS);
    int smem_size = sizeof(half) * (br * dim + 2 * bc * dim);
    printf("blocks is x%d y%d, threads is %d, smem size is %d\n", seq / CONFIG::BR, batch_size * head_num, CONFIG::THREADS, smem_size);

    flash_attention_kernel<CONFIG><<<grid, block>>>(seq, (half*)q.data_ptr(), (half*)k.data_ptr(), (half*)v.data_ptr(), (half*)out.data_ptr(), scale);
    cudaDeviceSynchronize();
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("flash", &flash, "a tiny flash attention!");
}