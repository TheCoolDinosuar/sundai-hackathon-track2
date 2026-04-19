#include <cuda_fp16.h>
#include <cstddef>
#include <cstdint>
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>

namespace {

static constexpr int WARP_SZ = 32;
static constexpr int QUANT_WARPS_PER_BLOCK = 8;

__device__ __forceinline__ int clamp_int4(int value) {
    return max(-8, min(7, value));
}

__device__ __forceinline__ float warp_allreduce_max(float value) {
    #pragma unroll
    for (int mask = WARP_SZ / 2; mask > 0; mask >>= 1) {
        value = fmaxf(value, __shfl_xor_sync(0xffffffffu, value, mask));
    }
    return value;
}

__device__ __forceinline__ float max_abs_half8(uint4 vec) {
    float max_abs = 0.0f;
    const uint32_t words[4] = {vec.x, vec.y, vec.z, vec.w};

    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        const uint32_t word = words[i];
        const half lo = __ushort_as_half(static_cast<unsigned short>(word & 0xffffu));
        const half hi = __ushort_as_half(static_cast<unsigned short>((word >> 16) & 0xffffu));
        max_abs = fmaxf(max_abs, fabsf(__half2float(lo)));
        max_abs = fmaxf(max_abs, fabsf(__half2float(hi)));
    }

    return max_abs;
}

__device__ __forceinline__ uint32_t pack_half8_to_int4(uint4 vec, float rscale) {
    uint32_t packed = 0;
    const uint32_t words[4] = {vec.x, vec.y, vec.z, vec.w};

    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        const uint32_t word = words[i];
        const half even_h = __ushort_as_half(static_cast<unsigned short>(word & 0xffffu));
        const half odd_h = __ushort_as_half(static_cast<unsigned short>((word >> 16) & 0xffffu));
        const int q_even = clamp_int4(__float2int_rn(__half2float(even_h) * rscale));
        const int q_odd = clamp_int4(__float2int_rn(__half2float(odd_h) * rscale));
        const uint32_t packed_byte =
            static_cast<uint32_t>(((q_odd & 0xf) << 4) | (q_even & 0xf));
        packed |= packed_byte << (8 * i);
    }

    return packed;
}

template <int WARPS_PER_BLOCK>
__global__ void quantize_int4_warp_kernel(
    const half* __restrict__ input,
    uint8_t* __restrict__ output,
    half* __restrict__ scales,
    int M,
    int K,
    int group_size
) {
    const int warp = threadIdx.x / WARP_SZ;
    const int lane = threadIdx.x % WARP_SZ;
    const int num_groups = K / group_size;
    const int task = blockIdx.x * WARPS_PER_BLOCK + warp;
    const int total_tasks = M * num_groups;

    if (task >= total_tasks) {
        return;
    }

    const int row = task / num_groups;
    const int group = task - row * num_groups;
    const half* row_ptr = input + static_cast<size_t>(row) * K + group * group_size;
    uint8_t* out_ptr = output + static_cast<size_t>(row) * (K / 2) + group * (group_size / 2);

    float local_max = 0.0f;

    const bool use_vec128 =
        ((group_size & 7) == 0) &&
        ((reinterpret_cast<uintptr_t>(row_ptr) & 0xf) == 0) &&
        ((reinterpret_cast<uintptr_t>(out_ptr) & 0x3) == 0);

    if (use_vec128) {
        const int num_vec = group_size / 8;
        const uint4* vec_ptr = reinterpret_cast<const uint4*>(row_ptr);
        for (int vec_idx = lane; vec_idx < num_vec; vec_idx += WARP_SZ) {
            local_max = fmaxf(local_max, max_abs_half8(vec_ptr[vec_idx]));
        }
    } else {
        const int num_pairs = group_size / 2;
        const half2* pair_ptr = reinterpret_cast<const half2*>(row_ptr);
        for (int pair_idx = lane; pair_idx < num_pairs; pair_idx += WARP_SZ) {
            const float2 pair = __half22float2(pair_ptr[pair_idx]);
            local_max = fmaxf(local_max, fabsf(pair.x));
            local_max = fmaxf(local_max, fabsf(pair.y));
        }
    }

    const float max_abs = warp_allreduce_max(local_max);
    if (lane == 0) {
        scales[row * num_groups + group] = __float2half(max_abs * (1.0f / 7.0f));
    }

    const float rscale = max_abs > 0.0f ? (7.0f / max_abs) : 0.0f;

    if (use_vec128) {
        const int num_vec = group_size / 8;
        const uint4* vec_ptr = reinterpret_cast<const uint4*>(row_ptr);
        uint32_t* out32 = reinterpret_cast<uint32_t*>(out_ptr);
        for (int vec_idx = lane; vec_idx < num_vec; vec_idx += WARP_SZ) {
            out32[vec_idx] = pack_half8_to_int4(vec_ptr[vec_idx], rscale);
        }
    } else {
        const int num_pairs = group_size / 2;
        const half2* pair_ptr = reinterpret_cast<const half2*>(row_ptr);
        for (int pair_idx = lane; pair_idx < num_pairs; pair_idx += WARP_SZ) {
            const float2 pair = __half22float2(pair_ptr[pair_idx]);
            const int q_even = clamp_int4(__float2int_rn(pair.x * rscale));
            const int q_odd = clamp_int4(__float2int_rn(pair.y * rscale));
            out_ptr[pair_idx] = static_cast<uint8_t>(((q_odd & 0xf) << 4) | (q_even & 0xf));
        }
    }
}

// Fallback for shapes/group sizes that do not match the tensor-core kernel assumptions.
__global__ void gemm_int4_naive_kernel(
    const uint8_t* __restrict__ A,
    const uint8_t* __restrict__ B,
    const half* __restrict__ scales_A,
    const half* __restrict__ scales_B,
    half* __restrict__ C,
    int M,
    int N,
    int K,
    int group_size
) {
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row >= M || col >= N) {
        return;
    }

    const int num_groups = K / group_size;
    const int half_group = group_size / 2;
    float acc = 0.0f;

    for (int g = 0; g < num_groups; ++g) {
        const float sa = __half2float(scales_A[row * num_groups + g]);
        const float sb = __half2float(scales_B[col * num_groups + g]);
        const int byte_base = g * half_group;
        int dot = 0;

        for (int b = 0; b < half_group; ++b) {
            const uint8_t a_packed = A[row * (K / 2) + byte_base + b];
            const uint8_t b_packed = B[col * (K / 2) + byte_base + b];

            int a_lo = static_cast<int>(a_packed & 0xf);
            int b_lo = static_cast<int>(b_packed & 0xf);
            int a_hi = static_cast<int>((a_packed >> 4) & 0xf);
            int b_hi = static_cast<int>((b_packed >> 4) & 0xf);

            if (a_lo >= 8) a_lo -= 16;
            if (b_lo >= 8) b_lo -= 16;
            if (a_hi >= 8) a_hi -= 16;
            if (b_hi >= 8) b_hi -= 16;

            dot += a_lo * b_lo + a_hi * b_hi;
        }

        acc += sa * sb * static_cast<float>(dot);
    }

    C[row * N + col] = __float2half(acc);
}

static constexpr int BLOCK_M = 128;
static constexpr int BLOCK_N = 128;
static constexpr int BLOCK_K = 64;
static constexpr int NUM_WARPS = 8;
static constexpr int WARP_M = BLOCK_M / NUM_WARPS;
static constexpr int TILES_N = BLOCK_N / 16;
static constexpr int SMEM_STRIDE = BLOCK_K / 2 + 16;

__device__ __forceinline__ void mma_s4(uint4 a, uint2 b, int (&c)[4]) {
#if __CUDA_ARCH__ >= 800
    asm volatile(
        "mma.sync.aligned.m16n8k64.row.col.s32.s4.s4.s32 "
        "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};\n"
        : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3])
        : "r"(a.x), "r"(a.y), "r"(a.z), "r"(a.w), "r"(b.x), "r"(b.y));
#else
    asm volatile(
        "{\n"
        ".reg .b32 t0,t1,t2,t3;\n"
        "mma.sync.aligned.m8n8k32.row.col.s32.s4.s4.s32 {t0,t1},{%4},{%8},{%0,%1};\n"
        "mma.sync.aligned.m8n8k32.row.col.s32.s4.s4.s32 {t2,t3},{%5},{%8},{%2,%3};\n"
        "mma.sync.aligned.m8n8k32.row.col.s32.s4.s4.s32 {%0,%1},{%6},{%9},{t0,t1};\n"
        "mma.sync.aligned.m8n8k32.row.col.s32.s4.s4.s32 {%2,%3},{%7},{%9},{t2,t3};\n"
        "}\n"
        : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3])
        : "r"(a.x), "r"(a.y), "r"(a.z), "r"(a.w), "r"(b.x), "r"(b.y));
#endif
}

__device__ __forceinline__ void cp_async_16(void* dst, const void* src, bool pred) {
#if __CUDA_ARCH__ >= 800
    const unsigned smem_addr = __cvta_generic_to_shared(dst);
    asm volatile(
        "{ .reg .pred p; setp.ne.b32 p,%2,0;\n"
        "  @p cp.async.ca.shared.global [%0],[%1],16;\n"
        "  @!p st.shared.v4.u32 [%0],{0,0,0,0}; }\n"
        :
        : "r"(smem_addr), "l"(src), "r"(static_cast<int>(pred)));
#else
    uint4 zero = make_uint4(0, 0, 0, 0);
    if (pred) {
        *reinterpret_cast<uint4*>(dst) = *reinterpret_cast<const uint4*>(src);
    } else {
        *reinterpret_cast<uint4*>(dst) = zero;
    }
#endif
}

__device__ __forceinline__ void cp_commit() {
#if __CUDA_ARCH__ >= 800
    asm volatile("cp.async.commit_group;\n");
#endif
}

__device__ __forceinline__ void cp_wait(int groups_remaining) {
#if __CUDA_ARCH__ >= 800
    if (groups_remaining == 0) {
        asm volatile("cp.async.wait_group 0;\n");
    } else {
        asm volatile("cp.async.wait_group 1;\n");
    }
#else
    (void)groups_remaining;
#endif
}

__device__ __forceinline__ uint4 load_a_frag(const uint8_t* base, int stride) {
    const int lane = threadIdx.x % WARP_SZ;
    const int row_lo = lane / 4;
    const int row_hi = row_lo + 8;
    const int col = (lane % 4) * 4;

    uint4 frag;
    frag.x = *reinterpret_cast<const uint32_t*>(base + row_lo * stride + col);
    frag.y = *reinterpret_cast<const uint32_t*>(base + row_hi * stride + col);
    frag.z = *reinterpret_cast<const uint32_t*>(base + row_lo * stride + 16 + col);
    frag.w = *reinterpret_cast<const uint32_t*>(base + row_hi * stride + 16 + col);
    return frag;
}

__device__ __forceinline__ uint2 load_b_frag(const uint8_t* base, int stride) {
    const int lane = threadIdx.x % WARP_SZ;
    const int row = lane / 4;
    const int col = (lane % 4) * 4;

    uint2 frag;
    frag.x = *reinterpret_cast<const uint32_t*>(base + row * stride + col);
    frag.y = *reinterpret_cast<const uint32_t*>(base + row * stride + 16 + col);
    return frag;
}

__device__ __forceinline__ void load_gemm_stage(
    uint8_t* sA_stage,
    uint8_t* sB_stage,
    const uint8_t* A,
    const uint8_t* B,
    int bm,
    int bn,
    int kt,
    int M,
    int N,
    int halfK
) {
    const int tid = threadIdx.x;
    const int row = tid / 2;
    const int half_tile = tid % 2;
    const int kb = kt * (BLOCK_K / 2);

    const bool pred_a = (bm + row < M) && (kb + half_tile * 16 < halfK);
    cp_async_16(
        sA_stage + row * SMEM_STRIDE + half_tile * 16,
        A + static_cast<size_t>(bm + row) * halfK + kb + half_tile * 16,
        pred_a);

    const bool pred_b = (bn + row < N) && (kb + half_tile * 16 < halfK);
    cp_async_16(
        sB_stage + row * SMEM_STRIDE + half_tile * 16,
        B + static_cast<size_t>(bn + row) * halfK + kb + half_tile * 16,
        pred_b);

    cp_commit();
}

__global__ void gemm_int4_mma_kernel(
    const uint8_t* __restrict__ A,
    const uint8_t* __restrict__ B,
    const half* __restrict__ scales_A,
    const half* __restrict__ scales_B,
    half* __restrict__ C,
    int M,
    int N,
    int K,
    int group_size
) {
    const int bm = blockIdx.y * BLOCK_M;
    const int bn = blockIdx.x * BLOCK_N;
    const int tid = threadIdx.x;
    const int warp_id = tid / WARP_SZ;
    const int lane_id = tid % WARP_SZ;
    const int halfK = K / 2;
    const int num_groups = K / group_size;
    const int num_k_tiles = K / BLOCK_K;

    extern __shared__ uint8_t smem[];
    const int tileA = BLOCK_M * SMEM_STRIDE;
    const int tileB = BLOCK_N * SMEM_STRIDE;
    uint8_t* sA0 = smem;
    uint8_t* sB0 = smem + tileA;
    uint8_t* sA1 = smem + tileA + tileB;
    uint8_t* sB1 = sA1 + tileA;
    uint8_t* sA[2] = {sA0, sA1};
    uint8_t* sB[2] = {sB0, sB1};

    float acc[TILES_N][2][4];
    #pragma unroll
    for (int nt = 0; nt < TILES_N; ++nt) {
        #pragma unroll
        for (int half_tile = 0; half_tile < 2; ++half_tile) {
            acc[nt][half_tile][0] = 0.0f;
            acc[nt][half_tile][1] = 0.0f;
            acc[nt][half_tile][2] = 0.0f;
            acc[nt][half_tile][3] = 0.0f;
        }
    }

    if (num_k_tiles > 0) {
        load_gemm_stage(sA[0], sB[0], A, B, bm, bn, 0, M, N, halfK);
    }

    for (int kt = 0; kt < num_k_tiles; ++kt) {
        const int stage = kt & 1;
        if (kt + 1 < num_k_tiles) {
            load_gemm_stage(
                sA[(kt + 1) & 1],
                sB[(kt + 1) & 1],
                A,
                B,
                bm,
                bn,
                kt + 1,
                M,
                N,
                halfK);
        }

        cp_wait(kt + 1 < num_k_tiles ? 1 : 0);
        __syncthreads();

        const int g = (kt * BLOCK_K) / group_size;
        const int m_lo = bm + warp_id * WARP_M + lane_id / 4;
        const int m_hi = m_lo + 8;
        const float sa_lo = (m_lo < M) ? __half2float(scales_A[m_lo * num_groups + g]) : 0.0f;
        const float sa_hi = (m_hi < M) ? __half2float(scales_A[m_hi * num_groups + g]) : 0.0f;

        const uint4 af = load_a_frag(sA[stage] + warp_id * WARP_M * SMEM_STRIDE, SMEM_STRIDE);

        #pragma unroll
        for (int nt = 0; nt < TILES_N; ++nt) {
            const int n_off = nt * 16;
            const uint2 bf0 = load_b_frag(sB[stage] + (n_off + 0) * SMEM_STRIDE, SMEM_STRIDE);
            const uint2 bf1 = load_b_frag(sB[stage] + (n_off + 8) * SMEM_STRIDE, SMEM_STRIDE);

            int p0[4] = {0, 0, 0, 0};
            int p1[4] = {0, 0, 0, 0};
            mma_s4(af, bf0, p0);
            mma_s4(af, bf1, p1);

            const int c0 = bn + n_off + (lane_id % 4) * 2;
            const int c1 = c0 + 1;
            const int c2 = c0 + 8;
            const int c3 = c2 + 1;

            const float sb0 = (c0 < N) ? __half2float(scales_B[c0 * num_groups + g]) : 0.0f;
            const float sb1 = (c1 < N) ? __half2float(scales_B[c1 * num_groups + g]) : 0.0f;
            const float sb2 = (c2 < N) ? __half2float(scales_B[c2 * num_groups + g]) : 0.0f;
            const float sb3 = (c3 < N) ? __half2float(scales_B[c3 * num_groups + g]) : 0.0f;

            acc[nt][0][0] += static_cast<float>(p0[0]) * sa_lo * sb0;
            acc[nt][0][1] += static_cast<float>(p0[1]) * sa_lo * sb1;
            acc[nt][0][2] += static_cast<float>(p0[2]) * sa_hi * sb0;
            acc[nt][0][3] += static_cast<float>(p0[3]) * sa_hi * sb1;
            acc[nt][1][0] += static_cast<float>(p1[0]) * sa_lo * sb2;
            acc[nt][1][1] += static_cast<float>(p1[1]) * sa_lo * sb3;
            acc[nt][1][2] += static_cast<float>(p1[2]) * sa_hi * sb2;
            acc[nt][1][3] += static_cast<float>(p1[3]) * sa_hi * sb3;
        }

        __syncthreads();
    }

    const int m_lo = bm + warp_id * WARP_M + lane_id / 4;
    const int m_hi = m_lo + 8;
    for (int nt = 0; nt < TILES_N; ++nt) {
        const int c0 = bn + nt * 16 + (lane_id % 4) * 2;
        const int c1 = c0 + 1;
        const int c2 = c0 + 8;
        const int c3 = c2 + 1;

        if (m_lo < M) {
            if (c0 < N) C[m_lo * N + c0] = __float2half(acc[nt][0][0]);
            if (c1 < N) C[m_lo * N + c1] = __float2half(acc[nt][0][1]);
            if (c2 < N) C[m_lo * N + c2] = __float2half(acc[nt][1][0]);
            if (c3 < N) C[m_lo * N + c3] = __float2half(acc[nt][1][1]);
        }
        if (m_hi < M) {
            if (c0 < N) C[m_hi * N + c0] = __float2half(acc[nt][0][2]);
            if (c1 < N) C[m_hi * N + c1] = __float2half(acc[nt][0][3]);
            if (c2 < N) C[m_hi * N + c2] = __float2half(acc[nt][1][2]);
            if (c3 < N) C[m_hi * N + c3] = __float2half(acc[nt][1][3]);
        }
    }
}

}  // namespace

std::vector<torch::Tensor> quantize_int4_custom(torch::Tensor input, int group_size) {
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(input.dtype() == torch::kHalf, "input must be float16");
    TORCH_CHECK(input.dim() == 2, "input must be 2D [M, K]");

    const int M = input.size(0);
    const int K = input.size(1);

    TORCH_CHECK(K % group_size == 0, "K must be divisible by group_size");
    TORCH_CHECK(group_size % 2 == 0, "group_size must be even");

    auto output = torch::empty(
        {M, K / 2},
        torch::TensorOptions().dtype(torch::kUInt8).device(input.device()));
    const int num_groups = K / group_size;
    auto scales = torch::empty(
        {M, num_groups},
        torch::TensorOptions().dtype(torch::kHalf).device(input.device()));

    const int total_tasks = M * num_groups;
    const dim3 block(QUANT_WARPS_PER_BLOCK * WARP_SZ);
    const dim3 grid((total_tasks + QUANT_WARPS_PER_BLOCK - 1) / QUANT_WARPS_PER_BLOCK);

    quantize_int4_warp_kernel<QUANT_WARPS_PER_BLOCK>
        <<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
            reinterpret_cast<const half*>(input.data_ptr<at::Half>()),
            output.data_ptr<uint8_t>(),
            reinterpret_cast<half*>(scales.data_ptr<at::Half>()),
            M,
            K,
            group_size);

    return {output, scales};
}

torch::Tensor gemm_int4_custom(
    torch::Tensor A_packed,
    torch::Tensor B_packed,
    torch::Tensor scales_A,
    torch::Tensor scales_B,
    int group_size
) {
    TORCH_CHECK(A_packed.is_cuda(), "A_packed must be a CUDA tensor");
    TORCH_CHECK(B_packed.is_cuda(), "B_packed must be a CUDA tensor");
    TORCH_CHECK(A_packed.dtype() == torch::kUInt8, "A_packed must be uint8");
    TORCH_CHECK(B_packed.dtype() == torch::kUInt8, "B_packed must be uint8");
    TORCH_CHECK(scales_A.dtype() == torch::kHalf, "scales_A must be float16");
    TORCH_CHECK(scales_B.dtype() == torch::kHalf, "scales_B must be float16");

    const int M = A_packed.size(0);
    const int K = A_packed.size(1) * 2;
    const int N = B_packed.size(0);

    TORCH_CHECK(B_packed.size(1) * 2 == K, "A and B must have the same K dimension");
    TORCH_CHECK(K % group_size == 0, "K must be divisible by group_size");

    auto C = torch::empty(
        {M, N},
        torch::TensorOptions().dtype(torch::kHalf).device(A_packed.device()));

    const bool use_mma_kernel =
        (K % BLOCK_K == 0) &&
        (group_size >= BLOCK_K) &&
        (group_size % BLOCK_K == 0);

    if (use_mma_kernel) {
        const dim3 grid((N + BLOCK_N - 1) / BLOCK_N, (M + BLOCK_M - 1) / BLOCK_M);
        const dim3 block(NUM_WARPS * WARP_SZ);
        const int smem_bytes = 2 * (BLOCK_M * SMEM_STRIDE + BLOCK_N * SMEM_STRIDE);

        gemm_int4_mma_kernel<<<grid, block, smem_bytes, at::cuda::getCurrentCUDAStream()>>>(
            A_packed.data_ptr<uint8_t>(),
            B_packed.data_ptr<uint8_t>(),
            reinterpret_cast<const half*>(scales_A.data_ptr<at::Half>()),
            reinterpret_cast<const half*>(scales_B.data_ptr<at::Half>()),
            reinterpret_cast<half*>(C.data_ptr<at::Half>()),
            M,
            N,
            K,
            group_size);
    } else {
        const dim3 block(16, 16);
        const dim3 grid((N + 15) / 16, (M + 15) / 16);

        gemm_int4_naive_kernel<<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
            A_packed.data_ptr<uint8_t>(),
            B_packed.data_ptr<uint8_t>(),
            reinterpret_cast<const half*>(scales_A.data_ptr<at::Half>()),
            reinterpret_cast<const half*>(scales_B.data_ptr<at::Half>()),
            reinterpret_cast<half*>(C.data_ptr<at::Half>()),
            M,
            N,
            K,
            group_size);
    }

    return C;
}
