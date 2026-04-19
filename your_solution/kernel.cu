/**
 * INT4 Quantization + GEMM kernel -- your_solution/kernel.cu
 *
 * Quantize kernel:
 *   - 128-bit vectorized loads (__ldg float4) from global memory
 *   - Each thread loads 8 consecutive FP16 values (one 128-bit chunk)
 *   - Thread i and thread i+1 hit adjacent 128-bit chunks (contiguous nibbles)
 *   - Warp-level max reduction via __shfl_xor_sync (no shared memory needed)
 *   - 32-bit stores for 4 packed INT4 bytes
 *
 * GEMM kernel:
 *   - Tensor Core mma.sync.aligned.m16n8k64.row.col.s32.s4.s4.s32 (Ampere)
 *   - 128-bit cp.async (16-byte) loads from global → shared memory
 *   - Double-buffered shared memory tiles (128×64 A, 128×64 B)
 *   - Warp-level scale broadcasting
 */

#include <cuda_fp16.h>
#include <cstdint>
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>


// =============================================================
// INT4 Quantization Kernel (vectorized 128-bit global loads)
// =============================================================
//
// Layout: group_size elements per group; group_size must be a multiple of 8.
// Threads per group: tpg = group_size / 8  (must be ≤ 32).
// Each thread loads 8 FP16 values via a single 128-bit (float4) load.
// Thread t in a group covers elements [t*8 .. t*8+7] of the group.
// Adjacent threads t, t+1 therefore cover adjacent 128-bit chunks of input.
//
// Global memory coalescing within a row:
//   - k_base = grp*group_size + t*8
//   - Thread 0: bytes [grp*group_size*2 .. +16)
//   - Thread 1: bytes [grp*group_size*2+16 .. +32)   ← adjacent 128-bit chunk
//   - Address offset between thread t and t+1 is exactly 16 bytes = 128 bits. ✓
//
__global__ void quantize_int4_kernel(
    const half* __restrict__ input,    // [M, K]
    uint8_t*    __restrict__ output,   // [M, K/2]
    half*       __restrict__ scales,   // [M, num_groups]
    int M, int K, int group_size)
{
    const int tpg = group_size / 8;          // threads per group
    const int rpb = blockDim.x / tpg;        // rows per block

    int t   = threadIdx.x % tpg;             // thread index within group
    int row = (int)blockIdx.x * rpb + (int)threadIdx.x / tpg;
    int grp = blockIdx.y;

    // k_base: starting element index for this thread within the row
    int k_base = grp * group_size + t * 8;

    // ---- 128-bit load (8 × fp16 = 128 bits) ----
    // All threads in the warp execute this path (OOB rows load zeros) so there
    // is no warp divergence before the __shfl_xor_sync calls below.
    float4 raw = {0.f, 0.f, 0.f, 0.f};
    if (row < M)
        raw = __ldg(reinterpret_cast<const float4*>(input + (size_t)row * K + k_base));

    const half* vals = reinterpret_cast<const half*>(&raw);

    // ---- Local max over the 8 loaded elements ----
    float lmax = 0.f;
    #pragma unroll
    for (int i = 0; i < 8; i++)
        lmax = fmaxf(lmax, fabsf(__half2float(vals[i])));

    // ---- Warp-level max reduction across the tpg threads of this group ----
    // Groups occupy contiguous lanes, so shfl_xor with offset < tpg never
    // crosses group boundaries. All 32 warp threads participate (no divergence).
    for (int off = tpg >> 1; off >= 1; off >>= 1)
        lmax = fmaxf(lmax, __shfl_xor_sync(0xFFFFFFFF, lmax, off));

    // From here, only in-bounds rows produce output.
    if (row >= M) return;

    int num_groups = K / group_size;
    float scale  = lmax / 7.f;
    float rscale = (lmax > 0.f) ? (7.f / lmax) : 0.f;

    // Thread 0 of each group writes the per-group scale
    if (t == 0)
        scales[(size_t)row * num_groups + grp] = __float2half(scale);

    // ---- Quantize 8 INT4 values and pack into 4 bytes (32-bit store) ----
    // Byte i: low nibble = element 2i (even), high nibble = element 2i+1 (odd)
    uint32_t packed = 0;
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        int q0 = max(-8, min(7, __float2int_rn(__half2float(vals[2*i  ]) * rscale)));
        int q1 = max(-8, min(7, __float2int_rn(__half2float(vals[2*i+1]) * rscale)));
        packed |= (uint32_t)(((q1 & 0xF) << 4) | (q0 & 0xF)) << (i * 8);
    }

    // ---- 32-bit aligned store (4 packed INT4 bytes per thread) ----
    // Thread t stores to output byte offset: group_byte_base + t*4
    // This ensures thread t and t+1 write to adjacent 4-byte chunks, covering
    // a contiguous 16-byte (128-bit) region per 4 threads within a group.
    size_t out_idx = (size_t)row * (K / 2) + (size_t)grp * (group_size / 2) + t * 4;
    *reinterpret_cast<uint32_t*>(output + out_idx) = packed;
}


std::vector<torch::Tensor> quantize_int4_custom(torch::Tensor input, int group_size) {
    TORCH_CHECK(input.is_cuda(),                   "input must be a CUDA tensor");
    TORCH_CHECK(input.dtype() == torch::kHalf,     "input must be float16");
    TORCH_CHECK(input.dim() == 2,                  "input must be 2D [M, K]");

    int M = input.size(0);
    int K = input.size(1);

    TORCH_CHECK(K % group_size == 0,   "K must be divisible by group_size");
    TORCH_CHECK(group_size % 8 == 0,   "group_size must be a multiple of 8");
    int tpg = group_size / 8;
    TORCH_CHECK(tpg <= 32,             "group_size must be <= 256 (single-warp reduction)");

    int num_groups = K / group_size;
    int rpb        = 256 / tpg;       // rows per block (256 threads total)

    auto output = torch::empty({M, K / 2},
        torch::TensorOptions().dtype(torch::kUInt8).device(input.device()));
    auto scales = torch::empty({M, num_groups},
        torch::TensorOptions().dtype(torch::kHalf).device(input.device()));

    dim3 block(256);
    dim3 grid((M + rpb - 1) / rpb, num_groups);

    quantize_int4_kernel<<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const half*>(input.data_ptr<at::Half>()),
        output.data_ptr<uint8_t>(),
        reinterpret_cast<half*>(scales.data_ptr<at::Half>()),
        M, K, group_size);

    return {output, scales};
}


// =============================================================
// MMA-based INT4 GEMM (Tensor Core m16n8k64, 128-bit cp.async)
// =============================================================
//
// Global→shared memory tile loads use cp.async.ca.shared.global with 16-byte
// (128-bit) transactions.  Thread layout for tile loading:
//   row = tid / 2   (which of BLOCK_M rows)
//   half = tid % 2  (first or second 16-byte half of the row's 32-byte tile slice)
// => Thread 2t and 2t+1 both serve row t and load the two adjacent 128-bit
//    chunks of that row: bytes [kb..kb+15] and [kb+16..kb+31].
//    Adjacent threads within each pair cover adjacent INT4 nibbles. ✓
//
// Shared→register loads (load_a_frag / load_b_frag) use aligned 32-bit reads.
//
// Accumulation: FP32 accumulators, written to FP16 output in the epilogue.

static constexpr int BLOCK_M   = 128;
static constexpr int BLOCK_N   = 128;
static constexpr int BLOCK_K   = 64;
static constexpr int WARP_SZ   = 32;
static constexpr int NUM_WARPS = 8;
static constexpr int WARP_M    = BLOCK_M / NUM_WARPS;   // 16
static constexpr int TILES_N   = BLOCK_N / 16;           // 8

// Shared memory row stride: BLOCK_K/2 bytes of data + 16 bytes of padding
// (keeps each row 16-byte aligned for cp.async)
static constexpr int SMEM_STRIDE = BLOCK_K / 2 + 16;    // 48 bytes


// ---- MMA wrapper: m16n8k64 INT4×INT4 → INT32 ----
__device__ __forceinline__ void mma_s4(uint4 a, uint2 b, int (&c)[4]) {
#if __CUDA_ARCH__ >= 800
    asm volatile(
        "mma.sync.aligned.m16n8k64.row.col.s32.s4.s4.s32 "
        "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};\n"
        : "+r"(c[0]),"+r"(c[1]),"+r"(c[2]),"+r"(c[3])
        : "r"(a.x),"r"(a.y),"r"(a.z),"r"(a.w),"r"(b.x),"r"(b.y));
#else
    // SM75 fallback via m8n8k32
    asm volatile("{"
        ".reg .b32 t0,t1,t2,t3;\n"
        "mma.sync.aligned.m8n8k32.row.col.s32.s4.s4.s32 {t0,t1},{%4},{%8},{%0,%1};\n"
        "mma.sync.aligned.m8n8k32.row.col.s32.s4.s4.s32 {t2,t3},{%5},{%8},{%2,%3};\n"
        "mma.sync.aligned.m8n8k32.row.col.s32.s4.s4.s32 {%0,%1},{%6},{%9},{t0,t1};\n"
        "mma.sync.aligned.m8n8k32.row.col.s32.s4.s4.s32 {%2,%3},{%7},{%9},{t2,t3};\n"
        "}\n"
        : "+r"(c[0]),"+r"(c[1]),"+r"(c[2]),"+r"(c[3])
        : "r"(a.x),"r"(a.y),"r"(a.z),"r"(a.w),"r"(b.x),"r"(b.y));
#endif
}


// ---- 128-bit async global→shared copy ----
__device__ __forceinline__ void cp_async_16(void *dst, const void *src, bool pred) {
    unsigned s = __cvta_generic_to_shared(dst);
    asm volatile(
        "{ .reg .pred p; setp.ne.b32 p,%2,0;\n"
        "  @p cp.async.ca.shared.global [%0],[%1],16;\n"
        "  @!p st.shared.v4.u32 [%0],{0,0,0,0}; }\n"
        :: "r"(s), "l"(src), "r"((int)pred));
}
__device__ __forceinline__ void cp_commit() {
    asm volatile("cp.async.commit_group;\n");
}
__device__ __forceinline__ void cp_wait(int n) {
    if (n == 0) asm volatile("cp.async.wait_group 0;\n");
    else        asm volatile("cp.async.wait_group 1;\n");
}


// ---- Load MMA A-fragment from shared memory (16×64 INT4, row-major packed) ----
// m16n8k64.row.s4 register mapping:
//   groupID = lane/4  (row pair: groupID and groupID+8)
//   localID = lane%4  (column group: 4 bytes = 8 INT4, k=localID*8..)
//   a.x = A[groupID,   localID*4..+4)   (k= 0..31 half)
//   a.y = A[groupID+8, localID*4..+4)
//   a.z = A[groupID,   16+localID*4..+4)  (k=32..63 half)
//   a.w = A[groupID+8, 16+localID*4..+4)
__device__ __forceinline__ uint4 load_a_frag(const uint8_t *base, int stride) {
    int lane   = threadIdx.x % WARP_SZ;
    int row_lo = lane / 4;
    int row_hi = row_lo + 8;
    int col    = (lane % 4) * 4;
    uint4 a;
    a.x = *(const uint32_t*)(base + row_lo * stride + col);
    a.y = *(const uint32_t*)(base + row_hi * stride + col);
    a.z = *(const uint32_t*)(base + row_lo * stride + 16 + col);
    a.w = *(const uint32_t*)(base + row_hi * stride + 16 + col);
    return a;
}

// ---- Load MMA B-fragment from shared memory (8×64 INT4, row-major packed) ----
// m16n8k64.col.s4 register mapping:
//   groupID = lane/4  (weight row 0-7 = output column)
//   localID = lane%4  (k-chunk: 4 bytes = 8 INT4)
__device__ __forceinline__ uint2 load_b_frag(const uint8_t *base, int stride) {
    int lane = threadIdx.x % WARP_SZ;
    int row  = lane / 4;
    int col  = (lane % 4) * 4;
    uint2 b;
    b.x = *(const uint32_t*)(base + row * stride + col);
    b.y = *(const uint32_t*)(base + row * stride + 16 + col);
    return b;
}


// ---- Main GEMM kernel ----
__global__ void gemm_int4_kernel(
    const uint8_t *__restrict__ A,
    const uint8_t *__restrict__ B,
    const half    *__restrict__ scales_A,
    const half    *__restrict__ scales_B,
    half          *__restrict__ C,
    int M, int N, int K, int group_size)
{
    const int bm = blockIdx.y * BLOCK_M;
    const int bn = blockIdx.x * BLOCK_N;
    const int tid    = threadIdx.x;
    const int warpId = tid / WARP_SZ;
    const int laneId = tid % WARP_SZ;
    const int halfK      = K / 2;
    const int num_groups = K / group_size;
    const int num_k_tiles = K / BLOCK_K;

    // Double-buffered shared memory
    extern __shared__ uint8_t smem[];
    const int tileA = BLOCK_M * SMEM_STRIDE;
    const int tileB = BLOCK_N * SMEM_STRIDE;
    uint8_t *sA0 = smem,          *sB0 = smem + tileA;
    uint8_t *sA1 = smem + tileA + tileB, *sB1 = sA1 + tileA;
    uint8_t *sA[2] = {sA0, sA1};
    uint8_t *sB[2] = {sB0, sB1};

    // FP32 accumulators [n_tile][mma_half][4]
    float acc[TILES_N][2][4];
    for (int j = 0; j < TILES_N; j++)
        for (int h = 0; h < 2; h++)
            acc[j][h][0] = acc[j][h][1] = acc[j][h][2] = acc[j][h][3] = 0.f;

    // ---- Tile loader (128-bit cp.async per thread) ----
    // Thread layout: row = tid/2, half_part = tid%2
    // Within a warp (32 threads), threads 2t and 2t+1 both serve row t:
    //   thread 2t   → bytes [kb..kb+16)  of row (bm+t) — first  128-bit chunk
    //   thread 2t+1 → bytes [kb+16..kb+32) of row (bm+t) — second 128-bit chunk
    // This covers all 32 bytes of the 64-INT4 tile slice per row.
    auto load_tile = [&](int kt, int s) {
        int kb = kt * (BLOCK_K / 2);
        {   // Load A tile
            int row      = tid / 2;
            int half_idx = tid % 2;
            bool p = (bm + row < M) && (kb + half_idx * 16 < halfK);
            cp_async_16(sA[s] + row * SMEM_STRIDE + half_idx * 16,
                        A + (size_t)(bm + row) * halfK + kb + half_idx * 16, p);
        }
        {   // Load B tile
            int row      = tid / 2;
            int half_idx = tid % 2;
            bool p = (bn + row < N) && (kb + half_idx * 16 < halfK);
            cp_async_16(sB[s] + row * SMEM_STRIDE + half_idx * 16,
                        B + (size_t)(bn + row) * halfK + kb + half_idx * 16, p);
        }
        cp_commit();
    };

    // Prefetch first tile
    if (num_k_tiles > 0) load_tile(0, 0);

    // ---- Main K-loop ----
    for (int kt = 0; kt < num_k_tiles; kt++) {
        int s = kt & 1;
        if (kt + 1 < num_k_tiles) load_tile(kt + 1, (kt + 1) & 1);
        cp_wait(kt + 1 < num_k_tiles ? 1 : 0);
        __syncthreads();

        // Group index for scales
        int g = (kt * BLOCK_K) / group_size;

        // Activation scales for this warp's rows (two MMA rows: lo and hi)
        int m_lo = bm + warpId * WARP_M + laneId / 4;
        int m_hi = m_lo + 8;
        float sa_lo = (m_lo < M) ? __half2float(scales_A[m_lo * num_groups + g]) : 0.f;
        float sa_hi = (m_hi < M) ? __half2float(scales_A[m_hi * num_groups + g]) : 0.f;

        // A-fragment (one per warp, reused across all N-tiles)
        uint4 af = load_a_frag(sA[s] + warpId * WARP_M * SMEM_STRIDE, SMEM_STRIDE);

        // Process each 16-column N-tile
        #pragma unroll
        for (int nt = 0; nt < TILES_N; nt++) {
            int n_off = nt * 16;

            uint2 bf0 = load_b_frag(sB[s] + (n_off + 0) * SMEM_STRIDE, SMEM_STRIDE);
            uint2 bf1 = load_b_frag(sB[s] + (n_off + 8) * SMEM_STRIDE, SMEM_STRIDE);

            int p0[4] = {0,0,0,0}, p1[4] = {0,0,0,0};
            mma_s4(af, bf0, p0);
            mma_s4(af, bf1, p1);

            // Weight scales for the 4 output columns this thread computes
            int c0 = bn + n_off + (laneId % 4) * 2;
            int c1 = c0 + 1, c2 = c0 + 8, c3 = c2 + 1;
            float sb0 = (c0 < N) ? __half2float(scales_B[c0 * num_groups + g]) : 0.f;
            float sb1 = (c1 < N) ? __half2float(scales_B[c1 * num_groups + g]) : 0.f;
            float sb2 = (c2 < N) ? __half2float(scales_B[c2 * num_groups + g]) : 0.f;
            float sb3 = (c3 < N) ? __half2float(scales_B[c3 * num_groups + g]) : 0.f;

            acc[nt][0][0] += (float)p0[0] * sa_lo * sb0;
            acc[nt][0][1] += (float)p0[1] * sa_lo * sb1;
            acc[nt][0][2] += (float)p0[2] * sa_hi * sb0;
            acc[nt][0][3] += (float)p0[3] * sa_hi * sb1;
            acc[nt][1][0] += (float)p1[0] * sa_lo * sb2;
            acc[nt][1][1] += (float)p1[1] * sa_lo * sb3;
            acc[nt][1][2] += (float)p1[2] * sa_hi * sb2;
            acc[nt][1][3] += (float)p1[3] * sa_hi * sb3;
        }
        __syncthreads();
    }

    // ---- Epilogue: store accumulators to global memory ----
    int m_lo = bm + warpId * WARP_M + laneId / 4;
    int m_hi = m_lo + 8;
    for (int nt = 0; nt < TILES_N; nt++) {
        int c0 = bn + nt * 16 + (laneId % 4) * 2;
        int c1 = c0 + 1, c2 = c0 + 8, c3 = c2 + 1;
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


// ---- Host wrapper ----
torch::Tensor gemm_int4_custom(
    torch::Tensor A_packed, torch::Tensor B_packed,
    torch::Tensor scales_A, torch::Tensor scales_B, int group_size)
{
    TORCH_CHECK(A_packed.is_cuda() && B_packed.is_cuda());
    TORCH_CHECK(A_packed.dtype() == torch::kUInt8);
    int M = A_packed.size(0), K = A_packed.size(1) * 2, N = B_packed.size(0);

    auto C = torch::zeros({M, N},
        torch::TensorOptions().dtype(torch::kHalf).device(A_packed.device()));

    dim3 grid((N + BLOCK_N - 1) / BLOCK_N, (M + BLOCK_M - 1) / BLOCK_M);
    dim3 block(WARP_SZ * NUM_WARPS);
    int  smem = 2 * (BLOCK_M * SMEM_STRIDE + BLOCK_N * SMEM_STRIDE);  // 24 KB

    gemm_int4_kernel<<<grid, block, smem, at::cuda::getCurrentCUDAStream()>>>(
        A_packed.data_ptr<uint8_t>(), B_packed.data_ptr<uint8_t>(),
        reinterpret_cast<const half*>(scales_A.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(scales_B.data_ptr<at::Half>()),
        reinterpret_cast<half*>(C.data_ptr<at::Half>()),
        M, N, K, group_size);

    return C;
}
