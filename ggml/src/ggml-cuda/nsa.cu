#include "nsa.cuh"

// nsa_sel: warps per CUDA block, sharing the keys of one query
#define NSA_SEL_WARPS 8

static __device__ __forceinline__ float nsa_to_f32(const float x) { return x; }
static __device__ __forceinline__ float nsa_to_f32(const half x)  { return __half2float(x); }

// compressed branch and block scores (ggml_nsa_cmp)
// one CUDA block per (KV head, token), one warp per query head of the GQA group, lanes split D
// each warp writes its block probabilities to shared memory, then the group sum runs in a fixed order
// (deterministic, same order as the CPU kernel)
template <int D>
__global__ void __launch_bounds__(1024) nsa_cmp_kernel(
        const float * __restrict__ q, const float * __restrict__ kc, const float * __restrict__ vc,
        const int32_t * __restrict__ pos, float * __restrict__ dst,
        const int64_t sq1, const int64_t sq2, const int64_t sk1, const int64_t sk2, const int64_t sv1, const int64_t sv2,
        const int n_head, const int n_head_kv, const int n_tokens, const int n_blk, const int bs, const float scale) {
    constexpr int R = D / WARP_SIZE;

    const int hk   = blockIdx.x;
    const int i    = blockIdx.y;
    const int warp = threadIdx.x / WARP_SIZE;
    const int lane = threadIdx.x % WARP_SIZE;
    const int G    = blockDim.x / WARP_SIZE;
    const int h    = hk * G + warp;

    // G rows of n_blk probabilities, one per warp
    extern __shared__ float smem[];
    float * p = smem + warp * n_blk;

    const int t     = pos[i];
    const int ic    = t / bs;
    const int n_vis = min((t + 1) / bs, n_blk); // complete blocks only

    float qr[R];
#pragma unroll
    for (int r = 0; r < R; ++r) {
        qr[r] = q[i * sq2 + h * sq1 + lane + r * WARP_SIZE] * scale;
    }

    // stays zero without a complete block, like the reference
    float acc[R] = { 0.0f };

    if (n_vis > 0) {
        float m = -INFINITY;
        for (int c = 0; c < n_vis; ++c) {
            const float * kr = kc + c * sk2 + hk * sk1;
            float part = 0.0f;
#pragma unroll
            for (int r = 0; r < R; ++r) {
                part += qr[r] * kr[lane + r * WARP_SIZE];
            }
            const float s = warp_reduce_sum(part);
            if (lane == 0) {
                p[c] = s;
            }
            m = fmaxf(m, s);
        }
        __syncwarp();

        float sum = 0.0f;
        for (int c = lane; c < n_vis; c += WARP_SIZE) {
            const float e = expf(p[c] - m);
            p[c] = e;
            sum += e;
        }
        sum = warp_reduce_sum(sum);
        const float inv = 1.0f / sum;
        for (int c = lane; c < n_vis; c += WARP_SIZE) {
            p[c] *= inv;
        }
        __syncwarp();

        for (int c = 0; c < n_vis; ++c) {
            const float pc = p[c];
            const float * vr = vc + c * sv2 + hk * sv1;
#pragma unroll
            for (int r = 0; r < R; ++r) {
                acc[r] += pc * vr[lane + r * WARP_SIZE];
            }
        }
    }

    float * o = dst + ((int64_t) i * n_head + h) * D;
#pragma unroll
    for (int r = 0; r < R; ++r) {
        o[lane + r * WARP_SIZE] = acc[r];
    }

    // probabilities of all the warps of the group are needed for the scores
    __syncthreads();

    // the scores plane follows the o_cmp plane
    float * scores = dst + (int64_t) n_tokens * n_head * D;
    for (int c = threadIdx.x; c < n_blk; c += blockDim.x) {
        float v = 0.0f;
        if (c < n_vis) {
            for (int w = 0; w < G; ++w) {
                v += smem[w * n_blk + c];
            }
        }
        // forced blocks: first, current and previous; blocks that are not visible are never chosen
        if (c == 0 || c == ic || c == ic - 1) {
            v = INFINITY;
        } else if (c >= n_vis) {
            v = -INFINITY;
        }
        scores[((int64_t) i * n_head_kv + hk) * n_blk + c] = v;
    }
}

// selected blocks (plane 0) and sliding window (plane 1) (ggml_nsa_sel)
// one CUDA block per (query head, token, plane); the NSA_SEL_WARPS warps take interleaved keys, each with its
// own online softmax, then warp 0 merges the states
// key j is at position j; keys after t are never read, so stale cache cells cannot inject NaN
template <int D, typename KT, typename VT>
__global__ void __launch_bounds__(NSA_SEL_WARPS * WARP_SIZE) nsa_sel_kernel(
        const float * __restrict__ q, const KT * __restrict__ k, const VT * __restrict__ v,
        const int32_t * __restrict__ idx, const int32_t * __restrict__ pos, float * __restrict__ dst,
        const int64_t sq1, const int64_t sq2, const int64_t sk1, const int64_t sk2, const int64_t sv1, const int64_t sv2,
        const int64_t si1, const int64_t si2, const int n_head, const int n_head_kv, const int n_tokens,
        const int n_kv, const int n_sel, const int bs, const int window, const float scale) {
    constexpr int R = D / WARP_SIZE;

    const int h     = blockIdx.x;
    const int i     = blockIdx.y;
    const int plane = blockIdx.z;
    const int warp  = threadIdx.x / WARP_SIZE;
    const int lane  = threadIdx.x % WARP_SIZE;
    const int hk    = h / (n_head / n_head_kv);

    // final (m, s, acc) of each warp
    __shared__ float sm[NSA_SEL_WARPS];
    __shared__ float ss[NSA_SEL_WARPS];
    __shared__ float sacc[NSA_SEL_WARPS][D];

    const int t  = pos[i];
    const int je = min(t + 1, n_kv);

    float qr[R];
#pragma unroll
    for (int r = 0; r < R; ++r) {
        qr[r] = q[i * sq2 + h * sq1 + lane + r * WARP_SIZE] * scale;
    }

    float m = -INFINITY;
    float s = 0.0f;
    float acc[R] = { 0.0f };

    auto add_key = [&](const int j) {
        const KT * kr = k + (int64_t) j * sk2 + hk * sk1;
        float part = 0.0f;
#pragma unroll
        for (int r = 0; r < R; ++r) {
            part += qr[r] * nsa_to_f32(kr[lane + r * WARP_SIZE]);
        }
        const float score = warp_reduce_sum(part);
        if (score > m) {
            const float c = expf(m - score);
            s *= c;
#pragma unroll
            for (int r = 0; r < R; ++r) {
                acc[r] *= c;
            }
            m = score;
        }
        const float w = expf(score - m);
        s += w;
        const VT * vr = v + (int64_t) j * sv2 + hk * sv1;
#pragma unroll
        for (int r = 0; r < R; ++r) {
            acc[r] += w * nsa_to_f32(vr[lane + r * WARP_SIZE]);
        }
    };

    if (plane == 0) {
        const int32_t * bi = idx + (int64_t) i * si2 + hk * si1;
        for (int is = 0; is < n_sel; ++is) {
            const int b = bi[is];
            if (b < 0) {
                continue;
            }
            const int j0 = b * bs;
            const int j1 = min(j0 + bs, je); // j1 <= j0 for a future block
            for (int j = j0 + warp; j < j1; j += NSA_SEL_WARPS) {
                add_key(j);
            }
        }
    } else {
        const int j0 = max(0, t - window + 1);
        for (int j = j0 + warp; j < je; j += NSA_SEL_WARPS) {
            add_key(j);
        }
    }

    if (lane == 0) {
        sm[warp] = m;
        ss[warp] = s;
    }
#pragma unroll
    for (int r = 0; r < R; ++r) {
        sacc[warp][lane + r * WARP_SIZE] = acc[r];
    }
    __syncthreads();

    if (warp == 0) {
        float M = -INFINITY;
        for (int w = 0; w < NSA_SEL_WARPS; ++w) {
            M = fmaxf(M, sm[w]);
        }
        float S = 0.0f;
        float o[R] = { 0.0f };
        for (int w = 0; w < NSA_SEL_WARPS; ++w) {
            // a warp that read no key has m = -inf: avoid exp(-inf - -inf) = NaN
            const float f = sm[w] == -INFINITY ? 0.0f : expf(sm[w] - M);
            S += ss[w] * f;
#pragma unroll
            for (int r = 0; r < R; ++r) {
                o[r] += sacc[w][lane + r * WARP_SIZE] * f;
            }
        }
        const float inv = S > 0.0f ? 1.0f / S : 0.0f;
        float * out = dst + (((int64_t) plane * n_tokens + i) * n_head + h) * D;
#pragma unroll
        for (int r = 0; r < R; ++r) {
            out[lane + r * WARP_SIZE] = o[r] * inv;
        }
    }
}

void ggml_cuda_op_nsa_cmp(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * q   = dst->src[0];
    const ggml_tensor * kc  = dst->src[1];
    const ggml_tensor * vc  = dst->src[2];
    const ggml_tensor * pos = dst->src[3];

    const int   bs    = ggml_get_op_params_i32(dst, 0);
    const float scale = ggml_get_op_params_f32(dst, 1);

    const int D         = q->ne[0];
    const int n_head    = q->ne[1];
    const int n_tokens  = q->ne[2];
    const int n_head_kv = kc->ne[1];
    const int n_blk     = kc->ne[2];
    const int G         = n_head / n_head_kv;

    const dim3   grid(n_head_kv, n_tokens);
    const dim3   block(G * WARP_SIZE);
    const size_t smem = (size_t) G * n_blk * sizeof(float);
    cudaStream_t stream = ctx.stream();

    const float * q_d  = (const float *) q->data;
    const float * kc_d = (const float *) kc->data;
    const float * vc_d = (const float *) vc->data;
    const int32_t * p_d = (const int32_t *) pos->data;
    float * dst_d = (float *) dst->data;

    // strides in elements
    const int64_t sq1 = q->nb[1] / sizeof(float);
    const int64_t sq2 = q->nb[2] / sizeof(float);
    const int64_t sk1 = kc->nb[1] / sizeof(float);
    const int64_t sk2 = kc->nb[2] / sizeof(float);
    const int64_t sv1 = vc->nb[1] / sizeof(float);
    const int64_t sv2 = vc->nb[2] / sizeof(float);

    switch (D) {
        case 64:
            nsa_cmp_kernel<64><<<grid, block, smem, stream>>>(
                q_d, kc_d, vc_d, p_d, dst_d, sq1, sq2, sk1, sk2, sv1, sv2,
                n_head, n_head_kv, n_tokens, n_blk, bs, scale);
            break;
        case 128:
            nsa_cmp_kernel<128><<<grid, block, smem, stream>>>(
                q_d, kc_d, vc_d, p_d, dst_d, sq1, sq2, sk1, sk2, sv1, sv2,
                n_head, n_head_kv, n_tokens, n_blk, bs, scale);
            break;
        default:
            GGML_ABORT("nsa_cmp: unsupported head size %d", D);
    }
}

template <int D, typename KT, typename VT>
static void nsa_sel_launch(const ggml_tensor * dst, cudaStream_t stream) {
    const ggml_tensor * q   = dst->src[0];
    const ggml_tensor * k   = dst->src[1];
    const ggml_tensor * v   = dst->src[2];
    const ggml_tensor * idx = dst->src[3];
    const ggml_tensor * pos = dst->src[4];

    const int   bs     = ggml_get_op_params_i32(dst, 0);
    const int   window = ggml_get_op_params_i32(dst, 1);
    const float scale  = ggml_get_op_params_f32(dst, 2);

    const int n_head    = q->ne[1];
    const int n_tokens  = q->ne[2];
    const int n_head_kv = k->ne[1];
    const int n_kv      = k->ne[2];
    const int n_sel     = idx->ne[0];

    const dim3 grid(n_head, n_tokens, 2);
    const dim3 block(NSA_SEL_WARPS * WARP_SIZE);
    // strides in elements; idx can be a strided view (ggml_argsort_top_k)
    nsa_sel_kernel<D, KT, VT><<<grid, block, 0, stream>>>(
        (const float *) q->data, (const KT *) k->data, (const VT *) v->data,
        (const int32_t *) idx->data, (const int32_t *) pos->data, (float *) dst->data,
        q->nb[1] / sizeof(float), q->nb[2] / sizeof(float), k->nb[1] / sizeof(KT), k->nb[2] / sizeof(KT),
        v->nb[1] / sizeof(VT), v->nb[2] / sizeof(VT), idx->nb[1] / sizeof(int32_t), idx->nb[2] / sizeof(int32_t),
        n_head, n_head_kv, n_tokens, n_kv, n_sel, bs, window, scale);
}

template <int D>
static void nsa_sel_dispatch_kv(const ggml_tensor * dst, cudaStream_t stream) {
    const bool k16 = dst->src[1]->type == GGML_TYPE_F16;
    const bool v16 = dst->src[2]->type == GGML_TYPE_F16;
    if (k16 && v16) {
        nsa_sel_launch<D, half, half>(dst, stream);
    } else if (k16) {
        nsa_sel_launch<D, half, float>(dst, stream);
    } else if (v16) {
        nsa_sel_launch<D, float, half>(dst, stream);
    } else {
        nsa_sel_launch<D, float, float>(dst, stream);
    }
}

void ggml_cuda_op_nsa_sel(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const int D = dst->src[0]->ne[0];
    switch (D) {
        case 64:
            nsa_sel_dispatch_kv<64>(dst, ctx.stream());
            break;
        case 128:
            nsa_sel_dispatch_kv<128>(dst, ctx.stream());
            break;
        default:
            GGML_ABORT("nsa_sel: unsupported head size %d", D);
    }
}

bool ggml_cuda_nsa_supported(const ggml_tensor * op) {
    const ggml_tensor * q = op->src[0];
    const ggml_tensor * k = op->src[1];
    const ggml_tensor * v = op->src[2];

    if (q->type != GGML_TYPE_F32 || (q->ne[0] != 64 && q->ne[0] != 128)) {
        return false;
    }
    // nsa_cmp runs one warp per query head of the group
    const int64_t G = q->ne[1] / k->ne[1];
    if (q->ne[1] % k->ne[1] != 0 || G > 32) {
        return false;
    }
    if (op->op == GGML_OP_NSA_CMP) {
        // block means are F32; dynamic shared memory is limited to 48 KB without opt-in (~49k tokens for G = 16)
        const bool f32 = k->type == GGML_TYPE_F32 && v->type == GGML_TYPE_F32;
        return f32 && G * k->ne[2] * sizeof(float) <= 48 * 1024;
    }
    const bool k_ok = k->type == GGML_TYPE_F16 || k->type == GGML_TYPE_F32;
    const bool v_ok = v->type == GGML_TYPE_F16 || v->type == GGML_TYPE_F32;
    return k_ok && v_ok;
}
