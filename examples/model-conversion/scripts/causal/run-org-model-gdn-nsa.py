#!/usr/bin/env python3
"""
Pure PyTorch (fp32, CPU) reference for Idiap Gated-DeltaNet + NSA checkpoints
(model_type "hybrid_gated_deltanet", e.g. Idiap/gated-deltanet-nsa-0.4B-10B).

The original model code (github.com/idiap/hybrid-linear-sparse-attention, a fork of
flash-linear-attention) needs Triton and FlashAttention-2, so it cannot run on CPU or on
pre-Ampere GPUs. This script re-implements the forward pass:
  - gated delta rule recurrence: copied from fla-org fla/ops/gated_delta_rule/naive.py
  - NSA top-k and selected attention: copied from idiap fla/ops/nsa/naive.py
    (varlen branches removed, einops repeat replaced by repeat_interleave)
  - replaced by equivalent torch code: mean pooling (Triton), compressed attention
    (flex_attention), sliding window attention (flash_attn_func)

Output files use the same format as run-org-model.py (utils.common.save_output_data),
so compare-logits.py and check-nmse.py work unchanged.
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import load_file
from transformers import AutoTokenizer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from utils.common import save_output_data


# ---------------------------------------------------------------------------
# copied from fla-org/flash-linear-attention fla/ops/gated_delta_rule/naive.py
# ---------------------------------------------------------------------------
def naive_recurrent_gated_delta_rule(q, k, v, beta, g, scale=None, initial_state=None, output_final_state=False):
    q, k, v, beta, g = map(lambda x: x.transpose(1, 2).contiguous().to(torch.float32), [q, k, v, beta, g])
    B, H, T, K, V = *k.shape, v.shape[-1]
    o = torch.zeros(B, H, T, V).to(v)
    h = torch.zeros(B, H, K, V).to(v)
    if initial_state is not None:
        h = initial_state.to(torch.float32)
    if scale is None:
        scale = 1 / (q.shape[-1] ** 0.5)
    q = q * scale

    for i in range(T):
        b_q = q[:, :, i]
        b_k = k[:, :, i]
        b_v = v[:, :, i].clone()
        h = h.clone() * g[:, :, i].exp()[..., None, None]
        b_beta = beta[:, :, i]
        b_v = b_v - (h.clone() * b_k[..., None]).sum(-2)
        b_v = b_v * b_beta[..., None]
        h = h.clone() + b_k.unsqueeze(-1) * b_v.unsqueeze(-2)
        o[:, :, i] = torch.einsum('bhd,bhdm->bhm', b_q, h)

    if not output_final_state:
        h = None
    o = o.transpose(1, 2).contiguous()
    return o, h


# ---------------------------------------------------------------------------
# copied from idiap/hybrid-linear-sparse-attention fla/ops/nsa/naive.py
# (non-varlen path only; repeat(x, 'b t h d -> b t (h g) d') == x.repeat_interleave(g, dim=2))
# ---------------------------------------------------------------------------
def naive_nsa_sel(q, k, v, block_indices, block_size=64, scale=None):
    dtype = q.dtype
    G = q.shape[2] // k.shape[2]
    BS = block_size
    k, v, block_indices = (x.repeat_interleave(G, dim=2) for x in (k, v, block_indices))
    q, k, v = map(lambda x: x.float(), (q, k, v))
    B = q.shape[0]

    o = torch.zeros_like(v)
    Tq = Tk = q.shape[1]
    for i in range(B):
        q_b, k_b, v_b, i_b = q[i], k[i], v[i], block_indices[i]
        i_b = i_b.unsqueeze(-1) * BS + i_b.new_tensor(range(BS))
        # [T, S*BS, HQ]
        i_b = i_b.view(Tq, block_indices.shape[2], -1).transpose(1, 2)
        for i_q in range(Tq):
            # [HQ, D]
            q_i = q_b[i_q] * scale
            # [S*BS, HQ]
            i_i = i_b[i_q]
            # [S*BS, HQ, -1]
            k_i, v_i = map(lambda x: x.gather(0, i_i.clamp(0, Tk-1).unsqueeze(-1).expand(*i_i.shape, x.shape[-1])),
                           (k_b, v_b))
            # [S*BS, HQ]
            attn = torch.einsum('h d, n h d -> n h', q_i, k_i).masked_fill(
                torch.logical_or(i_i > i_q, i_i < 0), float('-inf')).softmax(0)
            o[i, i_q] = torch.einsum('n h, n h v -> h v', attn, v_i)

    return o.to(dtype)


def naive_nsa_topk(q, k_cmp, block_counts, block_size, scale):
    B, Tq, Hq, _ = q.shape
    Hkv = k_cmp.shape[2]
    G = Hq // Hkv
    k_cmp = k_cmp.repeat_interleave(G, dim=2)

    device = q.device
    Tc = k_cmp.shape[1]
    S = int(block_counts)
    result = torch.full((B, Tq, Hkv, S), -1, device=device, dtype=torch.long)

    for i in range(B):
        q_b, k_b = q[i], k_cmp[i]

        logits = torch.einsum('t h d, s h d -> t h s', q_b, k_b) * scale  # [Tq, Hq, Tc]
        logits = logits.reshape(Tq, Hkv, G, Tc)
        t = torch.arange(Tq, device=device).unsqueeze(1)
        s = torch.arange(Tc, device=device).unsqueeze(0)
        block_last_pos = (s + 1) * block_size - 1
        base_allow = (block_last_pos <= t)  # [Tq,Tc]

        i_qb = (t // block_size)                                                 # [Tq,1]
        is_current_block = (s == i_qb) | (s == 0) | (s == i_qb - 1)              # [Tq,Tc]
        logits = logits.masked_fill(~base_allow[:, None, None, :], float("-inf"))
        allow = base_allow | is_current_block  # [Tq,Tc]

        probs_q = torch.softmax(logits, dim=-1)  # [Tq, Hkv, G, Tc]
        probs_q = torch.nan_to_num(probs_q, nan=0.0)  # rows with no valid blocks -> 0
        scores = probs_q.mean(dim=2)  # [Tq, Hkv, Tc]
        scores = torch.where(is_current_block[:, None, :], 1.0, scores)

        desired_k = torch.full((Tq, Hkv), S, dtype=torch.long, device=device)

        _, topi = torch.topk(scores, k=min(S, Tc), dim=-1)                             # [Tq,Hkv,S]

        # Validate selections against allow mask; pad with -1 where invalid or beyond quota
        allow_kv = allow[:, None, :].expand(Tq, Hkv, Tc)                      # [Tq,Hkv,Tc]
        sel_allowed = torch.gather(allow_kv.long(), dim=-1, index=topi).bool()    # [Tq,Hkv,S]

        idx = torch.arange(S, device=device).view(1, 1, S)
        within_quota = (idx < desired_k.unsqueeze(-1))[:, :, :Tc]                              # [Tq,Hkv,S]

        keep = sel_allowed & within_quota
        out = torch.full_like(topi, fill_value=-1)                                # pad with -1
        out = torch.where(keep, topi, out)

        if S > Tc:
            out = torch.cat((out, torch.full((Tq, Hkv, S - Tc), -1,
                                             device=device, dtype=topi.dtype)), dim=-1)
        result[i] = out
    return result


# ---------------------------------------------------------------------------
# replacements for the Triton / flex_attention / flash_attn parts
# ---------------------------------------------------------------------------
def mean_pooling(x, block_size):
    # fla/ops/utils/pooling.py: mean over non-overlapping blocks along T,
    # a partial last block is averaged over its real length. x: [T, H, D]
    T = x.shape[0]
    n = (T + block_size - 1) // block_size
    xs = F.pad(x, (0, 0, 0, 0, 0, n * block_size - T)).view(n, block_size, *x.shape[1:]).sum(1)
    cnt = torch.clamp(T - torch.arange(n) * block_size, max=block_size).to(x.dtype)
    return xs / cnt[:, None, None]


def nsa_cmp(q, k_cmp, v_cmp, block_size, scale):
    # replaces naive_nsa_cmp (flex_attention): query t sees compressed block c iff
    # t >= (c + 1) * block_size - 1, i.e. only complete blocks; fully masked rows give 0
    T, HQ, _ = q.shape
    NC, H, _ = k_cmp.shape
    G = HQ // H
    kc, vc = k_cmp.repeat_interleave(G, dim=1), v_cmp.repeat_interleave(G, dim=1)
    s = torch.einsum('thd,chd->thc', q, kc) * scale
    t = torch.arange(T)[:, None]
    c = torch.arange(NC)[None, :]
    allow = t >= (c + 1) * block_size - 1
    s = s.masked_fill(~allow[:, None, :], float('-inf'))
    p = torch.softmax(s, dim=-1).nan_to_num(0.0)
    return torch.einsum('thc,chd->thd', p, vc)


def swa_attn(q, k, v, window_size, scale):
    # replaces flash_attn_func(causal=True, window_size=(window_size-1, 0)): keys j in [t-window_size+1, t]
    T, HQ, _ = q.shape
    G = HQ // k.shape[1]
    kk, vv = k.repeat_interleave(G, dim=1), v.repeat_interleave(G, dim=1)
    s = torch.einsum('thd,shd->ths', q, kk) * scale
    t = torch.arange(T)[:, None]
    j = torch.arange(T)[None, :]
    allow = (j <= t) & (j > t - window_size)
    s = s.masked_fill(~allow[:, None, :], float('-inf'))
    return torch.einsum('ths,shd->thd', torch.softmax(s, dim=-1), vv)


# ---------------------------------------------------------------------------
# model
# ---------------------------------------------------------------------------
def rms_norm(x, w, eps):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * w


def short_conv(x, w):
    # fla ShortConvolution: depthwise causal conv1d, no bias, SiLU. x: [T, C], w: [C, 1, K]
    K = w.shape[-1]
    y = F.conv1d(F.pad(x.t().unsqueeze(0), (K - 1, 0)), w, groups=w.shape[0])
    return F.silu(y[0].t())


def l2norm(x):
    # use_qk_l2norm_in_kernel=True in fla kernels: x / sqrt(sum(x^2) + 1e-6)
    return x / torch.sqrt((x * x).sum(-1, keepdim=True) + 1e-6)


def to16(x, cast16):
    # the idiap layers cast fp32 q/k/v to fp16 before the kernels and the outputs back to fp32
    return x.half().float() if cast16 else x


def gdn_layer(x, p, cfg, cast16):
    T = x.shape[0]
    H = cfg['num_heads']
    Dk = cfg['head_dim']
    Dv = int(Dk * cfg['expand_v'])
    q = short_conv(x @ p['q_proj.weight'].t(), p['q_conv1d.weight']).view(T, H, Dk)
    k = short_conv(x @ p['k_proj.weight'].t(), p['k_conv1d.weight']).view(T, H, Dk)
    v = short_conv(x @ p['v_proj.weight'].t(), p['v_conv1d.weight']).view(T, H, Dv)
    beta = torch.sigmoid(x @ p['b_proj.weight'].t())                                       # [T, H]
    g = -p['A_log'].exp() * F.softplus(x @ p['a_proj.weight'].t() + p['dt_bias'])          # [T, H]
    q, k, v = (to16(t, cast16) for t in (q, k, v))
    q, k = l2norm(q), l2norm(k)
    o, _ = naive_recurrent_gated_delta_rule(q[None], k[None], v[None], beta[None], g[None])
    o = to16(o[0], cast16)                                                                 # [T, H, Dv]
    gate = (x @ p['g_proj.weight'].t()).view(T, H, Dv)
    o = rms_norm(o, p['o_norm.weight'], cfg['norm_eps']) * F.silu(gate)
    return o.reshape(T, H * Dv) @ p['o_proj.weight'].t()


def rope_neox(x, theta):
    # fla RotaryEmbedding, interleaved=False (rotate_half on the two halves). x: [T, H, D]
    T, _, D = x.shape
    inv_freq = 1.0 / (theta ** (torch.arange(0, D, 2, dtype=torch.float32) / D))
    ang = torch.arange(T, dtype=torch.float32)[:, None] * inv_freq[None, :]                # [T, D/2]
    cos, sin = ang.cos()[:, None, :], ang.sin()[:, None, :]
    x1, x2 = x[..., :D // 2], x[..., D // 2:]
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)


def nsa_layer(x, p, a, cast16, stats=None):
    T = x.shape[0]
    HQ, H, D = a['num_heads'], a['num_kv_heads'], a['head_dim']
    BS, S, W = a['block_size'], a['block_counts'], a['window_size']
    q = (x @ p['q_proj.weight'].t()).view(T, HQ, D)
    k = (x @ p['k_proj.weight'].t()).view(T, H, D)
    v = (x @ p['v_proj.weight'].t()).view(T, H, D)
    g = (x @ p['g_proj.weight'].t()).view(T, HQ, 3)          # index h*3 + j, j = cmp, slc, swa
    q, k = rope_neox(q, a['rope_theta']), rope_neox(k, a['rope_theta'])
    q, k, v = (to16(t, cast16) for t in (q, k, v))
    scale = D ** -0.5
    g_cmp, g_slc, g_swa = torch.sigmoid(g).unbind(-1)        # [T, HQ] each

    k_cmp, v_cmp = mean_pooling(k, BS), mean_pooling(v, BS)
    o_cmp = nsa_cmp(q, k_cmp, v_cmp, BS, scale)
    idx = naive_nsa_topk(q[None], k_cmp[None], S, BS, scale)          # [1, T, H, S]
    o_slc = naive_nsa_sel(q[None], k[None], v[None], idx, BS, scale)[0]
    o_swa = swa_attn(q, k, v, W, scale)
    if stats is not None:
        stats.append(idx[0])

    o = o_slc * g_slc[..., None]
    o = torch.addcmul(o, o_cmp, g_cmp[..., None])
    o = torch.addcmul(o, o_swa, g_swa[..., None])
    o = to16(o, cast16)
    return o.reshape(T, HQ * D) @ p['o_proj.weight'].t()


def forward(ids, W, cfg, cast16=False, hidden=None, sel=None):
    # full-sequence forward, returns logits [T, n_vocab]
    eps = cfg['norm_eps']
    attn = cfg['attn']
    x = W['model.embeddings.weight'][ids]
    if hidden is not None:
        hidden.append(x.clone())
    for il in range(cfg['num_hidden_layers']):
        pre = f'model.layers.{il}.'
        p = {n[len(pre) + 5:]: t for n, t in W.items() if n.startswith(pre + 'attn.')}
        h = rms_norm(x, W[pre + 'attn_norm.weight'], eps)
        if il in attn['layers']:
            h = nsa_layer(h, p, attn, cast16, sel)
        else:
            h = gdn_layer(h, p, cfg, cast16)
        x = x + h
        h = rms_norm(x, W[pre + 'mlp_norm.weight'], eps)
        h = F.silu(h @ W[pre + 'mlp.gate_proj.weight'].t()) * (h @ W[pre + 'mlp.up_proj.weight'].t())
        x = x + h @ W[pre + 'mlp.down_proj.weight'].t()
        if hidden is not None:
            hidden.append(x.clone())
    x = rms_norm(x, W['model.norm.weight'], eps)
    return x @ W['model.embeddings.weight'].t()            # tied embeddings


def get_prompt(args):
    if args.prompt_file:
        with open(args.prompt_file, encoding='utf-8') as f:
            return f.read()
    if os.getenv("MODEL_TESTING_PROMPT"):
        return os.getenv("MODEL_TESTING_PROMPT")
    return "Hello, my name is"


def perplexity(path, tokenizer, W, cfg, args):
    # same convention as llama-perplexity: BOS replaces the first token of each n_ctx chunk,
    # positions [n_ctx/2, n_ctx - 1) predict the next token
    with open(path, encoding='utf-8') as f:
        ids = tokenizer(f.read()).input_ids
    n_ctx = args.ppl_ctx
    first = n_ctx // 2
    nll, cnt = 0.0, 0
    for c in range(args.ppl_chunks):
        chunk = torch.tensor(ids[c * n_ctx:(c + 1) * n_ctx])
        chunk[0] = tokenizer.bos_token_id
        lp = torch.log_softmax(forward(chunk, W, cfg, args.cast_fp16), dim=-1)
        nll -= lp[first:n_ctx - 1].gather(1, chunk[first + 1:, None]).sum().item()
        cnt += n_ctx - 1 - first
        print(f"[{c + 1}] ppl = {np.exp(nll / cnt):.4f}", flush=True)
    return np.exp(nll / cnt)


def main():
    parser = argparse.ArgumentParser(description="fp32 CPU reference for hybrid_gated_deltanet (GDN + NSA)")
    parser.add_argument("--model-path", "-m", help="Path to the HF model directory")
    parser.add_argument("--prompt-file", "-f", help="Optional prompt file")
    parser.add_argument("--tokens-file", help="int32 token ids to use instead of tokenizing the prompt (e.g. llamacpp-*-tokens.bin)")
    parser.add_argument("--cast-fp16", action="store_true", help="emulate the fp16 cast of q/k/v in the idiap layers")
    parser.add_argument("--dump-dir", help="save all-position logits, per-layer hidden states and NSA block indices (.npy)")
    parser.add_argument("--n-predict", type=int, default=0, help="greedy generation length")
    parser.add_argument("--ppl-file", help="text file for a perplexity sanity check")
    parser.add_argument("--ppl-ctx", type=int, default=512)
    parser.add_argument("--ppl-chunks", type=int, default=4)
    args = parser.parse_args()

    model_path = os.environ.get("MODEL_PATH", args.model_path)
    if model_path is None:
        print("Error: Model path must be specified either via --model-path argument or MODEL_PATH environment variable")
        sys.exit(1)

    torch.manual_seed(0)
    with open(os.path.join(model_path, "config.json")) as f:
        cfg = json.load(f)
    assert cfg['model_type'] == 'hybrid_gated_deltanet' and cfg['attn']['attn_type'] == 'nsa'
    W = {n: t.float() for n, t in load_file(os.path.join(model_path, "model.safetensors")).items()}
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model_name = os.path.basename(os.path.normpath(model_path))

    with torch.no_grad():
        if args.ppl_file:
            print(f"Final estimate: PPL = {perplexity(args.ppl_file, tokenizer, W, cfg, args):.4f}")
            return

        prompt = get_prompt(args)
        if args.tokens_file:
            ids = torch.from_numpy(np.fromfile(args.tokens_file, dtype=np.int32).astype(np.int64))
        else:
            ids = tokenizer(prompt, return_tensors="pt").input_ids[0]
        token_ids = ids.tolist()
        print(f"Input tokens: {token_ids}")
        print(f"Input text: {repr(prompt)}")
        print(f"Tokenized: {tokenizer.convert_ids_to_tokens(ids)}")

        hidden, sel = [], []
        logits = forward(ids, W, cfg, args.cast_fp16, hidden, sel)
        last_logits = logits[-1].numpy()
        print(f"Logits shape: {tuple(logits.shape)}")
        top_indices = np.argsort(last_logits)[-5:][::-1]
        print("Top 5 predictions:")
        for idx in top_indices:
            print(f"  Token {idx} ({repr(tokenizer.decode([idx]))}): {last_logits[idx]:.6f}")

        suffix = "-cast16" if args.cast_fp16 else ""
        save_output_data(last_logits, token_ids, prompt, model_name, type_suffix=suffix)

        if args.dump_dir:
            os.makedirs(args.dump_dir, exist_ok=True)
            base = os.path.join(args.dump_dir, f"pytorch-{model_name}{suffix}")
            np.save(f"{base}-all-logits.npy", logits.numpy())                       # [T, n_vocab]
            np.save(f"{base}-hidden.npy", torch.stack(hidden).numpy())              # [n_layer + 1, T, n_embd]
            np.save(f"{base}-nsa-idx.npy", torch.stack(sel).numpy())                # [n_nsa_layer, T, H, S]
            print(f"Dumps saved to {base}-*.npy")

        if args.n_predict > 0:
            out = ids
            for _ in range(args.n_predict):
                nxt = forward(out, W, cfg, args.cast_fp16)[-1].argmax()
                out = torch.cat([out, nxt[None]])
            print(f"Greedy tokens: {out[len(ids):].tolist()}")
            print(f"Greedy text: {repr(tokenizer.decode(out))}")


if __name__ == "__main__":
    main()
