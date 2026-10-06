# Idiap Gated-DeltaNet + NSA 0.4B in qvac-fabric-llm.cpp

Native support for [Idiap/gated-deltanet-nsa-0.4B-10B](https://huggingface.co/Idiap/gated-deltanet-nsa-0.4B-10B)
(paper [arXiv 2510.20787](https://arxiv.org/abs/2510.20787)) in ggml/GGUF, on CPU and CUDA.
Native Sparse Attention (NSA) is implemented with two new first-class ggml ops, `GGML_OP_NSA_CMP` and
`GGML_OP_NSA_SEL`, used for both prompt processing and token-by-token generation.

## Summary

- The model converts to GGUF (F32, F16, Q8_0, Q4_K_M) and generates the same greedy text as the original
  implementation on CPU and on CUDA (RTX 2080 Ti), with no CPU fallback on the GPU path.
- Logits match an fp32 reference built from the official naive functions with NMSE ~1e-11, and the original
  Triton implementation (run on an Ampere GPU) within its own fp16 noise (top-1 identical, KL ~3e-7).
- CUDA: 10.6k t/s prompt processing and 280 t/s generation (F16, 512-token prompt); NSA ops take < 5 % of
  decode time at short context.

## Requirements of the assignment

| # | Requirement | Status | Evidence |
|---|---|---|---|
| 1 | Convert and load through GGUF | done | `conversion/gdn_nsa.py`, [tokenizer check](evidence/tokenizer-check.txt) |
| 2 | Real output on CPU | done | greedy text and logits below, [comparisons](evidence/reference-and-compare.txt) |
| 3 | Real output on one GPU backend, no silent CPU fallback | done (CUDA) | [node placement](evidence/placement-cuda.txt), [buffers](evidence/memory-cuda-buffers.txt) |
| 4 | Validate against the reference, methodology and tolerances | done | [Correctness](#correctness) |
| 5 | Automated tests and reproducible evidence | done | [tests](evidence/tests.txt), [reproduce.sh](reproduce.sh) |
| 6 | CPU/GPU performance, memory, limitations | done | [Performance](#performance), [Limitations](#limitations-and-next-steps) |
| bonus | Quantization | done (Q8_0, Q4_K_M, F16) | [quality](evidence/quality-quantization.txt) |
| bonus | Other GPU backends (Vulkan, Metal) | not done | NSA ops have CUDA kernels only |
| bonus | Mobile | not done | see [next steps](#limitations-and-next-steps) |

## The model

24 layers, hidden size 1024, 358 M parameters (F32 checkpoint), Mistral-7B-v0.1 vocabulary (32000, shipped as
`tokenizer.json` only).

- Even layers: Gated DeltaNet, 8 heads of 128, short causal conv (kernel 4) on q/k/v, L2-normalized q/k,
  gate `g = -exp(A_log) * softplus(a + dt_bias)`, `beta = sigmoid(b)`, gated RMSNorm on the output.
- Odd layers: NSA, 32 query heads, 2 KV heads (GQA group of 16), head size 64, NeoX RoPE (theta 10000).
  For a query at position t, three branches are mixed by per-head sigmoid gates:
  `o = g_slc * o_slc + g_cmp * o_cmp + g_swa * o_swa`
  - compressed: attention over the means of the complete blocks of 64 keys (parameter-free mean pooling in the
    checkpoint, the paper describes an MLP);
  - selected: attention over the 64 keys of each of the 16 best blocks; the block score is the compressed-branch
    probability summed over the 16 query heads of the KV head; blocks 0, t/64 and t/64 - 1 are always selected;
  - sliding window: attention over keys [t - 511, t].
- Every block: pre-RMSNorm, mixer, residual, RMSNorm, SwiGLU MLP (2816), residual. Tied embeddings.

The `hybrid_gated_deltanet` model type is not in fla-org/flash-linear-attention: it lives in the fork
[idiap/hybrid-linear-sparse-attention](https://github.com/idiap/hybrid-linear-sparse-attention), used as the
reference implementation (commit b786cbc6).

## What was implemented

Branch `gdn-nsa`, on top of tetherto/qvac-fabric-llm.cpp master `7bb2161c6`.

| Area | Files | Content |
|---|---|---|
| ggml ops | `ggml/include/ggml.h`, `ggml/src/ggml.c` | `ggml_nsa_cmp`, `ggml_nsa_sel` (API, shape checks, op params) |
| CPU kernels | `ggml/src/ggml-cpu/ops.cpp`, `ggml-cpu.c`, `ops.h` | multi-threaded kernels, work buffers |
| CUDA kernels | `ggml/src/ggml-cuda/nsa.cu`, `nsa.cuh`, `ggml-cuda.cu` | kernels, dispatch, `supports_op` |
| RPC | `ggml/include/ggml-rpc.h` | protocol version 109 -> 110 (new ops) |
| Conversion | `conversion/gdn_nsa.py`, `conversion/__init__.py`, `gguf-py/gguf/constants.py`, `gguf_writer.py` | HF -> GGUF, new keys `attention.nsa.{block_size,block_count,window_size}` |
| Model | `src/models/gdn-nsa.cpp`, `src/models/models.h`, `src/llama-arch.*`, `src/llama-hparams.h`, `src/llama-model.cpp`, `src/llama-context.cpp`, `src/llama-model-saver.cpp` | arch `gdn-nsa`, hparams, tensors, graph, hybrid memory |
| Shared fix | `src/llama-graph.cpp` | skip the KQ mask in the hybrid memory input when the graph does not use it |
| Tests | `tests/test-backend-ops.cpp`, `tests/test-llama-archs.cpp` | 24 NSA op cases, synthetic gdn-nsa model |
| Reference tools | `examples/model-conversion/scripts/causal/run-org-model-gdn-nsa*.py`, `compare-logits-gdn-nsa.py` | fp32 CPU reference, original implementation runner, logits comparison |

### NSA as two ggml ops

```
q (after RoPE) ----------------------------------------+-------------------------+
K/V cache view --> block means (mean, pad: existing) --> ggml_nsa_cmp --> o_cmp  |
                                                          |                      |
                                                          +-> scores --> ggml_argsort_top_k --> idx
                                                                                  |
q, K/V cache view, idx -----------------------------------------------------> ggml_nsa_sel --> o_slc, o_swa
o = g_slc * o_slc + g_cmp * o_cmp + g_swa * o_swa   (existing mul/add)
```

- `ggml_nsa_cmp(q, k_cmp, v_cmp, pos, block_size, scale)`: compressed attention over the complete blocks
  (`c < (t + 1) / block_size`) and block scores (softmax probabilities summed over the query heads of each KV
  head, `+INF` for the forced blocks, `-INF` for the other blocks that are not visible). Output: one tensor with
  two contiguous planes, like `ggml_gated_delta_net` packs output and state.
- `ggml_nsa_sel(q, k, v, idx, pos, block_size, window, scale)`: attention over the keys of the selected blocks
  (plane 0) and over the sliding window (plane 1), one online softmax per branch. Keys after the query position
  are skipped, never read, so stale cache cells cannot inject NaN.
- The discrete block choice is a separate, existing op. Backend tests compare continuous outputs only: a
  near-tie in the scores cannot flip a selection between CPU and GPU inside a tested op.
- Prompt processing and generation use the same ops; causality comes from the query positions.
- K/V come from the standard KV cache of the hybrid memory (recurrent state for GDN layers, KV cache for NSA
  layers, V not transposed). The full K/V history is kept: NSA reduces compute, not KV memory.

### Gated DeltaNet layers

The checkpoint stores separate q/k/v projections and convolutions. The converter concatenates them into the
qwen3next layout (`attn_qkv`, `ssm_conv1d`; the depthwise convolution makes channel concatenation exact),
interleaves `b`/`a` into `ssm_ba` and stores `ssm_a = -exp(A_log)`. The graph reuses the qwen3next linear
attention path (`ggml_ssm_conv`, `ggml_l2_norm`, `ggml_gated_delta_net`, gated RMSNorm).

### Tokenizer

The checkpoint has no `tokenizer.model`. The generic `LlamaHfVocab` path gives every token the same score, so
the SentencePiece tokenizer of llama.cpp merged in arbitrary order (1807 tokens instead of 1789 on the test
prompt). The converter now sets `score = -(rank of the BPE merge that creates the token)`. llama.cpp tokens
are identical to the HF tokenizer on the full wikitext-2 test set (328,880 tokens).

## Correctness

### Methodology

- Two references:
  - the original implementation (Triton + FlashAttention-2), run on an NVIDIA A40 (RunPod) with the versions it
    pins (torch 2.7.0, transformers 4.49.0, flash-attn 2.8.3); the local RTX 2080 Ti (Turing) and a CPU-only
    torch cannot run it ([run log](evidence/official-run.txt), [versions and results](evidence/official-run.json));
  - an fp32 CPU re-implementation built from the official naive functions (fla `naive_recurrent_gated_delta_rule`,
    idiap `naive_nsa_topk` and `naive_nsa_sel`), with torch code only for the Triton/flash-attn parts (mean
    pooling, compressed attention, sliding window). It writes the same files as `run-org-model.py`.
- Same input everywhere: identical token ids (tokenizer check above).
- A 1789-token wikitext prompt: 27 complete blocks, so the top-16 selection and the 512-key window are
  really sparse. A short prompt does not exercise NSA (no complete block).
- Prefill, chunked prefill (ubatch 512) and token-by-token decode (ubatch 1, 1789 graphs of one token).
- Metrics in float64: max abs logit difference, NMSE, KL, top-1 agreement; perplexity with the
  llama-perplexity convention; greedy generation.

### Tolerances

The original implementation casts q/k/v to fp16 inside the GDN and NSA layers, so its own fp16 rounding sets
the noise floor. Measured on the 1789 positions, fp32 reference vs the same reference with the fp16 cast:
max |d logit| 0.019, KL mean 1.4e-7 (max 6.4e-5), top-1 100 %, selected blocks identical for 99.99 % of
(layer, token, KV head).

- llama.cpp F32 vs fp32 reference: NMSE < 1e-6 required (the "essentially identical" level of
  `check-nmse.py`); `compare-logits-gdn-nsa.py --max-nmse 1e-6` fails otherwise.
- llama.cpp vs the original implementation: same order as its fp16 noise.

### Results

Last-token logits, 1789-token prompt, F32 weights:

| llama.cpp run | vs | max abs diff | NMSE | KL | top-1 |
|---|---|---:|---:|---:|---|
| CPU, KV F32, ubatch 512 | fp32 reference | 9.3e-5 | 2.0e-11 | 2.8e-10 | same |
| CPU, KV F32, ubatch 1 (decode) | fp32 reference | 9.3e-5 | 2.1e-11 | 2.8e-10 | same |
| CUDA, KV F32, ubatch 512 | fp32 reference | 1.3e-4 | 6.3e-11 | 7.5e-10 | same |
| CUDA, KV F32, ubatch 1 (decode) | fp32 reference | 1.3e-4 | 6.4e-11 | 7.7e-10 | same |
| CUDA, KV F16 (default) | fp32 reference | 1.2e-3 | 2.5e-9 | 5.3e-8 | same |
| CUDA, KV F32 | original implementation | 3.4e-3 | 1.0e-8 | 2.8e-7 | same |
| CPU, KV F32 | original implementation | 3.4e-3 | - | 2.9e-7 | same |

Reference vs original implementation:

| comparison | positions | max abs diff | KL mean / max | top-1 |
|---|---:|---:|---|---:|
| fp32 reference vs original (prefill) | 1789 | 3.1e-2 | 1.8e-7 / 5.6e-5 | 100 % |
| original decode (KV cache) vs original prefill | 32 | 4.8e-3 | 1.0e-7 / 6.4e-7 | 100 % |
| fp32 reference vs fp32 reference with fp16 cast | 1789 | 1.9e-2 | 1.4e-7 / 6.4e-5 | 100 % |

Hidden states, fp32 reference vs original: relative error 1.2e-4 after layer 1, growing smoothly to 3.7e-4
after layer 23 (no layer stands out).

Perplexity, wikitext-2, n_ctx 2048, 2 chunks, KV F32:

| | chunk 1 | final |
|---|---:|---:|
| original implementation (A40) | 11.5905 | 12.9167 |
| fp32 reference | 11.5904 | 12.9161 |
| llama.cpp CPU | 11.5905 | 12.9163 |
| llama.cpp CUDA | 11.5906 | 12.9163 |

Greedy generation, 24 tokens, prompt "Hello, my name is": identical token ids in the original implementation,
the fp32 reference, llama.cpp CPU and llama.cpp CUDA:
"Dr. R.A.S. and I am a 3rd year medical student at the University of California,".

### Automated tests

- `test-backend-ops -o NSA_CMP` (10 cases) and `-o NSA_SEL` (14 cases), CUDA vs CPU: decode (1 token) and
  prefill (7, 64, 512 tokens), cache sizes at block edges (1, 63, 64, 65) and 1000/4096, KV F32 and F16,
  head size 64 and 128, GQA 16 and 4. `NSA_CMP` overrides `err()` to compare finite values only (otherwise
  `inf - inf = NaN` and `NaN > threshold` is false, the test could not fail). `NSA_SEL` draws its block indices
  itself (distinct, ~10 % set to -1), so it does not depend on top-k.
- Mutation check: removing the q scaling in the CUDA `nsa_cmp` kernel fails 6/10 cases; shifting the window by
  one key in `nsa_sel` fails 10/14 cases (the others cannot see the bug: 0 or 1 visible block, window covering
  the whole cache).
- `test-llama-archs -a gdn-nsa`: random-weight model, consistency and save/load round trip, OK on CPU.
- `reproduce.sh compare`: end-to-end logits check with an NMSE threshold.

### No CPU fallback on CUDA

`-ngl 99`: 25/25 layers offloaded, `graph splits = 2`. With `GGML_SCHED_DEBUG=2`, the only CPU node is the
token embedding `GET_ROWS` (input layer, standard in llama.cpp); all other nodes run on CUDA0, including the
12 `NSA_CMP`, 12 `NSA_SEL`, 12 `ARGSORT`, 12 `GATED_DELTA_NET` ([placement](evidence/placement-cuda.txt)).

## Performance

Setup: Intel Core i5-13600KF (6 P + 8 E cores, 20 threads), 61 GiB RAM, NVIDIA RTX 2080 Ti 11 GB (sm_75,
616 GB/s), driver 580.178.04, CUDA 13.3, Linux 7.0. Release builds (`build-cpu`: CPU only, `build-cuda`:
CUDA, `CMAKE_CUDA_ARCHITECTURES=75`). `llama-bench`, 5 repetitions (GPU) or 3 (CPU), mean +/- std.

### CUDA (-ngl 99)

| weights | size | pp512 (t/s) | tg128 (t/s) |
|---|---:|---:|---:|
| F32 | 1.43 GB | 7230 +/- 232 | 209 +/- 8 |
| F16 | 718 MB | 10584 +/- 434 | 280 +/- 7 |
| Q8_0 | 382 MB | 9444 +/- 176 | 338 +/- 8 |
| Q4_K_M | 225 MB | 9160 +/- 168 | 371 +/- 11 |

### CPU

| weights | pp512, 20 threads | tg128, 6 threads |
|---|---:|---:|
| F32 | 539 +/- 5 | 15.9 +/- 0.8 |
| F16 | 466 +/- 1 | 31.0 +/- 0.2 |
| Q8_0 | 479 +/- 4 | 49.6 +/- 1.4 |
| Q4_K_M | 638 +/- 5 | 65.8 +/- 1.5 |

Generation is fastest with 6 threads (P cores); 12 and 20 threads are slower (E cores, synchronization).
Full table with 6/12/20 threads: [bench-cpu.md](evidence/bench-cpu.md).

### Generation vs context depth (F16)

| depth | 0 | 512 | 1024 | 2048 | 3584 |
|---|---:|---:|---:|---:|---:|
| CUDA tg64 (t/s) | 283 | 241 | 223 | 210 | 194 |
| CPU tg32, 6 threads (t/s) | 25.7 | - | 24.1 | 22.0 | 20.1 |

### Memory

| | value |
|---|---|
| CUDA buffers, F32, ctx 4096 | model 1367 MiB on GPU + 125 MiB token embeddings on CPU, KV cache 24 MiB, recurrent state 6.4 MiB, compute 64.5 MiB |
| peak VRAM of the process, F16, ctx 4096 | 964 MiB (includes the CUDA context) |
| peak RSS, CPU, ctx 4096 | F32 1.50 GB, F16 0.80 GB, Q4_K_M 0.47 GB |

KV cache: 12 NSA layers x 4096 positions x (128 + 128) values x 2 bytes = 24 MiB. Recurrent state: 12 GDN
layers x (3 x 3072 conv + 128 x 1024 state) floats = 6.4 MiB, constant in context length.

### Quantization quality

| weights | PPL, full wikitext-2 test (160 x 2048) | KLD vs F32, mean | same top-1 |
|---|---:|---:|---:|
| F32 | 12.890 +/- 0.089 | - | - |
| F16 | 12.891 | 7.4e-5 | 99.37 % |
| Q8_0 | 12.894 | 2.8e-4 | 99.12 % |
| Q4_K_M | 13.034 (+1.1 %) | 1.0e-2 | 94.47 % |

No code change was needed: `llama-quantize` only touches the matmul weights; the NSA and GDN ops work on F32
activations.

## Analysis

### Generation is bound by weight reads plus a fixed per-token cost

A fit over the four weight types gives `t_token = 2.28 ms + weight_bytes / 595 GB/s` (616 GB/s theoretical).
It predicts 287 t/s for F16 (measured 280-283) and 342 t/s for Q8_0 (338). F32 is dominated by weight reads
(2.4 of 4.7 ms); Q4_K_M by the fixed cost of ~630 small kernels per token. At depth 0, NSA (both ops, argsort
and block means) is ~4.6 % of GPU time ([profiles](evidence/profile-summary.md)). On CPU, F32 and F16 read
~22 GB/s of weights: also memory bound.

### What grows with the context

From depth 0 to 3584, CUDA generation loses 1.7 ms per token of GPU time: `nsa_sel` +0.63 ms (up to
16 x 64 + 512 = 1536 keys per query, reached around t = 1536), block means and their copies +0.57 ms
(recomputed over the whole cache at each token), `nsa_cmp` +0.15 ms. The selected and window branches are
bounded; only the compression part is O(context).

### Prompt processing

The first version used `ggml_top_k` for block selection. Profiling showed 75 % of pp512 GPU time in CUB
top-k: `ggml_cuda_op_top_k` launches one CUB top-k per row (512 tokens x 2 KV heads x 12 layers = 12288
launches). `ggml_argsort_top_k` sorts all rows in one kernel: pp512 F16 went from 2087 to 10724 t/s (same
indices, the order does not matter for `nsa_sel`). `nsa_sel` is now 53 % of pp512 GPU time.

A variant of `nsa_sel` with one key per lane (full dot product per lane, 2 warp reductions per 32 keys) was
correct but 3.3x slower: each lane reading its own key row makes every load instruction touch 32 rows instead
of one coalesced read. It was reverted.

## Limitations and next steps

- One sequence per context. The NSA ops take "cache cell j holds position j", true for a single sequence
  filled from position 0. Context creation refuses `n_seq_max > 1` and KV cache types other than F16/F32, and
  the KV cache shift is disabled for gdn-nsa, so unsupported setups fail with a clear message (for example
  `llama-perplexity` with `-b > -c`, or parallel slots in llama-server). Supporting several sequences needs the
  position of each cache cell passed to the ops.
- Block means are recomputed over the whole cache at each step (as in the reference): O(n_kv) per NSA layer and
  token. An incremental cache of the means (a block only changes until it is complete) would remove it.
- `nsa_sel` does not share K/V between the 16 query heads of a group (the extra reads mostly hit L2). Next: a
  shared-memory K/V tile per (token, KV head) loaded once for the 16 heads, and split-K for long windows in
  decode.
- CUDA only. On Vulkan or Metal, `supports_op` would be false for the NSA ops and the scheduler would run them
  on CPU (visible as extra graph splits, not silent). The kernels are simple (no tensor cores, F32 math) and
  would port to a Vulkan compute shader for Adreno/Mali.
- `nsa_cmp` uses `G x n_blk` floats of shared memory: up to ~49k tokens of context with G = 16. The model was
  trained with 4096 tokens; longer contexts are untested.
- `ggml_l2_norm` computes `x / max(||x||, eps)`, fla uses `x / sqrt(sum x^2 + 1e-6)`: identical except for
  near-zero vectors.
- The CUDA kernel comments follow a line-by-line style, more verbose than the repository guidelines.

## Use of AI

The work was done with Claude (Anthropic) as a pair programmer, with Leo directing and reviewing every step.
Claude explored the code base and the reference repositories, drafted most of the code (converter, model
graph, ggml ops, CPU and CUDA kernels, tests, reference scripts) and this report, and ran the measurements.
Leo chose the design (two ops plus an existing top-k, CUDA as GPU backend, validation against the original
implementation on a cloud GPU), reviewed the changes commit by commit and made every commit.

## Reproduce

```bash
export HF_DIR=models/gated-deltanet-nsa-0.4B-10B GGUF_DIR=models/gguf WIKI=data/wikitext-2-raw/wiki.test.raw
./docs/gdn-nsa/reproduce.sh build convert reference compare tests ppl placement bench
./docs/gdn-nsa/reproduce.sh official   # prints the commands for the original implementation (Ampere+ GPU)
```

Raw outputs used in this report: [evidence/](evidence/).
