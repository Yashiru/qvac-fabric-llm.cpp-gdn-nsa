# nsys profiles (RTX 2080 Ti, F16 model, -ngl 99)

Commands: `nsys profile -t cuda --cuda-graph-trace=node ./build-cuda/bin/llama-bench -m gdn-nsa-0.4B-F16.gguf -ngl 99 ...`,
then `nsys stats --report cuda_gpu_kern_sum` (kernel time grouped by kind) or `cuda_gpu_trace` (per launch).

## Prefill pp512, block selection with ggml_top_k (first version)

| kernel | share of GPU time |
|---|---|
| cub DeviceTopKKernel (two variants) + DeviceTopKLastFilterKernel | 75.0 % (49152 launches) |
| nsa_sel | 13.6 % |
| gated_delta_net | 2.3 % |
| matmul (turing h1688 gemm) | 3.0 % |

`ggml_cuda_op_top_k` runs one CUB top-k per row: 512 tokens x 2 kv heads x 12 layers = 12288 launches per pass.

## Prefill pp512, block selection with ggml_argsort_top_k (current version)

| kernel group | share of GPU time |
|---|---|
| nsa_sel | 52.9 % |
| matmul | 13.8 % |
| copies / converts (cpy, cont, cast) | 10.8 % |
| elementwise (add, mul, norm, activations, rope) | 10.6 % |
| gated_delta_net | 9.4 % |
| ssm_conv | 1.2 % |
| nsa_cmp | 0.7 % |
| argsort | 0.1 % |

pp512 F16: 2087 t/s -> 10724 t/s after the change.

## Decode tg, depth 0 (current version)

| kernel group | share of GPU time |
|---|---|
| matmul (mul_mat_vec) | 66.2 % |
| elementwise | 14.8 % |
| other | 6.6 % |
| copies / converts | 6.5 % |
| block means (mean, pad) | 1.6 % |
| nsa_sel | 1.5 % |
| gated_delta_net | 1.2 % |
| nsa_cmp | 0.8 % |
| argsort | 0.7 % |

About 620-640 kernel launches per token.

## Decode per token, depth 0 vs depth 3584 (decode kernels only, after the depth prefill)

| kernel group | us/token d=0 | us/token d=3584 | delta |
|---|---:|---:|---:|
| nsa_sel | 49 | 679 | +630 |
| block means (mean, pad) | 96 | 469 | +372 |
| matmul | 2102 | 2352 | +250 |
| copies (cpy, cont, cast) | 206 | 399 | +193 |
| nsa_cmp | 22 | 176 | +154 |
| argsort | 24 | 86 | +63 |
| other | 637 | 679 | +42 |
| gated_delta_net | 33 | 36 | +3 |
| total GPU kernel time | 3170 | 4875 | +1706 |

The block means are recomputed over the whole cache at each token (cast, pad, permute + cont, mean for K and V):
cost O(n_kv). nsa_sel reads at most 16 x 64 + 512 = 1536 keys per query, reached for t >= ~1536.

## Decode cost model (fit on the 4 variants, depth 0)

t_token = 2.28 ms + weight_bytes / 595 GB/s (RTX 2080 Ti: 616 GB/s theoretical)

| variant | weights read per token | predicted t/s | measured t/s |
|---|---:|---:|---:|
| F32 | 1.43 GB | 214 | 209-213 |
| F16 | 0.72 GB | 287 | 280-283 |
| Q8_0 | 0.38 GB | 342 | 338 |
| Q4_K_M | 0.22 GB | 376 | 371-376 |

## Rejected optimization (measured)

nsa_sel with one key per lane (32-key chunks, full dot product per lane, 2 warp reductions per chunk):
correct (same logits, 24/24 tests) but pp512 F16 10724 -> 3260 t/s. Each lane reads its own key row element
by element, so each load instruction of the warp touches 32 rows (32 transactions instead of one coalesced read).
