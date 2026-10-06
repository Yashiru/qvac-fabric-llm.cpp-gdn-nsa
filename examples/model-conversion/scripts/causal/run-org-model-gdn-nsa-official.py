#!/usr/bin/env python3
"""
Runs the original implementation of Idiap Gated-DeltaNet + NSA (github.com/idiap/hybrid-linear-sparse-attention,
Triton + FlashAttention-2, Ampere or newer GPU) and saves its outputs, to check the CPU reference
run-org-model-gdn-nsa.py and llama.cpp against it.

Outputs in --out-dir:
  official-all-logits.npy     [T, n_vocab]           prefill logits of the token ids in --tokens-file
  official-hidden.npy         [n_layer + 1, T, n_embd] hidden states (output_hidden_states)
  official-decode-logits.npy  [n_decode, n_vocab]    logits of the last n_decode tokens, one token at a time with the cache
  official.json               versions, greedy generation, perplexity
"""

import argparse
import json
import os
import subprocess

import numpy as np
import torch

import fla  # noqa: F401 - registers hybrid_gated_deltanet
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", "-m", required=True)
    parser.add_argument("--tokens-file", required=True, help="int32 token ids (BOS included)")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--n-decode", type=int, default=32)
    parser.add_argument("--ppl-file")
    parser.add_argument("--ppl-ctx", type=int, default=2048)
    parser.add_argument("--ppl-chunks", type=int, default=2)
    args = parser.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    os.makedirs(args.out_dir, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForCausalLM.from_pretrained(args.model_path).cuda().eval()
    info = {
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(),
        "dtype": str(next(model.parameters()).dtype),
    }
    for mod in ("triton", "flash_attn", "transformers"):
        info[mod] = __import__(mod).__version__
    info["fla_commit"] = subprocess.run(["git", "-C", os.path.dirname(os.path.dirname(fla.__file__)), "rev-parse", "HEAD"],
                                        capture_output=True, text=True).stdout.strip()

    ids = torch.from_numpy(np.fromfile(args.tokens_file, dtype=np.int32).astype(np.int64))[None].cuda()
    T = ids.shape[1]
    info["n_tokens"] = T

    with torch.no_grad():
        # prefill of the whole sequence
        out = model(ids, use_cache=False, output_hidden_states=True)
        np.save(os.path.join(args.out_dir, "official-all-logits.npy"), out.logits[0].float().cpu().numpy())
        np.save(os.path.join(args.out_dir, "official-hidden.npy"), torch.stack(out.hidden_states)[:, 0].float().cpu().numpy())

        # decode path: prefill T - n_decode tokens, then one token at a time with the cache
        n0 = T - args.n_decode
        out = model(ids[:, :n0], use_cache=True)
        past = out.past_key_values
        dec = []
        for i in range(n0, T):
            out = model(ids[:, i:i + 1], past_key_values=past, use_cache=True)
            past = out.past_key_values
            dec.append(out.logits[0, -1].float().cpu())
        np.save(os.path.join(args.out_dir, "official-decode-logits.npy"), torch.stack(dec).numpy())

        # greedy generation
        prompt = tok("Hello, my name is", return_tensors="pt").input_ids.cuda()
        gen = model.generate(prompt, max_new_tokens=24, do_sample=False)
        info["greedy_tokens"] = gen[0, prompt.shape[1]:].tolist()
        info["greedy_text"] = tok.decode(gen[0])

        # perplexity, same convention as llama-perplexity
        if args.ppl_file:
            with open(args.ppl_file, encoding="utf-8") as f:
                pids = tok(f.read()).input_ids
            n_ctx, first = args.ppl_ctx, args.ppl_ctx // 2
            nll, cnt, per_chunk = 0.0, 0, []
            for c in range(args.ppl_chunks):
                chunk = torch.tensor(pids[c * n_ctx:(c + 1) * n_ctx])
                chunk[0] = tok.bos_token_id
                lp = torch.log_softmax(model(chunk[None].cuda(), use_cache=False).logits[0].float(), dim=-1).cpu()
                nll -= lp[first:n_ctx - 1].gather(1, chunk[first + 1:, None]).sum().item()
                cnt += n_ctx - 1 - first
                per_chunk.append(float(np.exp(nll / cnt)))
            info["ppl"] = per_chunk

    with open(os.path.join(args.out_dir, "official.json"), "w") as f:
        json.dump(info, f, indent=1)
    print(json.dumps(info, indent=1))


if __name__ == "__main__":
    main()
