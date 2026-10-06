#!/usr/bin/env python3
"""
Compare two sets of logits for the gdn-nsa validation.

Inputs are either .bin files (float32 logits of the last token, as written by
run-org-model*.py and llama-debug --save-logits) or .npy files ([T, n_vocab],
all positions, as written by run-org-model-gdn-nsa*.py --dump-dir).
Metrics are computed in float64: max abs diff, NMSE, KL(b || a) and top-1 agreement.
"""

import argparse
import sys

import numpy as np


def load(path, n_vocab):
    if path.endswith(".npy"):
        x = np.load(path)
    else:
        x = np.fromfile(path, dtype=np.float32)
    x = x.astype(np.float64)
    return x.reshape(-1, n_vocab)


def log_softmax(x):
    x = x - x.max(axis=-1, keepdims=True)
    return x - np.log(np.exp(x).sum(axis=-1, keepdims=True))


def main():
    parser = argparse.ArgumentParser(description="compare two logits files (a = tested, b = reference)")
    parser.add_argument("a", help="tested logits (.bin or .npy)")
    parser.add_argument("b", help="reference logits (.bin or .npy)")
    parser.add_argument("--n-vocab", type=int, default=32000)
    parser.add_argument("--last", type=int, default=0, help="compare only the last N positions (0 = all common positions)")
    parser.add_argument("--max-nmse", type=float, default=None, help="exit with code 1 if NMSE is above this value")
    args = parser.parse_args()

    a, b = load(args.a, args.n_vocab), load(args.b, args.n_vocab)
    n = min(len(a), len(b))
    if args.last:
        n = min(n, args.last)
    a, b = a[-n:], b[-n:]

    la, lb = log_softmax(a), log_softmax(b)
    kl = (np.exp(lb) * (lb - la)).sum(axis=-1)
    nmse = ((a - b) ** 2).sum() / (b ** 2).sum()
    top1 = (a.argmax(axis=-1) == b.argmax(axis=-1)).mean() * 100

    print(f"positions {n} | max|d| {np.abs(a - b).max():.3e} | NMSE {nmse:.3e} | "
          f"KL mean {kl.mean():.3e} max {kl.max():.3e} | top-1 {top1:.2f} %")

    if args.max_nmse is not None and nmse > args.max_nmse:
        print(f"FAIL: NMSE {nmse:.3e} > {args.max_nmse:.1e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
