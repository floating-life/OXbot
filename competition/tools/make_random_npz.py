# -*- coding: utf-8 -*-
"""Write a randomly initialised FableDan transformer .npz (numpy only).

Same keys/shapes as model_torch.export_npz with the default ModelConfig, so
the bot's real inference path (model_np.NumpyModel) can be exercised for
legality / speed / memory tests on machines without PyTorch.

    python tools/make_random_npz.py --out /tmp/rand.npz
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from fabledan.encode import FEAT_DIM, MAX_SEQ, VOCAB  # noqa: E402


def build(seed=0, d=128, n_blocks=4, n_heads=4, qk=64, v=64, ffn=512,
          hand_hidden=512, n_hand_layers=3, q_hidden=1024, n_q_layers=3):
    rng = np.random.default_rng(seed)

    def lin(o, i, bias=False):
        w = (rng.standard_normal((o, i)) / np.sqrt(i)).astype(np.float32)
        return (w, np.zeros(o, np.float32)) if bias else w

    a = {"token_emb.weight": (rng.standard_normal((VOCAB, d)) * 0.02).astype(np.float32)}
    for b in range(n_blocks):
        p = "blocks.%d." % b
        a[p + "attn_norm.weight"] = np.ones(d, np.float32)
        a[p + "attn.q_proj.weight"] = lin(n_heads * qk, d)
        a[p + "attn.k_proj.weight"] = lin(n_heads * qk, d)
        a[p + "attn.v_proj.weight"] = lin(n_heads * v, d)
        a[p + "attn.out_proj.weight"] = lin(d, n_heads * v)
        a[p + "attn.q_norm.weight"] = np.ones(qk, np.float32)
        a[p + "attn.k_norm.weight"] = np.ones(qk, np.float32)
        a[p + "ffn_norm.weight"] = np.ones(d, np.float32)
        a[p + "ffn.gate_proj.weight"] = lin(ffn, d)
        a[p + "ffn.up_proj.weight"] = lin(ffn, d)
        a[p + "ffn.down_proj.weight"] = lin(d, ffn)
    a["final_norm.weight"] = np.ones(d, np.float32)
    half = qk // 2
    freqs = 1.0 / (10000.0 ** (np.arange(half, dtype=np.float32) / half))
    ang = np.outer(np.arange(MAX_SEQ, dtype=np.float32), freqs)
    a["rope_cos"] = np.cos(ang).astype(np.float32)
    a["rope_sin"] = np.sin(ang).astype(np.float32)

    def mlp(prefix, dims):
        for j in range(len(dims) - 1):
            w, bvec = lin(dims[j + 1], dims[j], bias=True)
            a["%s%d.weight" % (prefix, 2 * j)] = w
            a["%s%d.bias" % (prefix, 2 * j)] = bvec

    mlp("hand_mlp.", [FEAT_DIM] + [hand_hidden] * n_hand_layers + [d])
    mlp("q_head.", [2 * d] + [q_hidden] * n_q_layers + [1])
    cfg = dict(d_model=d, n_blocks=n_blocks, n_heads=n_heads, qk_dim=qk,
               v_dim=v, ffn_hidden=ffn)
    a["__config__"] = np.array(["%s=%s" % kv for kv in cfg.items()], dtype=np.str_)
    return a


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    np.savez_compressed(args.out, **build(args.seed))
    print("wrote %s (%.1f MB)" % (args.out, os.path.getsize(args.out) / 1e6))


if __name__ == "__main__":
    main()
