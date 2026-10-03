# -*- coding: utf-8 -*-
"""Pure-numpy inference of FableDanNet (for Botzone deployment & sandbox).

Loads the .npz produced by model_torch.export_npz. Single-sequence forward.

Memory: uses float32 throughout (no float64 upcast) to stay under Botzone's
256 MB sandbox limit.  Peak allocation ~20 MB for a 512-token sequence with
~100 legal moves.
"""

import gc

import numpy as np

from .encode import FEAT_DIM, STRUCTURE_FEAT_DIM


def _rms(x, w, eps=1e-6):
    """RMS norm in float32 (avoid float64 to reduce memory on constrained sandbox)."""
    v = np.mean(x.astype(np.float32) ** 2, axis=-1, keepdims=True)
    return (x / np.sqrt(v + eps)) * w


def _silu(x):
    return x / (1.0 + np.exp(-x))


def _relu(x):
    return np.maximum(x, 0.0)


def _softmax(x, axis=-1):
    m = x.max(axis=axis, keepdims=True)
    e = np.exp(x - m)
    return e / e.sum(axis=axis, keepdims=True)


class NumpyModel:
    def __init__(self, npz_path_or_dict):
        if isinstance(npz_path_or_dict, dict):
            w = npz_path_or_dict
        else:
            w = dict(np.load(npz_path_or_dict, allow_pickle=False))
        self.w = {k: np.asarray(v) for k, v in w.items() if k != "__config__"}
        cfg = {}
        if "__config__" in w:
            for item in w["__config__"]:
                k, _, v = str(item).partition("=")
                try:
                    cfg[k] = int(v)
                except ValueError:
                    try:
                        cfg[k] = float(v)
                    except ValueError:
                        cfg[k] = v
        defaults = {"d_model": 128, "n_blocks": 4, "n_heads": 4, "qk_dim": 64,
                    "v_dim": 64, "ffn_hidden": 512, "hand_hidden": 512,
                    "n_hand_layers": 3, "q_hidden": 1024, "n_q_layers": 3,
                    "max_seq": 512, "vocab": 48, "feat_dim": FEAT_DIM,
                    "feature_version": 1}
        for name, default in defaults.items():
            value = cfg.get(name, default)
            if not isinstance(value, (int, np.integer)) or value <= 0:
                raise ValueError("invalid positive integer model config: %s" % name)
            cfg[name] = int(value)
        self.cfg = cfg
        self.feat_dim = cfg["feat_dim"]
        self.feature_version = cfg["feature_version"]
        if {FEAT_DIM: 1, STRUCTURE_FEAT_DIM: 2}.get(self.feat_dim) != self.feature_version:
            raise ValueError("feature_version does not match supported feat_dim")
        if cfg["qk_dim"] % 2:
            raise ValueError("qk_dim must be even for rotary encoding")
        self._validate_shapes()
        self.n_blocks = int(cfg.get("n_blocks", 4))
        self.n_heads = int(cfg.get("n_heads", 4))
        self.qk = int(cfg.get("qk_dim", 64))
        self.v = int(cfg.get("v_dim", 64))
        self.rope_cos = self.w["rope_cos"]
        self.rope_sin = self.w["rope_sin"]

    def _validate_shapes(self):
        """Reject stale configs or incompatible weight matrices at load time."""
        cfg = self.cfg
        d, h, qk, v = (cfg[key] for key in ("d_model", "n_heads", "qk_dim", "v_dim"))
        expected = {"token_emb.weight": (cfg["vocab"], d),
                    "rope_cos": (cfg["max_seq"], qk // 2),
                    "rope_sin": (cfg["max_seq"], qk // 2),
                    "final_norm.weight": (d,)}
        for i in range(cfg["n_blocks"]):
            pre = "blocks.%d." % i
            expected.update({
                pre + "attn_norm.weight": (d,), pre + "ffn_norm.weight": (d,),
                pre + "attn.q_proj.weight": (h * qk, d),
                pre + "attn.k_proj.weight": (h * qk, d),
                pre + "attn.v_proj.weight": (h * v, d),
                pre + "attn.out_proj.weight": (d, h * v),
                pre + "attn.q_norm.weight": (qk,), pre + "attn.k_norm.weight": (qk,),
                pre + "ffn.gate_proj.weight": (cfg["ffn_hidden"], d),
                pre + "ffn.up_proj.weight": (cfg["ffn_hidden"], d),
                pre + "ffn.down_proj.weight": (d, cfg["ffn_hidden"]),
            })
        for prefix, in_dim, hidden, layers, out_dim in (
                ("hand_mlp.", self.feat_dim, cfg["hand_hidden"], cfg["n_hand_layers"], d),
                ("q_head.", d * 2, cfg["q_hidden"], cfg["n_q_layers"], 1)):
            for i in range(layers + 1):
                width = out_dim if i == layers else hidden
                expected[prefix + "%d.weight" % (i * 2)] = (width, in_dim)
                expected[prefix + "%d.bias" % (i * 2)] = (width,)
                in_dim = width
        for name, shape in expected.items():
            value = self.w.get(name)
            if value is None or value.shape != shape:
                raise ValueError("weight shape mismatch for %s: expected %s" % (name, shape))
            if value.dtype.kind != "f" or not np.isfinite(value).all():
                raise ValueError("weight must contain finite floating-point values: %s" % name)
            self.w[name] = value.astype(np.float32, copy=False)
        extra = [name for name in self.w if name not in expected
                 and not name.startswith(("ntp_head.", "belief_head."))]
        if extra:
            raise ValueError("unexpected model weight: %s" % extra[0])

    # ------------------------------------------------------------------
    def _attn(self, x, i):
        w = self.w
        T = x.shape[0]
        h, qk, vd = self.n_heads, self.qk, self.v
        pre = "blocks.%d.attn." % i
        q = x @ w[pre + "q_proj.weight"].T
        k = x @ w[pre + "k_proj.weight"].T
        v = x @ w[pre + "v_proj.weight"].T
        q = q.reshape(T, h, qk).transpose(1, 0, 2)
        k = k.reshape(T, h, qk).transpose(1, 0, 2)
        v = v.reshape(T, h, vd).transpose(1, 0, 2)
        q = _rms(q, w[pre + "q_norm.weight"])
        k = _rms(k, w[pre + "k_norm.weight"])
        q = self._rope(q, T)
        k = self._rope(k, T)
        scores = q @ k.transpose(0, 2, 1) / np.sqrt(qk)
        mask = np.triu(np.full((T, T), -1e30, dtype=np.float32), k=1)
        scores = scores + mask[None]
        a = _softmax(scores, axis=-1)
        o = a @ v                       # [h, T, vd]
        o = o.transpose(1, 0, 2).reshape(T, h * vd)
        return o @ w[pre + "out_proj.weight"].T

    def _rope(self, x, T):
        d = x.shape[-1] // 2
        c = self.rope_cos[:T][None]
        s = self.rope_sin[:T][None]
        x1, x2 = x[..., :d], x[..., d:]
        return np.concatenate([x1 * c - x2 * s, x1 * s + x2 * c], axis=-1)

    def _mlp(self, x, prefix, n_layers):
        w = self.w
        # Sequential indices 0,2,4,... are Linear layers
        idxs = [k for k in w if k.startswith(prefix) and k.endswith(".weight")]
        order = sorted(int(k[len(prefix):].split(".")[0]) for k in idxs)
        for j, li in enumerate(order):
            x = x @ w["%s%d.weight" % (prefix, li)].T + w["%s%d.bias" % (prefix, li)]
            if j < len(order) - 1:
                x = _relu(x)
        return x

    # ------------------------------------------------------------------
    def context(self, tokens):
        """tokens: list[int] -> ctx vector [d]."""
        w = self.w
        tokens = np.asarray(tokens)
        if (tokens.ndim != 1 or not 0 < len(tokens) <= self.cfg["max_seq"]
                or tokens.dtype.kind not in "iu" or np.any(tokens < 0)
                or np.any(tokens >= self.cfg["vocab"])):
            raise ValueError("invalid model token sequence")
        x = w["token_emb.weight"][tokens.astype(np.int64, copy=False)]
        for i in range(self.n_blocks):
            pre = "blocks.%d." % i
            x = x + self._attn(_rms(x, w[pre + "attn_norm.weight"]), i)
            h = _rms(x, w[pre + "ffn_norm.weight"])
            g = h @ w[pre + "ffn.gate_proj.weight"].T
            u = h @ w[pre + "ffn.up_proj.weight"].T
            x = x + (_silu(g) * u) @ w[pre + "ffn.down_proj.weight"].T
        x = _rms(x, w["final_norm.weight"])
        # Free attention intermediates accumulated over blocks
        gc.collect()
        return x[-1]

    def q_values(self, tokens, feats, candidate_chunk=512):
        """Score every [A, cfg.feat_dim] row with one shared history encoding."""
        try:
            feats = np.asarray(feats, dtype=np.float32)
            if feats.ndim != 2 or feats.shape[1] != self.feat_dim or not np.isfinite(feats).all():
                raise ValueError("model action feature shape or values invalid")
            if candidate_chunk <= 0:
                raise ValueError("candidate_chunk must be positive")
            ctx = self.context(tokens)
            scores = []
            for start in range(0, len(feats), candidate_chunk):
                hemb = self._mlp(feats[start:start + candidate_chunk], "hand_mlp.", None)
                c = np.broadcast_to(ctx, (hemb.shape[0], ctx.shape[0]))
                q = self._mlp(np.concatenate([c, hemb], axis=-1), "q_head.", None)
                scores.append(q[:, 0])
            return np.concatenate(scores) if scores else np.empty(0, dtype=np.float32)
        finally:
            gc.collect()
