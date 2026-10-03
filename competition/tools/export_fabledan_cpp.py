#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Export FableDan's NumPy checkpoint to native FBDN001/FBDN002 format.

The Botzone Python package consumes ``.npz`` weights, but the C++ backend
should not need a ZIP/NPZ reader.  This exporter writes a small little-endian
header, a deterministic tensor table, and a contiguous payload.  Float16 is
the default (about 8.7 MB for the released four-block model); ``--dtype
fp32`` is useful for a reference parity run.

Format (all integer fields little-endian)::

    magic[8] = b"FBDN001\\0" (80 features) or b"FBDN002\\0" (224 features)
    u32 version, dtype(1=fp16, 2=fp32), header_bytes, tensor_count
    u32 d_model, n_blocks, n_heads, qk_dim, v_dim, ffn_hidden,
        hand_hidden, n_hand_layers, q_hidden, n_q_layers, max_seq, vocab,
        feat_dim
    f32 rms_epsilon
    u32 payload_bytes
    u8[32] payload_sha256 (raw SHA-256 digest)
    tensor_count records:
        u16 name_bytes, u8 rank, u8 reserved
        u32 shape[rank], u64 payload_offset, u64 payload_bytes
        name bytes (UTF-8)
    payload bytes (float16/float32, row-major, little-endian)

No pickle or Python object is placed in the output.  The C++ loader validates
the exact released architecture and every tensor shape before exposing it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
from pathlib import Path
from typing import Dict, List, Mapping, Tuple

import numpy as np


FORMATS = {80: (b"FBDN001\0", 1), 224: (b"FBDN002\0", 2)}
DTYPE_CODES = {"fp16": 1, "fp32": 2}

EXPECTED_CONFIG = {
    "d_model": 128,
    "n_blocks": 4,
    "n_heads": 4,
    "qk_dim": 64,
    "v_dim": 64,
    "ffn_hidden": 512,
    "hand_hidden": 512,
    "n_hand_layers": 3,
    "q_hidden": 1024,
    "n_q_layers": 3,
    "max_seq": 512,
    "vocab": 48,
    "feat_dim": 80,
}


def _tensor_names() -> List[str]:
    names = ["rope_cos", "rope_sin", "token_emb.weight"]
    for block in range(4):
        p = f"blocks.{block}."
        names.extend(
            [
                p + "attn_norm.weight",
                p + "attn.q_proj.weight",
                p + "attn.k_proj.weight",
                p + "attn.v_proj.weight",
                p + "attn.out_proj.weight",
                p + "attn.q_norm.weight",
                p + "attn.k_norm.weight",
                p + "ffn_norm.weight",
                p + "ffn.gate_proj.weight",
                p + "ffn.up_proj.weight",
                p + "ffn.down_proj.weight",
            ]
        )
    names.extend(
        [
            "final_norm.weight",
            "hand_mlp.0.weight",
            "hand_mlp.0.bias",
            "hand_mlp.2.weight",
            "hand_mlp.2.bias",
            "hand_mlp.4.weight",
            "hand_mlp.4.bias",
            "hand_mlp.6.weight",
            "hand_mlp.6.bias",
            "q_head.0.weight",
            "q_head.0.bias",
            "q_head.2.weight",
            "q_head.2.bias",
            "q_head.4.weight",
            "q_head.4.bias",
            "q_head.6.weight",
            "q_head.6.bias",
        ]
    )
    assert len(names) == 64
    return names


def _parse_config(raw: np.ndarray) -> Dict[str, object]:
    config: Dict[str, object] = {}
    for item in np.asarray(raw).reshape(-1).tolist():
        text = str(item)
        key, sep, value = text.partition("=")
        if not sep:
            raise ValueError(f"invalid __config__ entry: {text!r}")
        try:
            config[key] = int(value)
        except ValueError:
            try:
                config[key] = float(value)
            except ValueError:
                config[key] = value
    return config


def _check_config(config: Mapping[str, object]) -> None:
    missing = [k for k in EXPECTED_CONFIG if k not in config]
    if missing:
        raise ValueError("missing model config: " + ", ".join(missing))
    wrong = [
        f"{key}={config[key]!r} (expected {value})"
        for key, value in EXPECTED_CONFIG.items()
        if key != "feat_dim" and config[key] != value
    ]
    if wrong:
        raise ValueError("unsupported FableDan architecture: " + "; ".join(wrong))
    feat_dim = config["feat_dim"]
    if isinstance(feat_dim, bool) or feat_dim not in FORMATS:
        raise ValueError(f"unsupported FableDan feature dimension: {feat_dim!r}")
    feature_version = config.get("feature_version", 1 if feat_dim == 80 else None)
    if isinstance(feature_version, bool) or feature_version != FORMATS[feat_dim][1]:
        raise ValueError(
            f"incompatible FableDan feature_version={feature_version!r} for feat_dim={feat_dim}"
        )


def _array_bytes(array: np.ndarray, dtype: str) -> bytes:
    source = np.asarray(array)
    if source.ndim == 0:
        raise ValueError("scalar tensor is not supported")
    if not np.issubdtype(source.dtype, np.floating):
        raise ValueError(f"tensor dtype must be floating point, got {source.dtype}")
    source = np.asarray(source, dtype=np.float32)
    if not np.all(np.isfinite(source)):
        raise ValueError("tensor contains NaN/Inf")
    if dtype == "fp16":
        stored = source.astype("<f2")
        if not np.all(np.isfinite(stored.astype(np.float32))):
            raise ValueError("tensor overflows float16")
    else:
        stored = source.astype("<f4", copy=False)
    return np.ascontiguousarray(stored).tobytes(order="C")


def export_npz(input_path: Path, output_path: Path, dtype: str) -> Dict[str, object]:
    names = _tensor_names()
    with np.load(str(input_path), allow_pickle=False) as archive:
        if "__config__" not in archive.files:
            raise ValueError("input NPZ has no __config__")
        config = _parse_config(archive["__config__"])
        _check_config(config)
        feat_dim = int(config["feat_dim"])
        magic, version = FORMATS[feat_dim]
        export_config = dict(EXPECTED_CONFIG, feat_dim=feat_dim, feature_version=version)
        missing = [name for name in names if name not in archive.files]
        if missing:
            raise ValueError("input NPZ is missing: " + ", ".join(missing))
        unexpected = [
            name
            for name in archive.files
            if name != "__config__" and name not in names
        ]
        if unexpected:
            raise ValueError("unsupported tensors in input NPZ: " + ", ".join(unexpected))

        payload = bytearray()
        table: List[Tuple[str, Tuple[int, ...], int, int]] = []
        item_bytes = 2 if dtype == "fp16" else 4
        for name in names:
            array = np.asarray(archive[name])
            if name == "hand_mlp.0.weight" and array.shape != (512, feat_dim):
                raise ValueError(
                    f"weight shape mismatch for {name}: expected {(512, feat_dim)}, got {array.shape}"
                )
            raw = _array_bytes(array, dtype)
            if len(raw) != int(array.size) * item_bytes:
                raise AssertionError(f"internal size mismatch for {name}")
            offset = len(payload)
            payload.extend(raw)
            table.append((name, tuple(int(v) for v in array.shape), offset, len(raw)))

    digest = hashlib.sha256(payload).digest()
    header = bytearray()
    header.extend(magic)
    # version, dtype, header_bytes (patched below), tensor_count
    header.extend(struct.pack("<IIII", version, DTYPE_CODES[dtype], 0, len(table)))
    for key in (
        "d_model",
        "n_blocks",
        "n_heads",
        "qk_dim",
        "v_dim",
        "ffn_hidden",
        "hand_hidden",
        "n_hand_layers",
        "q_hidden",
        "n_q_layers",
        "max_seq",
        "vocab",
        "feat_dim",
    ):
        header.extend(struct.pack("<I", export_config[key]))
    header.extend(struct.pack("<fI", 1e-6, len(payload)))
    header.extend(digest)

    for name, shape, offset, byte_count in table:
        encoded_name = name.encode("utf-8")
        if len(encoded_name) > 0xFFFF or len(shape) > 0xFF:
            raise ValueError(f"tensor metadata too large for {name}")
        header.extend(struct.pack("<HBB", len(encoded_name), len(shape), 0))
        for dimension in shape:
            if dimension <= 0 or dimension > 0xFFFFFFFF:
                raise ValueError(f"invalid shape for {name}: {shape}")
            header.extend(struct.pack("<I", dimension))
        header.extend(struct.pack("<QQ", offset, byte_count))
        header.extend(encoded_name)

    # header_bytes is at byte offset 16: magic[8] + version/dtype[8].
    struct.pack_into("<I", header, 16, len(header))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(header + payload)

    manifest = {
        "format": magic[:-1].decode("ascii"),
        "version": version,
        "dtype": dtype,
        "header_bytes": len(header),
        "payload_bytes": len(payload),
        "payload_sha256": digest.hex(),
        "tensor_count": len(table),
        "config": export_config,
        "source_npz": str(input_path),
        "output": str(output_path),
    }
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="FableDan export_npz .npz")
    parser.add_argument("output", type=Path, help="destination .fbdn file")
    parser.add_argument("--dtype", choices=sorted(DTYPE_CODES), default="fp16")
    parser.add_argument(
        "--manifest",
        type=Path,
        help="optional JSON sidecar containing config and payload SHA-256",
    )
    args = parser.parse_args()
    manifest = export_npz(args.input, args.output, args.dtype)
    if args.manifest:
        args.manifest.parent.mkdir(parents=True, exist_ok=True)
        args.manifest.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(manifest, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
