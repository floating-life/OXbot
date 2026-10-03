"""Versioned little-endian float32 export; no pickle is loaded by the C++ Bot."""
from __future__ import annotations
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import struct
import torch
from model import CandidateModel, ModelConfig
from features import FEATURE_VERSION

RULES_CONTRACT = "botzone-corrected-fa63589d-v1"
DEFAULT_MODEL_VERSION = "oxbot-model-v1"
SELECTION_MANIFEST = {
    "schema": "oxbot-selection-v1",
    "default": "raw",
    "supported": ["raw", "raw-pass-bias", "group-logmeanexp"],
    "pass_bias": -0.5,
    "group_definition": {
        "key": "kind,length,key,secondary,rank-count multiset,wildcard count",
        "wildcard": "heart face at current level",
        "straight_flush": "full canonical face-count/suit signature retained",
        "aggregation": "logmeanexp(raw member scores)",
        "member": "highest raw score within selected group",
    },
}


def _validate_model_version(value: str) -> str:
    if not isinstance(value, str) or not value or len(value) > 96:
        raise ValueError("invalid model version")
    if any(not (char.isascii() and (char.isalnum() or char in ".-_")) for char in value):
        raise ValueError("invalid model version")
    return value


def export(model, destination: Path, provenance=None, selection_default="raw", model_version=DEFAULT_MODEL_VERSION):
    if selection_default not in SELECTION_MANIFEST["supported"]:
        raise ValueError(f"unsupported selection default: {selection_default}")
    model_version = _validate_model_version(model_version)
    config = asdict(ModelConfig())
    if model.architecture() != config:
        raise ValueError("C++ evaluator supports only the exact fixed v1 architecture")
    payload, tensors = bytearray(), []
    for name, tensor in model.state_dict().items():
        array = tensor.detach().cpu().float().numpy().astype("<f4")
        if not __import__("numpy").isfinite(array).all():
            raise ValueError(f"nonfinite weights in {name}")
        tensors.append({"name": name, "shape": list(array.shape), "count": array.size, "offset": len(payload)})
        payload.extend(array.tobytes(order="C"))
    selection = dict(SELECTION_MANIFEST)
    selection["default"] = selection_default
    manifest = {"architecture": config["architecture"], "config": config, "feature_version": FEATURE_VERSION,
                "rules_contract": RULES_CONTRACT, "dtype": "float32-le", "tensors": tensors,
                "selection": selection, "model_version": model_version,
                "payload_sha256": hashlib.sha256(payload).hexdigest(), "provenance": provenance or {}}
    header = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode("utf-8")
    if len(header) > 65536:
        raise ValueError("model header exceeds C++ 64 KiB limit; store large provenance tables outside the binary")
    content = b"OXGDQ001" + struct.pack("<I", len(header)) + header + payload
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(content)
    sidecar = dict(manifest, file_sha256=hashlib.sha256(content).hexdigest(), bytes=len(content), file=destination.name)
    destination.with_suffix(".manifest.json").write_text(json.dumps(sidecar, indent=2), encoding="utf-8")
    return sidecar


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--selection-default", choices=tuple(SELECTION_MANIFEST["supported"]), default="raw",
                        help="header manifest default; does not alter model weights")
    parser.add_argument("--model-version", default=DEFAULT_MODEL_VERSION,
                        help="diagnostic candidate identity stored in the model header")
    args = parser.parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model = CandidateModel(ModelConfig(**checkpoint["config"]))
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    manifest = export(model, args.output, checkpoint.get("provenance", {}), args.selection_default, args.model_version)
    print(json.dumps({k: manifest[k] for k in ("file", "bytes", "file_sha256", "payload_sha256")}))


if __name__ == "__main__":
    main()
