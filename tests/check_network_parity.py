"""Numerical deployment check with deterministic synthetic weights, not a release."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import struct
import sys
import tempfile
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "tools"), str(ROOT / "train")]
from model import CandidateModel, ModelConfig
from export_model import export
from probe import Probe


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe", type=Path, default=ROOT / "bin/network_probe")
    parser.add_argument("--report", type=Path, default=ROOT / "reports/network_parity.json")
    parser.add_argument("--checkpoint", type=Path, help="verify genuine trained weights instead of synthetic initialization")
    args = parser.parse_args()
    torch.manual_seed(20261001)
    torch.set_num_threads(1)
    provenance = {"purpose": "synthetic_numerical_parity_only"}
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        model = CandidateModel(ModelConfig(**checkpoint["config"]))
        model.load_state_dict(checkpoint["state_dict"], strict=True)
        provenance = checkpoint["provenance"]
    else:
        model = CandidateModel()
    model.eval()
    maximum = 0
    checks = []
    rejected = []
    with tempfile.TemporaryDirectory(prefix="oxbot-parity-") as tmp, Probe(args.probe) as probe:
        path = Path(tmp) / "synthetic-do-not-publish.bin"
        manifest = export(model, path, provenance)
        for length, candidates in ((2, 1), (17, 64), (128, 256), (256, 1024)):
            tokens = torch.randint(1, 128, (1, length))
            state = torch.randn(1, 128)
            actions = torch.randn(1, candidates, 128)
            with torch.no_grad():
                expected = model(tokens, torch.tensor([length]), state, actions)[0].numpy()
                traces = {}
                x = model.embedding(tokens) + model.position(torch.arange(length))[None]
                traces["embedding"] = x
                for layer, block in enumerate(model.blocks):
                    x = block(x,torch.ones(1,length,dtype=torch.bool))
                    traces[f"blocks.{layer}"] = x
                traces["final_norm"] = model.final_norm(x)
                traces["state_projection"] = torch.relu(model.state_projection(state))
            actual = probe.call(path=str(path), tokens=tokens[0].tolist(), state=state[0].tolist(), actions=actions[0].tolist(), trace=True)
            optimized = probe.call(path=str(path), tokens=tokens[0].tolist(), state=state[0].tolist(), actions=actions[0].tolist())
            assert actual["ok"] and actual["sha"] == manifest["payload_sha256"], actual
            assert optimized["ok"] and optimized["scores"] == actual["scores"], "optimized and complete block paths differ"
            error = float(np.max(np.abs(expected-np.asarray(actual["scores"]))))
            assert error < 1e-4, error
            assert int(np.argmax(expected)) == int(np.argmax(actual["scores"])), actual
            layer_errors = {name:float(np.max(np.abs(value.numpy().ravel()-actual["trace"][name]))) for name,value in traces.items()}
            assert all(e < 1e-4 for e in layer_errors.values()), layer_errors
            maximum = max(maximum,error)
            checks.append({"history_length": length, "candidates": candidates, "max_abs_error": error,
                           "cpp_ms": optimized["milliseconds"], "full_trace_ms": actual["milliseconds"],
                           "optimized_exactly_equal": True, "same_argmax": True, "layer_max_errors":layer_errors})
        data = path.read_bytes()
        header_length = struct.unpack("<I", data[8:12])[0]
        header = json.loads(data[12:12+header_length])
        payload = data[12+header_length:]
        variants = {"truncated": data[:10], "bad_magic": b"INVALID!"+data[8:],
                    "checksum": data[:-1]+bytes([data[-1]^1])}
        for key in ("feature_version", "rules_contract", "architecture"):
            bad = dict(header, **{key: "incompatible"})
            raw = json.dumps(bad).encode()
            variants[key] = b"OXGDQ001"+struct.pack("<I",len(raw))+raw+payload
        bad_config = dict(header, config=dict(header["config"], heads=8))
        raw = json.dumps(bad_config).encode()
        variants["config"] = b"OXGDQ001"+struct.pack("<I", len(raw))+raw+payload
        for name, content in variants.items():
            bad_path = Path(tmp)/(name+".bin")
            bad_path.write_bytes(content)
            response = probe.call(path=str(bad_path))
            assert not response["ok"], (name,response)
            rejected.append({"case":name,"status":response["status"]})
        assert not probe.call(path=str(Path(tmp)/"missing.bin"))["ok"]
    report = {"status":"passed", "max_absolute_error":maximum, "cases":checks, "rejections":rejected,
              "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest() if args.checkpoint else None,
              "payload_sha256": manifest["payload_sha256"],
              "scope": ("trained-checkpoint numerical parity and artifact validation; not gameplay strength"
                        if args.checkpoint else "synthetic forward parity and artifact validation; not gameplay strength or trained-checkpoint parity")}
    args.report.parent.mkdir(parents=True,exist_ok=True)
    args.report.write_text(json.dumps(report,indent=2),encoding="utf-8")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
