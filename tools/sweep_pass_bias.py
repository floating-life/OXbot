"""Validation-only pass-score bias sweep for the frozen v1 checkpoint.

No checkpoint or online code is changed.  This file deliberately enumerates
only train-free validation shards and refuses any other split.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "train"))
from model import CandidateModel, ModelConfig  # noqa: E402


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def files_for(data: Path) -> list[Path]:
    files = sorted((data / "validation").glob("*.npz"))
    if not files:
        raise FileNotFoundError(f"no validation shards under {data}")
    return files


def count_bin(n: int) -> str:
    if n == 1: return "1"
    if n <= 3: return "2-3"
    if n <= 8: return "4-8"
    if n <= 16: return "9-16"
    if n <= 32: return "17-32"
    if n <= 64: return "33-64"
    if n <= 128: return "65-128"
    if n <= 256: return "129-256"
    if n <= 512: return "257-512"
    return "513+"


def collate(rows, state, tokens, lengths, actions, offsets, positive):
    import torch
    counts = [int(offsets[r + 1] - offsets[r]) for r in rows]
    width = max(counts)
    length = int(lengths[rows].max())
    a = np.zeros((len(rows), width, 128), np.float32)
    m = np.zeros((len(rows), width), np.bool_)
    p = np.zeros((len(rows), width), np.bool_)
    for j, r in enumerate(rows):
        b, e = int(offsets[r]), int(offsets[r + 1])
        a[j, :e-b] = actions[b:e]; m[j, :e-b] = True; p[j, :e-b] = positive[b:e]
    return ({"tokens": torch.from_numpy(tokens[rows, :length].astype(np.int64)),
             "lengths": torch.from_numpy(lengths[rows].astype(np.int64)),
             "state": torch.from_numpy(state[rows].astype(np.float32)),
             "actions": torch.from_numpy(a), "mask": torch.from_numpy(m)}, p, counts)


def new_bucket() -> dict:
    return {"rows": 0, "correct": 0, "choice_rows": 0, "choice_correct": 0,
            "demo_pass": 0, "pred_pass": 0, "false_pass": 0,
            "demo_size_ge4": 0, "size_ge4_correct": 0, "size_ge4_same": 0}


def run(args) -> dict:
    import torch
    data = args.data.resolve(); checkpoint = args.checkpoint.resolve()
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model = CandidateModel(ModelConfig(**ckpt["config"]))
    model.load_state_dict(ckpt["state_dict"], strict=True)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    model = model.float().to(device).eval()
    biases = [float(x) for x in args.bias]
    # Each mode uses the same frozen scores and validation rows.  'follow' is
    # equivalent to global here because the referee never emits a pass lead.
    modes = ("global", "follow")
    out = {mode: {str(b): {"all": new_bucket(), "lead": new_bucket(), "follow": new_bucket(),
                             "by_candidate_bin": {}} for b in biases} for mode in modes}
    files = files_for(data)
    with torch.inference_mode():
        for path in files:
            with np.load(path, allow_pickle=False) as d:
                state, tokens, lengths = d["state"], d["tokens"], d["lengths"]
                actions, offsets, positive = d["actions"], d["offsets"], d["positives"]
                rows = np.arange(len(lengths), dtype=np.int64)
                for start in range(0, len(rows), args.batch_size):
                    chosen = rows[start:start + args.batch_size]
                    batch, positive_pad, counts = collate(chosen, state, tokens, lengths, actions, offsets, positive)
                    scores = model(**{k: v.to(device) for k, v in batch.items()}).cpu().numpy()
                    action_pad = batch["actions"].numpy()
                    for j, row in enumerate(chosen):
                        n = counts[j]; sc0 = scores[j, :n].astype(np.float64)
                        aa = action_pad[j, :n]; pp = positive_pad[j, :n]
                        pass_mask = aa[:, 126] > .5
                        demo_pass = bool((pass_mask & pp).any())
                        lead = bool(state[row, 125] > .5)
                        context = "lead" if lead else "follow"
                        demo_sizes = np.rint(aa[pp, 120] * 10).astype(int)
                        demo_ge4 = bool((demo_sizes >= 4).any())
                        for mode in modes:
                            for bias in biases:
                                key = str(bias); row_out = out[mode][key]
                                sc = sc0.copy()
                                if pass_mask.any() and (mode == "global" or context == "follow"):
                                    sc[pass_mask] += bias
                                winner = int(sc.argmax())
                                pred_pass = bool(pass_mask[winner])
                                correct = bool(pp[winner])
                                pred_size = int(round(float(aa[winner, 120]) * 10))
                                same_size = bool(demo_sizes.size and pred_size in set(demo_sizes.tolist()))
                                for name, bucket in (("all", row_out["all"]), (context, row_out[context])):
                                    bucket["rows"] += 1; bucket["correct"] += int(correct)
                                    bucket["choice_rows"] += int(n > 1); bucket["choice_correct"] += int(correct and n > 1)
                                    bucket["demo_pass"] += int(demo_pass); bucket["pred_pass"] += int(pred_pass)
                                    bucket["false_pass"] += int(pred_pass and not demo_pass)
                                    bucket["demo_size_ge4"] += int(demo_ge4)
                                    bucket["size_ge4_correct"] += int(demo_ge4 and correct)
                                    bucket["size_ge4_same"] += int(demo_ge4 and same_size)
                                bname = count_bin(n)
                                bin_out = row_out["by_candidate_bin"].setdefault(bname, new_bucket())
                                bin_out["rows"] += 1; bin_out["correct"] += int(correct)
                                bin_out["choice_rows"] += int(n > 1); bin_out["choice_correct"] += int(correct and n > 1)
                                bin_out["demo_pass"] += int(demo_pass); bin_out["pred_pass"] += int(pred_pass)
                                bin_out["false_pass"] += int(pred_pass and not demo_pass)
                                bin_out["demo_size_ge4"] += int(demo_ge4)
                                bin_out["size_ge4_correct"] += int(demo_ge4 and correct)
                                bin_out["size_ge4_same"] += int(demo_ge4 and same_size)
    def finish(bucket: dict) -> dict:
        r = bucket["rows"]
        bucket["accuracy"] = bucket["correct"] / max(1, r)
        bucket["choice_accuracy"] = bucket["choice_correct"] / max(1, bucket["choice_rows"])
        bucket["demo_pass_rate"] = bucket["demo_pass"] / max(1, r)
        bucket["pred_pass_rate"] = bucket["pred_pass"] / max(1, r)
        bucket["pass_rate_error"] = bucket["pred_pass_rate"] - bucket["demo_pass_rate"]
        bucket["false_pass_rate"] = bucket["false_pass"] / max(1, r - bucket["demo_pass"])
        bucket["size_ge4_accuracy"] = bucket["size_ge4_correct"] / max(1, bucket["demo_size_ge4"])
        bucket["size_ge4_same_size_recall"] = bucket["size_ge4_same"] / max(1, bucket["demo_size_ge4"])
        return bucket
    for mode in modes:
        for row in out[mode].values():
            for key in ("all", "lead", "follow"):
                finish(row[key])
            for bucket in row["by_candidate_bin"].values(): finish(bucket)
    return {"schema": "oxbot-pass-bias-sweep-v1", "status": "complete", "split": "validation",
            "test_data_opened": False, "checkpoint_sha256": sha256(checkpoint),
            "data_manifest_sha256": sha256(data / "manifest.json"), "biases": biases,
            "modes": out}


def markdown(report: dict) -> str:
    lines = ["# Frozen v1 pass-score bias sweep (validation only)", "",
             "The checkpoint is unchanged. Every row comes from the prepared validation split; held-out test data was not opened.", "",
             "## Overall and follow metrics", "",
             "| mode | bias | raw accuracy | choice accuracy | demo pass | predicted pass | pass error | false pass on play | size >=4 exact | size >=4 same-size |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for mode, values in report["modes"].items():
        for bias, row in values.items():
            x = row["all"]; f = row["follow"]
            lines.append(f"| {mode} | {bias} | {x['accuracy']:.2%} | {x['choice_accuracy']:.2%} | {f['demo_pass_rate']:.2%} | {f['pred_pass_rate']:.2%} | {f['pass_rate_error']:+.2%} | {f['false_pass_rate']:.2%} | {f['size_ge4_accuracy']:.2%} | {f['size_ge4_same_size_recall']:.2%} |")
    lines += ["", "Pass bias only changes rows containing a legal pass candidate (follow rows); lead rows are unchanged. The useful choice is the smallest negative bias that reduces false passes without sacrificing exact multi-card choices.", "", "## Candidate-count bins (global mode)", "", "| bias | bin | raw accuracy | choice accuracy | predicted pass | pass error |", "|---:|---|---:|---:|---:|---:|"]
    for bias, row in report["modes"]["global"].items():
        for b in ("1", "2-3", "4-8", "9-16", "17-32", "33-64", "65-128", "129-256", "257-512", "513+"):
            if b not in row["by_candidate_bin"]: continue
            x = row["by_candidate_bin"][b]
            lines.append(f"| {bias} | {b} | {x['accuracy']:.2%} | {x['choice_accuracy']:.2%} | {x['pred_pass']/x['rows']:.2%} | {(x['pred_pass']-x['demo_pass'])/x['rows']:+.2%} |")
    lines += ["", "Interpretation: this is a calibration diagnostic. A bias that improves validation pass frequency while leaving size >=4 exact recall and high-candidate bins unchanged is a candidate for a reversible policy experiment; it does not establish gameplay strength.", ""]
    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, default=ROOT / "data/processed/bc-v1")
    p.add_argument("--checkpoint", type=Path, default=ROOT / "models/bc-v1/best.pt")
    p.add_argument("--output", type=Path, default=ROOT / "reports/bc-v1-pass-bias-sweep.json")
    p.add_argument("--markdown", type=Path, default=ROOT / "reports/bc-v1-pass-bias-sweep.md")
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--bias", type=float, nargs="+", default=[-1.0, -.75, -.5, -.25, 0., .25, .5])
    args = p.parse_args()
    report = run(args)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    args.markdown.write_text(markdown(report), encoding="utf-8")
    print(json.dumps({"status": report["status"], "test_data_opened": False,
                      "biases": report["biases"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
