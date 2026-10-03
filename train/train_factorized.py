"""Research BC with factorized action-kind/size and pass-mass auxiliaries.

The release scorer and 128-dimensional feature contract are unchanged.  For
each row the observed positive candidate supplies a kind and action-size
label; auxiliary cross-entropies train the *mass* of candidate scores grouped
by kind and size.  A separate optional BCE trains the pass-vs-play mass on
follow rows.  Only the train and validation shards are opened.  The
kind/size terms are deliberately small and ignore pass rows by default so a
coarse label cannot overwhelm concrete behavioral cloning.
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import hashlib
import json
import math
from pathlib import Path
import random
import time

import numpy as np
import torch
import torch.nn.functional as F

from model import CandidateModel, ModelConfig
from features import FEATURE_VERSION
from train_bc import (atomic_json, batches, marginal_loss, pass_mass_loss,
                      pass_mass_terms, sha256_file, shard_paths)


def _mass_logits(scores: torch.Tensor, actions: torch.Tensor, mask: torch.Tensor,
                 feature: str, maximum: int) -> torch.Tensor:
    """Aggregate candidate logits by action kind or physical size."""
    if feature == "kind":
        index = actions[..., 108:120].argmax(dim=-1)
    elif feature == "size":
        index = torch.round(actions[..., 120] * 10.0).to(dtype=torch.long).clamp(0, maximum)
    else:
        raise ValueError("unknown factorized feature")
    values = []
    for value in range(maximum + 1):
        eligible = mask & index.eq(value)
        values.append(scores.masked_fill(~eligible, float("-inf")).logsumexp(dim=-1))
    return torch.stack(values, dim=-1)


def factorized_auxiliary(scores: torch.Tensor, actions: torch.Tensor,
                         positives: torch.Tensor, mask: torch.Tensor,
                         kind_weight: float, size_weight: float,
                         nonpass_only: bool, pass_mass_beta: float = 0.0):
    """Return BC plus factorized auxiliaries and pass-mass diagnostics.

    ``pass_mass_beta`` applies the same follow-row pass-vs-play BCE used by
    the baseline trainer.  It is deliberately separate from ``include_pass``:
    the latter controls coarse kind/size labels, while pass mass is exactly
    the signal that is meaningful on rows containing both pass and play
    candidates.  The helper always reports pass-mass diagnostics so a zero
    beta candidate remains directly comparable to the existing factorized
    anchor.
    """
    if not math.isfinite(pass_mass_beta) or pass_mass_beta < 0.0:
        raise ValueError("pass_mass_beta must be finite and nonnegative")
    pos_index = positives.to(dtype=torch.float32).argmax(dim=-1)
    target_kind = actions[..., 108:120].argmax(dim=-1).gather(1, pos_index[:, None]).squeeze(1)
    target_size = torch.round(actions[..., 120] * 10.0).to(dtype=torch.long).gather(1, pos_index[:, None]).squeeze(1)
    eligible = (target_kind != 0) if nonpass_only else torch.ones_like(target_kind, dtype=torch.bool)
    kind_loss = scores.new_zeros((), dtype=torch.float32)
    size_loss = scores.new_zeros((), dtype=torch.float32)
    if bool(eligible.any()):
        kind_logits = _mass_logits(scores, actions, mask, "kind", 11)
        size_logits = _mass_logits(scores, actions, mask, "size", 27)
        kind_loss = F.cross_entropy(kind_logits[eligible].float(), target_kind[eligible])
        size_loss = F.cross_entropy(size_logits[eligible].float(), target_size[eligible])
    base = marginal_loss(scores, positives)
    pass_aux, pass_eligible, pass_positive_rows = pass_mass_loss(scores, actions, positives)
    pass_mass_samples = int(pass_eligible.sum().item())
    pass_mass_correct = 0
    if pass_mass_samples:
        pass_logits, _, pass_labels = pass_mass_terms(scores, actions, positives)
        pass_mass_correct = int(((pass_logits[pass_eligible] > 0) ==
                                 pass_labels[pass_eligible].bool()).sum().item())
    loss = base + kind_weight * kind_loss + size_weight * size_loss
    if pass_mass_beta:
        loss = loss + pass_mass_beta * pass_aux
    return loss, {"bc_nll": float(base.detach()), "kind_loss": float(kind_loss.detach()),
                  "size_loss": float(size_loss.detach()), "aux_rows": int(eligible.sum().item()),
                  "target_kind_nonpass": int((target_kind != 0).sum().item()),
                  "pass_mass_bce": float(pass_aux.detach()),
                  "pass_mass_samples": pass_mass_samples,
                  "pass_mass_positive_rows": int(pass_positive_rows),
                  "pass_mass_correct": pass_mass_correct}


def fast_measure(model, files, batch_size, device):
    """Validation metrics needed for checkpoint selection, without test data."""
    model.eval()
    totals = {"samples": 0, "correct": 0, "choice_samples": 0, "choice_correct": 0,
              "nll_sum": 0.0, "pass_samples": 0, "pass_correct": 0,
              "play_samples": 0, "play_correct": 0}
    with torch.inference_mode():
        for batch in batches(files, batch_size, 0, False):
            batch = {key: value.to(device) for key, value in batch.items()}
            positives = batch.pop("positives")
            scores = model(**batch)
            losses = marginal_loss(scores, positives, "none")
            winners = scores.argmax(dim=-1)
            correct = positives.gather(1, winners[:, None]).squeeze(1)
            choices = batch["mask"].sum(-1) > 1
            passes = (batch["actions"][..., 126].bool() & positives).any(-1)
            totals["samples"] += len(losses)
            totals["nll_sum"] += float(losses.sum())
            totals["correct"] += int(correct.sum())
            totals["choice_samples"] += int(choices.sum())
            totals["choice_correct"] += int((choices & correct).sum())
            for name, selected in (("pass", passes), ("play", ~passes)):
                totals[name + "_samples"] += int(selected.sum())
                totals[name + "_correct"] += int((selected & correct).sum())
    totals["nll"] = totals["nll_sum"] / max(1, totals["samples"])
    totals["accuracy"] = totals["correct"] / max(1, totals["samples"])
    totals["choice_accuracy"] = totals["choice_correct"] / max(1, totals["choice_samples"])
    totals["pass_accuracy"] = totals["pass_correct"] / max(1, totals["pass_samples"])
    totals["play_accuracy"] = totals["play_correct"] / max(1, totals["play_samples"])
    return totals


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data/processed/bc-v2-full"))
    parser.add_argument("--output", type=Path, default=Path("models/bc-v3-factorized005"))
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=20261001)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--fp32", action="store_true")
    parser.add_argument("--kind-weight", type=float, default=0.05)
    parser.add_argument("--size-weight", type=float, default=0.05)
    parser.add_argument("--pass-mass-beta", type=float, default=0.0,
                        help="follow-row pass-vs-play mass BCE weight (default: 0)")
    parser.add_argument("--include-pass", action="store_true", help="include pass rows in coarse auxiliaries")
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.kind_weight < 0 or args.size_weight < 0:
        raise ValueError("invalid training arguments")
    if not math.isfinite(args.pass_mass_beta) or args.pass_mass_beta < 0.0:
        raise ValueError("pass-mass-beta must be finite and nonnegative")
    if (args.output / "best.pt").exists():
        raise FileExistsError("choose a new output directory")
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    torch.set_num_threads(4)
    device = torch.device(args.device)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable")
        torch.cuda.manual_seed_all(args.seed); torch.backends.cuda.matmul.allow_tf32 = False
    train_files = shard_paths(args.data, "train")
    validation_files = shard_paths(args.data, "validation")
    data_manifest = args.data / "manifest.json"
    if not data_manifest.is_file():
        raise ValueError("prepared manifest is required")
    manifest = json.loads(data_manifest.read_text(encoding="utf-8"))
    if manifest.get("status") != "complete" or manifest.get("feature_version") != FEATURE_VERSION:
        raise ValueError("incompatible prepared manifest")
    expected = {item["path"]: item["sha256"] for split in ("train", "validation") for item in manifest["splits"][split]["shards"]}
    actual = {path.relative_to(args.data).as_posix(): sha256_file(path) for path in train_files + validation_files}
    if actual != expected:
        raise ValueError("prepared shard hashes do not match manifest")
    model = CandidateModel(ModelConfig()).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=.01)
    args.output.mkdir(parents=True, exist_ok=True)
    provenance = {"purpose": "research_factorized_behavioral_cloning", "seed": args.seed,
                  "data_manifest_sha256": sha256_file(data_manifest), "input_shards": actual,
                  "train_script_sha256": sha256_file(Path(__file__)), "model_script_sha256": sha256_file(Path(__file__).with_name("model.py")),
                  "torch": str(torch.__version__), "device": str(device), "epochs_requested": args.epochs,
                  "batch_size": args.batch_size, "learning_rate": args.lr,
                  "kind_weight": args.kind_weight, "size_weight": args.size_weight,
                  "pass_mass_beta": args.pass_mass_beta,
                  "pass_mass_definition": "BCE(logsumexp(pass)-logsumexp(play)) on rows with finite pass and play candidates",
                  "nonpass_only": not args.include_pass,
                  "objective": "marginal_bc_plus_factorized_kind_and_size_mass_ce_plus_pass_mass_bce",
                  "test_used": False, "release_eligible": False}
    report = {"schema": "oxbot-factorized-bc-training-v1", "status": "running", "provenance": provenance,
              "epochs": []}
    atomic_json(args.output / "training.json", report)
    best = float("inf"); started = time.monotonic()
    mixed = device.type == "cuda" and not args.fp32
    global_step = 0
    for epoch in range(args.epochs):
        model.train(); sums = {"loss": 0., "bc_nll": 0., "kind_loss": 0., "size_loss": 0., "samples": 0, "aux_rows": 0,
                               "pass_mass_bce": 0., "pass_mass_samples": 0,
                               "pass_mass_positive_rows": 0, "pass_mass_correct": 0}
        for batch in batches(train_files, args.batch_size, args.seed + epoch, True):
            batch = {key: value.to(device) for key, value in batch.items()}
            positives = batch.pop("positives")
            optimizer.zero_grad(set_to_none=True)
            # The baseline trainer warms up over the first 100 updates. Keep
            # the same stable cosine-by-epoch schedule for fair comparison.
            warmup = min(1.0, (global_step + 1) / 100.0)
            decay = .15 + .85 * .5 * (1 + math.cos(math.pi * epoch / args.epochs))
            for group in optimizer.param_groups:
                group["lr"] = args.lr * warmup * decay
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16) if mixed else nullcontext():
                scores = model(**batch)
                loss, info = factorized_auxiliary(scores, batch["actions"], positives,
                                                  batch["mask"], args.kind_weight,
                                                  args.size_weight, not args.include_pass,
                                                  args.pass_mass_beta)
            if not torch.isfinite(loss): raise RuntimeError("nonfinite factorized loss")
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True); optimizer.step()
            n = len(positives); sums["loss"] += float(loss.detach()) * n; sums["bc_nll"] += info["bc_nll"] * n; sums["kind_loss"] += info["kind_loss"] * n; sums["size_loss"] += info["size_loss"] * n; sums["samples"] += n; sums["aux_rows"] += info["aux_rows"]
            sums["pass_mass_bce"] += info["pass_mass_bce"] * info["pass_mass_samples"]
            sums["pass_mass_samples"] += info["pass_mass_samples"]
            sums["pass_mass_positive_rows"] += info["pass_mass_positive_rows"]
            sums["pass_mass_correct"] += info["pass_mass_correct"]
            global_step += 1
        validation = fast_measure(model, validation_files, args.batch_size, device)
        row = {"epoch": epoch + 1, "train_loss": sums["loss"] / max(1, sums["samples"]), "train_bc_nll": sums["bc_nll"] / max(1, sums["samples"]), "train_kind_loss": sums["kind_loss"] / max(1, sums["samples"]), "train_size_loss": sums["size_loss"] / max(1, sums["samples"]), "train_aux_rows": sums["aux_rows"],
               "train_pass_mass_bce": sums["pass_mass_bce"] / max(1, sums["pass_mass_samples"]),
               "train_pass_mass_samples": sums["pass_mass_samples"],
               "train_pass_mass_positive_rows": sums["pass_mass_positive_rows"],
               "train_pass_mass_accuracy": sums["pass_mass_correct"] / max(1, sums["pass_mass_samples"]),
               "validation": validation, "elapsed_seconds": time.monotonic() - started}
        report["epochs"].append(row); print(json.dumps({"event": "epoch", **row}), flush=True)
        if validation["nll"] < best:
            best = validation["nll"]; report["selected_epoch"] = epoch + 1
            torch.save({"state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()}, "config": model.architecture(), "provenance": provenance, "validation": validation, "epoch": epoch + 1}, args.output / "best.pt")
        atomic_json(args.output / "training.json", report)
    report["status"] = "complete"; report["elapsed_seconds"] = time.monotonic() - started; atomic_json(args.output / "training.json", report)
    print(json.dumps({"status": "complete", "selected_epoch": report.get("selected_epoch"), "validation": report["epochs"][report.get("selected_epoch", 1) - 1]["validation"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
