"""Research-only grouped behavioral cloning for the fixed C++ scorer.

This trainer keeps the release feature/model contract unchanged and reads only
the prepared train and validation shards.  Concrete deck-copy/claim variants
are collapsed into the same semantic candidate group during the objective:

    L = L_group + within_weight * L_within

where a group's logit is the log-mean-exp of its member logits, and the target
group is positive when any concrete member is a demonstrated positive.  The
checkpoint remains exportable by ``train/export_model.py``; inference can use
the existing C++ ``group-logmeanexp`` selection policy.  Held-out test shards
are deliberately never opened.

The script is separate from train_bc.py so release training remains unchanged
until a candidate passes validation and strength gates.
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
from pathlib import Path
import hashlib
import json
import math
import random
import time

import numpy as np
import torch
import torch.nn.functional as F

from model import CandidateModel, ModelConfig
from features import FEATURE_VERSION
from train_bc import (
    atomic_json,
    batches,
    marginal_loss,
    measure,
    prepared_sample_alignment,
    sha256_file,
    shard_paths,
)


def _group_keys(actions: torch.Tensor) -> list[tuple]:
    """Return deterministic semantic keys for one row's action features.

    The encoding mirrors the C++ ``group_key`` contract: kind, physical
    action length, key/secondary, rank multiplicity, current-level wildcard
    count, and a full face/suit signature only for straight flushes.  Ordinary
    actions therefore collapse duplicate physical deck assignments exactly as
    the C++ selector does, while straight-flush suits remain distinct.
    """
    if actions.ndim != 2 or actions.shape[1] < 125:
        raise ValueError("invalid action feature row")
    result: list[tuple] = []
    # These features are exact multiples of 0.5/0.1/1/14/8 in the v1
    # contract.  Round after moving to CPU so BF16 autocast cannot perturb
    # grouping identities.
    values = actions.detach().float().cpu().numpy()
    for row in values:
        faces = tuple(np.rint(row[:54] * 2.0).astype(np.int16).tolist())
        kind = int(np.argmax(row[108:120]))
        length = int(round(float(row[120]) * 10.0))
        key = int(round(float(row[121]) * 14.0))
        secondary = int(round(float(row[122]) * 14.0))
        # C++ group_key counts only the heart card at the current level;
        # action feature 123 stores this count divided by two.  Feature 124
        # is all four suits at that rank and must not be used here.
        wild = int(round(float(row[123]) * 2.0))
        rank_counts = [0] * 15
        for face, count in enumerate(faces):
            if count:
                rank = face // 4 if face < 52 else 13 + face - 52
                rank_counts[rank] += count
        straight_flush = kind == 10  # KINDS index for straight_flush
        signature = faces if straight_flush else (0,) * 54
        result.append((kind, length, key, secondary, tuple(rank_counts), wild,
                       straight_flush, signature))
    return result


def grouped_scores(scores: torch.Tensor, actions: torch.Tensor, mask: torch.Tensor,
                   positives: torch.Tensor):
    """Build row-wise group log-mean-exp scores and positive group labels."""
    if scores.ndim != 2 or actions.shape[:2] != scores.shape or mask.shape != scores.shape or positives.shape != scores.shape:
        raise ValueError("grouped score tensor shape mismatch")
    batch, count = scores.shape
    grouped: list[torch.Tensor] = []
    positive_groups: list[torch.Tensor] = []
    member_groups: list[list[list[int]]] = []
    for row in range(batch):
        valid = int(mask[row].sum().item())
        keys = _group_keys(actions[row, :valid])
        indexes: dict[tuple, int] = {}
        members: list[list[int]] = []
        for index, key in enumerate(keys):
            group = indexes.get(key)
            if group is None:
                group = len(members)
                indexes[key] = group
                members.append([])
            members[group].append(index)
        values = []
        for member in members:
            idx = torch.as_tensor(member, device=scores.device, dtype=torch.long)
            values.append(torch.logsumexp(scores[row, idx].float(), dim=0) - math.log(len(member)))
        # Derive labels from an explicit positive mask below; never infer them
        # from score values or padded candidates.
        labels = []
        for member in members:
            idx = torch.as_tensor(member, device=scores.device, dtype=torch.long)
            labels.append(bool(torch.any(positives[row, idx]).item()))
        grouped.append(torch.stack(values))
        positive_groups.append(torch.as_tensor(labels, device=scores.device, dtype=torch.bool))
        member_groups.append(members)
    return grouped, positive_groups, member_groups


def grouped_loss(scores: torch.Tensor, actions: torch.Tensor, positives: torch.Tensor,
                 mask: torch.Tensor, within_weight: float):
    """Compute grouped objective and diagnostics without reading test data."""
    if positives.shape != scores.shape:
        raise ValueError("positive shape mismatch")
    values, group_positive, members = grouped_scores(scores, actions, mask, positives)
    group_total = scores.new_zeros((), dtype=torch.float32)
    within_total = scores.new_zeros((), dtype=torch.float32)
    group_count = positive_group_count = within_groups = within_correct = within_pairs = 0
    for row, (row_scores, row_positive, row_members) in enumerate(zip(values, group_positive, members, strict=True)):
        group_total = group_total + (torch.logsumexp(row_scores, dim=0) - torch.logsumexp(row_scores[row_positive], dim=0))
        group_count += int(row_scores.numel())
        positive_group_count += int(row_positive.sum().item())
        for member in row_members:
            idx = torch.as_tensor(member, device=scores.device, dtype=torch.long)
            member_scores = scores[row, idx].float()
            member_positive = positives[row, idx]
            if bool(member_positive.any()):
                within_total = within_total + (torch.logsumexp(member_scores, dim=0) - torch.logsumexp(member_scores[member_positive], dim=0))
                within_groups += 1
                within_correct += int(bool(member_positive[member_scores.argmax()].item()))
                within_pairs += int(member_scores.numel())
    rows = max(1, scores.shape[0])
    group_term = group_total / rows
    within_term = within_total / max(1, within_groups)
    loss = group_term + within_weight * within_term
    return loss, {"group_nll": float(group_term.detach()), "within_nll": float(within_term.detach()),
                  "groups": group_count, "positive_groups": positive_group_count,
                  "within_groups": within_groups, "within_correct": within_correct,
                  "within_pairs": within_pairs}


def grouped_argmax(scores: torch.Tensor, actions: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Select the highest member in the highest log-mean-exp group."""
    selected = []
    for row in range(scores.shape[0]):
        valid = int(mask[row].sum().item())
        keys = _group_keys(actions[row, :valid])
        groups: dict[tuple, list[int]] = {}
        for index, key in enumerate(keys):
            groups.setdefault(key, []).append(index)
        best_index = 0
        best_group = -float("inf")
        for member in groups.values():
            idx = torch.as_tensor(member, device=scores.device, dtype=torch.long)
            value = float((torch.logsumexp(scores[row, idx].float(), dim=0) - math.log(len(member))).item())
            if value > best_group:
                best_group, best_index = value, int(member[int(torch.argmax(scores[row, idx]).item())])
        selected.append(best_index)
    return torch.as_tensor(selected, device=scores.device, dtype=torch.long)


def evaluate_group(model, files, batch_size, device):
    model.eval()
    totals = {"samples": 0, "correct": 0, "choice_samples": 0, "choice_correct": 0,
              "nll_sum": 0.0, "raw_correct": 0, "group_correct": 0, "group_choice_correct": 0}
    with torch.inference_mode():
        for batch in batches(files, batch_size, 0, False):
            batch = {key: value.to(device) for key, value in batch.items()}
            positives = batch.pop("positives")
            scores = model(**batch)
            selected = grouped_argmax(scores, batch["actions"], batch["mask"])
            correct = positives.gather(1, selected[:, None]).squeeze(1)
            raw = scores.argmax(-1)
            raw_correct = positives.gather(1, raw[:, None]).squeeze(1)
            losses = marginal_loss(scores, positives, "none")
            choices = batch["mask"].sum(-1) > 1
            totals["samples"] += len(losses)
            totals["nll_sum"] += float(losses.sum())
            totals["correct"] += int(correct.sum())
            totals["raw_correct"] += int(raw_correct.sum())
            totals["group_correct"] += int(correct.sum())
            totals["choice_samples"] += int(choices.sum())
            totals["choice_correct"] += int((choices & correct).sum())
            totals["group_choice_correct"] += int((choices & raw_correct).sum())
    totals["nll"] = totals["nll_sum"] / max(1, totals["samples"])
    totals["group_accuracy"] = totals["group_correct"] / max(1, totals["samples"])
    totals["raw_accuracy"] = totals["raw_correct"] / max(1, totals["samples"])
    totals["group_choice_accuracy"] = totals["choice_correct"] / max(1, totals["choice_samples"])
    totals["raw_choice_accuracy"] = totals["group_choice_correct"] / max(1, totals["choice_samples"])
    return totals


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data/processed/bc-v2-full"))
    parser.add_argument("--output", type=Path, default=Path("models/bc-v3-group02"))
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=20261001)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--fp32", action="store_true")
    parser.add_argument("--within-weight", type=float, default=0.2)
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.within_weight < 0 or not math.isfinite(args.within_weight):
        raise ValueError("invalid training arguments")
    if (args.output / "best.pt").exists():
        raise FileExistsError("choose a new output directory")
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    torch.set_num_threads(4)
    device = torch.device(args.device)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable")
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cuda.matmul.allow_tf32 = False
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
    provenance = {"purpose": "research_grouped_behavioral_cloning", "seed": args.seed,
                  "data_manifest_sha256": sha256_file(data_manifest), "input_shards": actual,
                  "train_script_sha256": sha256_file(Path(__file__)), "model_script_sha256": sha256_file(Path(__file__).with_name("model.py")),
                  "torch": str(torch.__version__), "device": str(device), "epochs_requested": args.epochs,
                  "batch_size": args.batch_size, "learning_rate": args.lr, "within_weight": args.within_weight,
                  "objective": "group_logmeanexp_plus_within_positive_group_marginal", "test_used": False,
                  "release_eligible": False}
    baseline = measure(model, validation_files, args.batch_size, device)
    report = {"schema": "oxbot-grouped-bc-training-v1", "status": "running", "provenance": provenance,
              "initial_validation_raw": baseline, "epochs": []}
    atomic_json(args.output / "training.json", report)
    best = float("inf"); started = time.monotonic(); mixed = device.type == "cuda" and not args.fp32
    for epoch in range(args.epochs):
        model.train(); sums = {"loss": 0.0, "group_nll": 0.0, "within_nll": 0.0, "samples": 0, "within_groups": 0, "within_correct": 0, "within_pairs": 0}
        for batch in batches(train_files, args.batch_size, args.seed + epoch, True):
            batch = {key: value.to(device) for key, value in batch.items()}
            positives = batch.pop("positives")
            optimizer.zero_grad(set_to_none=True)
            warmup = min(1.0, (epoch + 1) / max(1, args.epochs))
            for group in optimizer.param_groups: group["lr"] = args.lr * warmup
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16) if mixed else nullcontext():
                scores = model(**batch)
                loss, info = grouped_loss(scores, batch["actions"], positives, batch["mask"], args.within_weight)
            if not torch.isfinite(loss): raise RuntimeError("nonfinite grouped loss")
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True); optimizer.step()
            n = len(positives); sums["loss"] += float(loss.detach()) * n; sums["group_nll"] += info["group_nll"] * n; sums["within_nll"] += info["within_nll"] * n; sums["samples"] += n
            sums["within_groups"] += info["within_groups"]; sums["within_correct"] += info["within_correct"]; sums["within_pairs"] += info["within_pairs"]
        validation = evaluate_group(model, validation_files, args.batch_size, device)
        row = {"epoch": epoch + 1, "train_loss": sums["loss"] / max(1, sums["samples"]), "train_group_nll": sums["group_nll"] / max(1, sums["samples"]), "train_within_nll": sums["within_nll"] / max(1, sums["samples"]), "train_within_accuracy": sums["within_correct"] / max(1, sums["within_pairs"]), "validation": validation, "elapsed_seconds": time.monotonic() - started}
        report["epochs"].append(row); print(json.dumps({"event": "epoch", **row}), flush=True)
        # Select only on validation grouped NLL; test remains unopened.
        if validation["nll"] < best:
            best = validation["nll"]
            torch.save({"state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()}, "config": model.architecture(), "provenance": provenance, "validation": validation, "epoch": epoch + 1}, args.output / "best.pt")
            report["selected_epoch"] = epoch + 1
        atomic_json(args.output / "training.json", report)
    report["status"] = "complete"; report["elapsed_seconds"] = time.monotonic() - started; atomic_json(args.output / "training.json", report)
    print(json.dumps({"status": "complete", "selected_epoch": report.get("selected_epoch"), "validation": report["epochs"][report.get("selected_epoch", 1) - 1]["validation"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
