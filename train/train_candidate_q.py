"""Research-only candidate-level outcome/Q ranking on the audited BC shards.

This experiment keeps the release model contract unchanged.  It starts from a
BC checkpoint and adds a conservative, candidate-level pairwise objective:

* the demonstrated (positive) candidate is paired with a deterministic cap of
  legal negatives from the *same* information set;
* on rows whose eventual team won, the observed candidate is the preferred
  action; on rows whose team lost, the direction is reversed (an offline
  counterfactual ranking hypothesis, not a measured counterfactual return);
* a small regression term anchors positive candidate scores to the observed
  terminal team return (+1/-1), while the ordinary multi-positive BC loss is
  retained as the conservative anchor.

The direction-reversed rows are intentionally explicit and auditable.  They do
not claim that every unobserved legal move would have won; this is an offline
Q/advantage *probe* only.  The held-out ``test`` split is never opened.  The
resulting checkpoint is not a release artifact and is not consumed by the
BotZone package unless it independently passes all release gates.
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
from dataclasses import asdict
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
from train_bc import (batches, marginal_loss, measure, sha256_file, shard_paths,
                      prepared_sample_alignment)


SCHEMA = "oxbot-candidate-outcome-q-experiment-v1"


def atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def load_outcome_targets(path: Path, data_manifest: Path, alignment: dict[str, dict[str, int]]) -> tuple[dict[str, int], str]:
    """Fail-closed loading of the existing train/validation-only outcome map."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != "oxbot-action-utility-targets-v1":
        raise ValueError("outcome target schema is incompatible")
    if payload.get("test_used") is not False or payload.get("splits_read") != ["train", "validation"]:
        raise ValueError("outcome targets must be train/validation-only")
    manifest = json.loads(data_manifest.read_text(encoding="utf-8"))
    if payload.get("source_manifest_sha256") != manifest.get("source_manifest_sha256"):
        raise ValueError("outcome target source manifest does not match prepared data")
    expected_splits = {s: manifest.get("source_splits_sha256", {}).get(s)
                       for s in ("train", "validation")}
    if payload.get("source_split_sha256") != expected_splits:
        raise ValueError("outcome target source split SHA256 does not match prepared data")
    values = payload.get("targets")
    if not isinstance(values, dict) or not values:
        raise ValueError("outcome target map is empty")
    # Alignment is read only from train and validation shards.  It catches
    # stale/foreign IDs without ever touching the held-out test shard.
    expected_ids = set(alignment["train"]) | set(alignment["validation"])
    missing = expected_ids - set(values)
    if missing:
        raise ValueError(f"outcome target map misses {len(missing)} prepared rows")
    if any(type(v) is not int or v not in (0, 1) for v in values.values()):
        raise ValueError("outcome targets must be binary 0/1")
    return {key: int(values[key]) for key in expected_ids}, sha256_file(path)


def outcome_pairwise_loss(scores: torch.Tensor, positives: torch.Tensor,
                          mask: torch.Tensor, returns: torch.Tensor,
                          max_pairs: int = 16, margin: float = 0.10):
    """Pair observed candidates against legal negatives at candidate level.

    ``returns`` is +1 for an eventual team win and -1 for a loss.  Positive
    candidates are kept as one face/claim-equivalent representative.  For a
    winning row we optimize ``positive > negative``; for a losing row the
    ranking is deliberately reversed as a counterfactual offline hypothesis.
    Pair count is capped deterministically so large lead rows cannot dominate.
    """
    if scores.ndim != 2 or positives.shape != scores.shape or mask.shape != scores.shape:
        raise ValueError("invalid outcome-pairwise shapes")
    if returns.shape != (scores.shape[0],):
        raise ValueError("return shape does not match batch")
    total = scores.new_zeros((), dtype=torch.float32)
    pair_count = row_count = correct = 0
    for row in range(scores.shape[0]):
        valid = mask[row]
        pos = torch.nonzero(valid & positives[row], as_tuple=False).flatten()
        neg = torch.nonzero(valid & ~positives[row], as_tuple=False).flatten()
        if pos.numel() == 0 or neg.numel() == 0:
            continue
        # Claim variants have identical face-level features.  A single
        # representative prevents multiplying the same pair by deck copies.
        pi = int(pos[0])
        neg = neg[:max_pairs]
        # direction = +1 means observed > negative; -1 means negative > obs.
        direction = returns[row].to(dtype=torch.float32)
        delta = direction * (scores[row, pi] - scores[row, neg])
        total = total + F.softplus(margin - delta).float().sum()
        pair_count += int(neg.numel())
        row_count += 1
        correct += int((delta > 0).sum())
    if pair_count == 0:
        return scores.new_zeros((), dtype=torch.float32), 0, 0, 0
    return total / pair_count, pair_count, row_count, correct


def positive_value_loss(scores: torch.Tensor, positives: torch.Tensor,
                        returns: torch.Tensor) -> torch.Tensor:
    """Huber regression of observed candidate scores to +/-1 terminal return."""
    target = returns[:, None].expand_as(scores)
    selected = scores[positives]
    selected_target = target[positives]
    if selected.numel() == 0:
        return scores.new_zeros((), dtype=torch.float32)
    return F.smooth_l1_loss(selected.float(), selected_target.float())


def evaluate_q(model, files, outcome_targets, batch_size, device, max_pairs, margin):
    """Validation-only diagnostics; no test shard is opened."""
    totals = {"rows": 0, "pair_count": 0, "pair_correct": 0, "pair_rows": 0,
              "value_count": 0, "value_abs_error": 0.0, "win_positive_sum": 0.0,
              "loss_positive_sum": 0.0, "win_positive_count": 0, "loss_positive_count": 0}
    model.eval()
    with torch.inference_mode():
        for batch in batches(files, batch_size, 0, False, include_ids=True):
            ids = batch.pop("ids")
            returns = torch.as_tensor([1.0 if outcome_targets[x] else -1.0 for x in ids],
                                      dtype=torch.float32, device=device)
            batch = {key: value.to(device) for key, value in batch.items()}
            positives = batch.pop("positives")
            scores = model(**batch)
            _, pairs, pair_rows, pair_correct = outcome_pairwise_loss(
                scores, positives, batch["mask"], returns, max_pairs, margin)
            totals["rows"] += len(ids)
            totals["pair_count"] += pairs
            totals["pair_rows"] += pair_rows
            totals["pair_correct"] += pair_correct
            values = scores[positives].float()
            labels = returns[:, None].expand_as(scores)[positives]
            totals["value_count"] += int(values.numel())
            totals["value_abs_error"] += float((values - labels).abs().sum())
            win = labels > 0
            totals["win_positive_sum"] += float(values[win].sum())
            totals["loss_positive_sum"] += float(values[~win].sum())
            totals["win_positive_count"] += int(win.sum())
            totals["loss_positive_count"] += int((~win).sum())
    totals["pair_accuracy"] = totals["pair_correct"] / max(1, totals["pair_count"])
    totals["value_mae"] = totals["value_abs_error"] / max(1, totals["value_count"])
    totals["win_positive_mean"] = totals["win_positive_sum"] / max(1, totals["win_positive_count"])
    totals["loss_positive_mean"] = totals["loss_positive_sum"] / max(1, totals["loss_positive_count"])
    return totals


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/processed/bc-v2-full"))
    parser.add_argument("--outcome-targets", type=Path, default=Path("reports/action_utility_targets_v2.json"))
    parser.add_argument("--init", type=Path, default=Path("models/bc-v2-full-fp32"))
    parser.add_argument("--output", type=Path, default=Path("models/research-candidate-q-v1"))
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--seed", type=int, default=20261002)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--fp32", action="store_true")
    parser.add_argument("--pair-weight", type=float, default=0.05)
    parser.add_argument("--value-weight", type=float, default=0.01)
    parser.add_argument("--max-pairs", type=int, default=16)
    parser.add_argument("--margin", type=float, default=0.10)
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.max_pairs < 1:
        raise ValueError("epochs, batch-size and max-pairs must be positive")
    if not (math.isfinite(args.lr) and args.lr > 0 and math.isfinite(args.pair_weight) and args.pair_weight >= 0
            and math.isfinite(args.value_weight) and args.value_weight >= 0 and math.isfinite(args.margin)
            and args.margin >= 0):
        raise ValueError("invalid optimizer/objective weight")
    if (args.output / "best.pt").exists():
        raise FileExistsError("choose a new output directory; existing checkpoint is preserved")
    data = args.data.resolve()
    outcome_path = args.outcome_targets.resolve()
    init_path = args.init.resolve()
    train_files, validation_files = shard_paths(data, "train"), shard_paths(data, "validation")
    manifest_path = data / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "complete" or manifest.get("feature_version") != FEATURE_VERSION:
        raise ValueError("prepared manifest is incomplete or has incompatible features")
    expected_hashes = {item["path"]: item["sha256"] for split in ("train", "validation")
                       for item in manifest["splits"][split]["shards"]}
    actual_hashes = {p.relative_to(data).as_posix(): sha256_file(p) for p in train_files + validation_files}
    if actual_hashes != expected_hashes:
        raise ValueError("prepared train/validation shard SHA256 differs from manifest")
    alignment = prepared_sample_alignment({"train": train_files, "validation": validation_files})
    outcome_targets, outcome_sha = load_outcome_targets(outcome_path, manifest_path, alignment)
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    torch.set_num_threads(4)
    device = torch.device(args.device)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable; refusing silent CPU fallback")
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cuda.matmul.allow_tf32 = False
    checkpoint_path = init_path / "best.pt" if init_path.is_dir() else init_path
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = CandidateModel(ModelConfig(**checkpoint.get("config", asdict(ModelConfig())))).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=.01)
    args.output.mkdir(parents=True, exist_ok=True)
    provenance = {"schema": SCHEMA, "seed": args.seed, "data_manifest_sha256": sha256_file(manifest_path),
                  "input_shards": actual_hashes, "outcome_targets": str(outcome_path),
                  "outcome_targets_sha256": outcome_sha, "init_checkpoint": str(checkpoint_path),
                  "init_checkpoint_sha256": sha256_file(checkpoint_path), "train_script_sha256": sha256_file(__file__),
                  "model_script_sha256": sha256_file(Path(__file__).with_name("model.py")),
                  "torch": str(torch.__version__), "device": str(device), "epochs_requested": args.epochs,
                  "batch_size": args.batch_size, "learning_rate": args.lr, "pair_weight": args.pair_weight,
                  "value_weight": args.value_weight, "max_pairs": args.max_pairs, "margin": args.margin,
                  "objective": "marginal BC + pair_weight*outcome-direction pairwise + value_weight*Huber",
                  "direction_definition": "+1 winning observed candidate > negatives; -1 losing observed candidate < negatives",
                  "test_used": False, "release_artifact": False}
    initial_bc = measure(model, validation_files, args.batch_size, device)
    initial_q = evaluate_q(model, validation_files, outcome_targets, args.batch_size, device, args.max_pairs, args.margin)
    report = {"status": "running", "provenance": provenance, "initial_validation_bc": initial_bc,
              "initial_validation_q": initial_q, "epochs": []}
    atomic_json(args.output / "training.json", report)
    print(json.dumps({"event": "initial", "bc": initial_bc, "q": initial_q}, ensure_ascii=False), flush=True)
    best = float("inf"); global_step = 0; started = time.monotonic()
    mixed = device.type == "cuda" and not args.fp32
    for epoch in range(args.epochs):
        model.train(); epoch_started = time.monotonic()
        sums = {"bc": 0.0, "pair": 0.0, "value": 0.0, "pairs": 0, "pair_rows": 0,
                "pair_correct": 0, "samples": 0}
        for batch in batches(train_files, args.batch_size, args.seed + epoch, True, include_ids=True):
            ids = batch.pop("ids")
            returns = torch.as_tensor([1.0 if outcome_targets[x] else -1.0 for x in ids],
                                      dtype=torch.float32, device=device)
            batch = {key: value.to(device) for key, value in batch.items()}
            positives = batch.pop("positives")
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16) if mixed else nullcontext():
                scores = model(**batch)
                bc_loss = marginal_loss(scores, positives)
                pair_loss, pairs, pair_rows, pair_correct = outcome_pairwise_loss(
                    scores, positives, batch["mask"], returns, args.max_pairs, args.margin)
                value_loss = positive_value_loss(scores, positives, returns)
                loss = bc_loss + args.pair_weight * pair_loss + args.value_weight * value_loss
            if not torch.isfinite(loss):
                raise RuntimeError("nonfinite candidate-Q loss")
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True); optimizer.step()
            sums["bc"] += float(bc_loss.detach()) * len(ids); sums["pair"] += float(pair_loss.detach()) * pairs
            sums["value"] += float(value_loss.detach()) * len(ids); sums["pairs"] += pairs
            sums["pair_rows"] += pair_rows; sums["pair_correct"] += pair_correct; sums["samples"] += len(ids)
            global_step += 1
        validation_bc = measure(model, validation_files, args.batch_size, device)
        validation_q = evaluate_q(model, validation_files, outcome_targets, args.batch_size, device, args.max_pairs, args.margin)
        # Selection remains BC validation NLL, so the research objective
        # cannot silently trade away the release contract's main diagnostic.
        row = {"epoch": epoch + 1, "train_samples": sums["samples"],
               "train_bc_nll": sums["bc"] / max(1, sums["samples"]),
               "train_pairwise": sums["pair"] / max(1, sums["pairs"]),
               "train_value": sums["value"] / max(1, sums["samples"]),
               "train_pair_count": sums["pairs"], "train_pair_rows": sums["pair_rows"],
               "train_pair_accuracy": sums["pair_correct"] / max(1, sums["pairs"]),
               "validation_bc": validation_bc, "validation_q": validation_q,
               "seconds": time.monotonic() - epoch_started, "steps": global_step}
        if validation_bc["nll"] < best:
            best = validation_bc["nll"]
            state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
            saved = {"state_dict": state, "config": asdict(model.config),
                     "provenance": dict(provenance, selected_epoch=epoch + 1,
                                         validation_bc=validation_bc, validation_q=validation_q,
                                         optimizer_steps=global_step)}
            temp = args.output / "best.pt.tmp"; torch.save(saved, temp); temp.replace(args.output / "best.pt")
            row["new_best"] = True; report["best_epoch"] = epoch + 1
        report["epochs"].append(row); report["seconds"] = time.monotonic() - started
        atomic_json(args.output / "training.json", report)
        print(json.dumps({"event": "epoch_done", **row}, ensure_ascii=False), flush=True)
    report["status"] = "completed"; report["checkpoint_sha256"] = sha256_file(args.output / "best.pt")
    report["peak_cuda_mebibytes"] = torch.cuda.max_memory_allocated() / 2**20 if device.type == "cuda" else None
    atomic_json(args.output / "training.json", report)
    print(json.dumps({"event": "completed", "best_epoch": report.get("best_epoch"),
                      "best_validation_nll": best, "seconds": report["seconds"],
                      "checkpoint_sha256": report["checkpoint_sha256"]}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
