"""Reproducible multi-positive behavioral cloning on audited NJUPT shards.

Train uses the train split only; checkpoint selection uses validation only.
The held-out test split is never opened by this program.
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


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def prepared_sample_alignment(files_by_split):
    """Return the exact prepared sample IDs and full candidate counts.

    Auxiliary target sidecars are allowed to be sparse (only rows with a
    positive/negative pair are useful), but every entry they do contain must
    refer to a prepared row with the same full candidate denominator.  This
    helper reads only ``ids`` and ``offsets`` from the selected train and
    validation shards; callers must not pass held-out test shards.
    """
    if not isinstance(files_by_split, dict) or set(files_by_split) != {"train", "validation"}:
        raise ValueError("prepared alignment requires train and validation shard sets")
    result = {}
    for split in ("train", "validation"):
        files = list(files_by_split[split])
        if not files:
            raise ValueError(f"prepared alignment has no {split} shards")
        mapping = {}
        for path in files:
            path = Path(path)
            try:
                with np.load(path, allow_pickle=False) as data:
                    if "ids" not in data.files or "offsets" not in data.files:
                        raise ValueError(f"{path}: prepared shard lacks ids/offsets")
                    ids, offsets = data["ids"], data["offsets"]
                    if len(offsets) != len(ids) + 1 or int(offsets[0]) != 0:
                        raise ValueError(f"{path}: malformed prepared offsets")
                    total = int(offsets[-1])
                    for index, value in enumerate(ids):
                        identity = str(value)
                        begin, end = int(offsets[index]), int(offsets[index + 1])
                        if not identity or identity in mapping or begin < 0 or end <= begin or end > total:
                            raise ValueError(f"{path}: malformed/duplicate prepared sample ID {identity!r}")
                        mapping[identity] = end - begin
            except OSError as exc:
                raise ValueError(f"cannot read prepared shard {path}: {exc}") from exc
        result[split] = mapping
    return result


def alignment_digest(mapping):
    """Stable digest of ``sample_id`` and candidate count pairs."""
    payload = "\n".join(f"{identity}\t{mapping[identity]}" for identity in sorted(mapping))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def marginal_loss(scores, positives, reduction="mean"):
    """Optimize total probability of every claim compatible with a demo."""
    values = torch.logsumexp(scores.float(), dim=-1) - torch.logsumexp(
        scores.float().masked_fill(~positives, float("-inf")), dim=-1)
    return values.mean() if reduction == "mean" else values


def multi_action_rows(actions, positives):
    """Return rows whose positive action contains at least two cards.

    ``action_features`` stores the physical action length in feature 120 as
    ``len(action) / 10``.  Looking only at positive candidates is important:
    a row can contain many multi-card *negative* candidates while the
    demonstrated action is a pass or a single.  All claim variants of one
    demonstrated face multiset have the same length, so ``any`` is safe here.
    """
    if actions.ndim != 3 or actions.shape[-1] <= 120:
        raise ValueError("invalid action feature tensor for multi-action weighting")
    if positives.shape != actions.shape[:2]:
        raise ValueError("positive labels do not match action candidates")
    return ((actions[..., 120] > 0.10000001) & positives).any(dim=-1)


def weighted_marginal_loss(scores, positives, actions, multi_weight, row_weights=None):
    """Compute a row-normalized weighted marginal loss.

    The validation objective remains the unweighted marginal NLL.  During
    training, rows whose positive action is multi-card receive
    ``multi_weight``; dividing by the sum of row weights keeps the effective
    learning-rate scale comparable to the baseline.  ``multi_weight=1`` is
    exactly the original objective.
    """
    losses = marginal_loss(scores, positives, "none")
    if row_weights is not None and row_weights.shape != losses.shape:
        raise ValueError("utility row weights do not match marginal-loss rows")
    if row_weights is not None and not torch.isfinite(row_weights).all():
        raise ValueError("utility row weights must be finite")
    if multi_weight == 1.0 and row_weights is None:
        return losses.mean(), losses, torch.zeros_like(losses, dtype=torch.bool)
    selected = multi_action_rows(actions, positives)
    weights = torch.ones_like(losses)
    if multi_weight != 1.0:
        weights = torch.where(selected, torch.as_tensor(multi_weight, device=losses.device, dtype=losses.dtype), weights)
    if row_weights is not None:
        weights = weights * row_weights.to(device=losses.device, dtype=losses.dtype)
    return (losses * weights).sum() / weights.sum(), losses, selected


def pass_mass_terms(scores, actions, positives):
    """Return pass-vs-play mass logits and eligible row labels.

    The scorer already masks padded candidates to ``-inf``.  Requiring finite
    scores here makes the helper robust to either padded batches or a caller
    that forgot to pass a candidate mask.  Leading rows normally have no
    pass/play competition and therefore drop out through ``eligible``.
    """
    if scores.ndim != 2 or actions.ndim != 3 or actions.shape[:2] != scores.shape:
        raise ValueError("scores and action candidates have incompatible shapes")
    if positives.shape != scores.shape:
        raise ValueError("positive labels do not match candidate scores")
    finite = torch.isfinite(scores)
    candidate_pass = actions[..., 126] > 0.5
    pass_mask = candidate_pass & finite
    play_mask = (~candidate_pass) & finite
    eligible = pass_mask.any(dim=-1) & play_mask.any(dim=-1)
    pass_scores = scores.float().masked_fill(~pass_mask, float("-inf"))
    play_scores = scores.float().masked_fill(~play_mask, float("-inf"))
    logits = torch.logsumexp(pass_scores, dim=-1) - torch.logsumexp(play_scores, dim=-1)
    labels = (pass_mask & positives).any(dim=-1).to(dtype=scores.dtype)
    return logits, eligible, labels


def pass_mass_loss(scores, actions, positives):
    """Auxiliary BCE for follow-row pass/play probability mass.

    ``p_logit = logsumexp(pass scores) - logsumexp(play scores)`` and the
    target is one iff a legal positive candidate is pass.  Rows without both
    pass and play candidates are excluded.  The third return value is an
    integer count of eligible positive-pass rows, useful for audit logging.
    """
    logits, eligible, labels = pass_mass_terms(scores, actions, positives)
    if not bool(eligible.any()):
        # Do not derive the zero from scores: padded candidate masks may make
        # the tensor sum -inf, and (-inf) * 0 would become NaN.
        return torch.zeros((), device=scores.device, dtype=torch.float32), eligible, 0
    loss = F.binary_cross_entropy_with_logits(logits[eligible].float(), labels[eligible].float())
    return loss, eligible, int(labels[eligible].sum().item())


def hard_pairwise_loss(scores, actions, positives, mask, max_pairs=32):
    """Rank each demonstrated action above same-kind, same-size negatives.

    Action features deliberately expose the semantic kind (108:120) and the
    physical action length (feature 120).  The positive face multiset can have
    several physical-card variants with identical features; using one positive
    representative avoids multiplying their otherwise identical pair losses.
    Candidate order is stable in prepared shards, so taking the first
    ``max_pairs`` hard negatives is deterministic and requires no test data.
    Rows without a same-kind/size negative contribute zero.
    """
    if scores.ndim != 2 or actions.ndim != 3 or mask.shape != scores.shape:
        raise ValueError("invalid hard-pairwise tensor shapes")
    if positives.shape != scores.shape:
        raise ValueError("positive labels do not match candidate scores")
    kinds = actions[..., 108:120].argmax(dim=-1)
    sizes = torch.round(actions[..., 120] * 10).to(dtype=torch.long)
    total = scores.new_zeros((), dtype=torch.float32)
    pair_count = 0
    row_count = 0
    correct = 0
    for row in range(scores.shape[0]):
        valid = mask[row]
        positive = torch.nonzero(valid & positives[row], as_tuple=False).flatten()
        if positive.numel() == 0:
            continue
        # Face-multiset claim variants have identical action features; one is
        # enough and avoids overweighting deck-copy ambiguity.
        pi = positive[0]
        negative = torch.nonzero(
            valid & ~positives[row]
            & (kinds[row] == kinds[row, pi])
            & (sizes[row] == sizes[row, pi]), as_tuple=False).flatten()
        if negative.numel() == 0:
            continue
        negative = negative[:max_pairs]
        total = total + torch.nn.functional.softplus(scores[row, negative] - scores[row, pi]).float().sum()
        pair_count += int(negative.numel())
        row_count += 1
        correct += int((scores[row, pi] > scores[row, negative]).sum())
    if pair_count == 0:
        return torch.zeros((), device=scores.device, dtype=torch.float32), 0, 0, 0
    return total / pair_count, pair_count, row_count, correct


def continuation_pairwise_loss(scores, labels, mask, max_pairs=32):
    """Rank continuation-compatible candidates above incompatible candidates.

    ``labels`` is a development-only sidecar aligned to the exact full
    candidate order: +1 means the observed public continuation remains
    compatible, -1 means the candidate is legal but blocks that continuation,
    and 0 means unsupported/ignored.  This is not a Q target.  Pairing is
    deterministic and capped so sparse sidecars cannot dominate the BC loss.
    """
    if scores.ndim != 2 or labels.shape != scores.shape or mask.shape != scores.shape:
        raise ValueError("invalid continuation pairwise tensor shapes")
    if max_pairs < 1:
        raise ValueError("max_pairs must be positive")
    if not torch.isfinite(scores.masked_fill(~mask, 0.)).all():
        raise ValueError("continuation pairwise scores contain nonfinite valid values")
    total = scores.new_zeros((), dtype=torch.float32)
    pair_count = row_count = correct = 0
    for row in range(scores.shape[0]):
        valid = mask[row]
        positive = torch.nonzero(valid & (labels[row] > 0), as_tuple=False).flatten()
        negative = torch.nonzero(valid & (labels[row] < 0), as_tuple=False).flatten()
        if positive.numel() == 0 or negative.numel() == 0:
            continue
        pairs = [(int(pi), int(ni)) for pi in positive.tolist() for ni in negative.tolist()]
        pairs = pairs[:max_pairs]
        if not pairs:
            continue
        pi = torch.as_tensor([item[0] for item in pairs], device=scores.device)
        ni = torch.as_tensor([item[1] for item in pairs], device=scores.device)
        total = total + torch.nn.functional.softplus(scores[row, ni] - scores[row, pi]).float().sum()
        pair_count += len(pairs)
        row_count += 1
        correct += int((scores[row, pi] > scores[row, ni]).sum())
    if pair_count == 0:
        return torch.zeros((), device=scores.device, dtype=torch.float32), 0, 0, 0
    return total / pair_count, pair_count, row_count, correct


def load_continuation_targets(path, data_manifest, source_manifest, expected_alignment=None):
    """Load and validate a non-release continuation sidecar.

    The loader fail-closes on audit-only files, test provenance, mismatched
    prepared/source manifests, malformed labels, or duplicate IDs.  It does
    not open the held-out test split.
    """
    path = Path(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != "oxbot-candidate-continuation-targets-v1":
        raise ValueError("continuation target schema is incompatible")
    if payload.get("test_used") is not False or payload.get("splits_read") != ["train", "validation"]:
        raise ValueError("continuation targets have invalid split provenance")
    if payload.get("audit_only") or payload.get("trainable") is not True:
        raise ValueError("continuation target is audit-only or not trainable")
    if payload.get("prepared_manifest_sha256") != sha256_file(data_manifest):
        raise ValueError("continuation target prepared manifest does not match data")
    if payload.get("source_manifest_sha256") != source_manifest.get("source_manifest_sha256"):
        raise ValueError("continuation target source manifest does not match data")
    expected_source_splits = {split: source_manifest.get("source_splits_sha256", {}).get(split)
                              for split in ("train", "validation")}
    if payload.get("source_split_sha256") != expected_source_splits:
        raise ValueError("continuation target source split SHA256 does not match data")
    split_targets = payload.get("targets")
    if not isinstance(split_targets, dict):
        raise ValueError("continuation target map is missing")
    if expected_alignment is not None:
        if payload.get("target_coverage") != "paired_rows_only":
            raise ValueError("continuation target coverage policy is missing or incompatible")
        if not isinstance(expected_alignment, dict) or set(expected_alignment) != {"train", "validation"}:
            raise ValueError("continuation expected prepared alignment is malformed")
        payload_alignment = payload.get("prepared_alignment")
        if not isinstance(payload_alignment, dict):
            raise ValueError("continuation target prepared alignment metadata is missing")
        for split in ("train", "validation"):
            expected = expected_alignment[split]
            if not isinstance(expected, dict):
                raise ValueError(f"continuation expected prepared alignment is malformed: {split}")
            metadata = payload_alignment.get(split)
            if not isinstance(metadata, dict) or metadata.get("record_count") != len(expected):
                raise ValueError(f"continuation prepared alignment record count mismatch: {split}")
            if metadata.get("sha256") != alignment_digest(expected):
                raise ValueError(f"continuation prepared alignment digest mismatch: {split}")
    result = {}
    for split in ("train", "validation"):
        values = split_targets.get(split)
        if not isinstance(values, dict):
            raise ValueError(f"continuation target split is missing: {split}")
        clean = {}
        expected = expected_alignment.get(split) if expected_alignment is not None else None
        for identity, target in values.items():
            if not isinstance(identity, str) or not isinstance(target, dict):
                raise ValueError("malformed continuation target entry")
            if expected is not None and identity not in expected:
                raise ValueError(f"continuation target ID is not in prepared {split} shards: {identity}")
            count = target.get("candidate_count")
            labels = target.get("labels")
            if type(count) is not int or count < 1 or not isinstance(labels, list) or len(labels) != count:
                raise ValueError(f"malformed continuation labels for {identity}")
            if expected is not None and count != expected[identity]:
                raise ValueError(f"continuation candidate count mismatch for {identity}: {count} != {expected[identity]}")
            if any(type(label) is not int or label not in (-1, 0, 1) for label in labels):
                raise ValueError(f"invalid continuation labels for {identity}")
            if not any(label > 0 for label in labels) or not any(label < 0 for label in labels):
                raise ValueError(f"continuation target has no positive/negative pair for {identity}")
            clean[identity] = {"candidate_count": count, "labels": labels}
        result[split] = clean
    return result, sha256_file(path)


def shard_paths(root, split):
    # Preparation has one manifest plus files inside per-split directories.
    files = sorted((root / split).glob("*.npz"))
    if not files:
        files = sorted(root.glob(split + "-*.npz"))
    if not files:
        files = sorted(root.glob(split + "_*.npz"))
    if not files:
        raise ValueError(f"no {split} shards found under {root}")
    return files


def batches(files, batch_size, seed, training, include_ids=False):
    rng = np.random.default_rng(seed)
    files = list(files)
    if training:
        rng.shuffle(files)
    for path in files:
        with np.load(path, allow_pickle=False) as data:
            required = {"state", "tokens", "lengths", "actions", "offsets", "positives"}
            if include_ids:
                required.add("ids")
            missing = required - set(data.files)
            if missing:
                raise ValueError(f"{path}: missing keys {sorted(missing)}; keys={data.files}")
            state, tokens, lengths = data["state"], data["tokens"], data["lengths"]
            actions, offsets, positives = data["actions"], data["offsets"], data["positives"]
            ids = data["ids"] if include_ids else None
            if state.shape != (len(lengths), 128) or len(offsets) != len(lengths) + 1:
                raise ValueError(f"invalid shape in {path}")
            if offsets[0] != 0 or offsets[-1] != len(actions) or len(positives) != len(actions):
                raise ValueError(f"invalid candidate offsets in {path}")
            indices = np.arange(len(lengths))
            if training:
                rng.shuffle(indices)
            pending = []
            max_candidates = 0
            for index in indices:
                count = int(offsets[index + 1] - offsets[index])
                if count < 1 or not positives[offsets[index]:offsets[index+1]].any():
                    raise ValueError(f"empty candidate or positive set in {path}")
                # Full validation retains every legal candidate. Bound padded
                # batches so an unusual large hand cannot exhaust GPU memory.
                next_max = max(max_candidates, count)
                if pending and (len(pending) >= batch_size or next_max * (len(pending) + 1) > 16384):
                    yield collate(pending, state, tokens, lengths, actions, offsets, positives, ids)
                    pending, max_candidates = [], 0
                pending.append(int(index))
                max_candidates = max(max_candidates, count)
            if pending:
                yield collate(pending, state, tokens, lengths, actions, offsets, positives, ids)


def collate(indices, state, tokens, lengths, actions, offsets, positives, ids=None):
    selected_lengths = lengths[indices].astype(np.int64)
    max_length = int(selected_lengths.max())
    counts = [int(offsets[i+1] - offsets[i]) for i in indices]
    shape = (len(indices), max(counts))
    candidates = np.zeros((*shape, 128), dtype=np.float32)
    positive = np.zeros(shape, dtype=np.bool_)
    mask = np.zeros(shape, dtype=np.bool_)
    for row, (index, count) in enumerate(zip(indices, counts, strict=True)):
        begin, end = int(offsets[index]), int(offsets[index+1])
        candidates[row, :count] = actions[begin:end]
        positive[row, :count] = positives[begin:end]
        mask[row, :count] = True
    result = {"tokens": torch.from_numpy(tokens[indices, :max_length].astype(np.int64)),
            "lengths": torch.from_numpy(selected_lengths),
            "state": torch.from_numpy(state[indices].astype(np.float32)),
            "actions": torch.from_numpy(candidates), "mask": torch.from_numpy(mask),
            "positives": torch.from_numpy(positive)}
    if ids is not None:
        # IDs are provenance strings, not model inputs. Keep them out of the
        # tensor move path so an auxiliary target can be looked up per row.
        result["ids"] = [str(ids[index]) for index in indices]
    return result


def measure(model, files, batch_size, device):
    totals = {"samples": 0, "correct": 0, "nll_sum": 0., "choice_samples": 0, "choice_correct": 0,
              "pass_samples": 0, "pass_correct": 0, "play_samples": 0, "play_correct": 0,
              "pass_mass_samples": 0, "pass_mass_bce_sum": 0., "pass_mass_correct": 0,
              "pass_mass_positive_rows": 0, "rank_pair_samples": 0,
              "rank_pair_correct": 0, "rank_pair_rows": 0}
    model.eval()
    with torch.inference_mode():
        for batch in batches(files, batch_size, 0, False):
            batch = {key: value.to(device) for key, value in batch.items()}
            positives = batch.pop("positives")
            scores = model(**batch)
            losses = marginal_loss(scores, positives, "none")
            _, rank_pairs, rank_rows, rank_correct = hard_pairwise_loss(
                scores, batch["actions"], positives, batch["mask"])
            totals["rank_pair_samples"] += rank_pairs
            totals["rank_pair_correct"] += rank_correct
            totals["rank_pair_rows"] += rank_rows
            pass_mass, eligible, positive_rows = pass_mass_loss(scores, batch["actions"], positives)
            if bool(eligible.any()):
                logits, _, labels = pass_mass_terms(scores, batch["actions"], positives)
                totals["pass_mass_samples"] += int(eligible.sum())
                totals["pass_mass_bce_sum"] += float(pass_mass.detach()) * int(eligible.sum())
                totals["pass_mass_correct"] += int(((logits[eligible] > 0) == labels[eligible].bool()).sum())
                totals["pass_mass_positive_rows"] += positive_rows
            winners = scores.argmax(-1)
            correct = positives.gather(1, winners[:, None]).squeeze(1)
            choices = batch["mask"].sum(-1) > 1
            passes = (batch["actions"][:, :, 126].bool() & positives).any(-1)
            totals["samples"] += len(losses)
            totals["correct"] += int(correct.sum())
            totals["nll_sum"] += float(losses.sum())
            for name, selected in (("choice", choices), ("pass", passes), ("play", ~passes)):
                totals[name+"_samples"] += int(selected.sum())
                totals[name+"_correct"] += int((selected & correct).sum())
    totals["nll"] = totals["nll_sum"] / max(1, totals["samples"])
    totals["accuracy"] = totals["correct"] / max(1, totals["samples"])
    for name in ("choice", "pass", "play"):
        totals[name+"_accuracy"] = totals[name+"_correct"] / max(1, totals[name+"_samples"])
    totals["pass_mass_bce"] = totals["pass_mass_bce_sum"] / max(1, totals["pass_mass_samples"])
    totals["pass_mass_accuracy"] = totals["pass_mass_correct"] / max(1, totals["pass_mass_samples"])
    totals["pass_mass_positive_rate"] = totals["pass_mass_positive_rows"] / max(1, totals["pass_mass_samples"])
    totals["rank_pair_accuracy"] = totals["rank_pair_correct"] / max(1, totals["rank_pair_samples"])
    return totals


def atomic_json(path, data):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temp.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data/processed/bc-v1"))
    parser.add_argument("--output", type=Path, default=Path("models/bc-v1"))
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=20261001)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--fp32", action="store_true", help="disable BF16 mixed precision")
    parser.add_argument("--multi-weight", type=float, default=1.0,
                        help="row weight for positive actions with at least two cards (default: 1)")
    parser.add_argument("--pass-mass-beta", type=float, default=0.0,
                        help="follow-row pass-vs-play mass BCE weight (default: 0)")
    parser.add_argument("--rank-weight", type=float, default=0.0,
                        help="same-kind/same-size hard-negative pairwise weight (default: 0)")
    parser.add_argument("--utility-targets", type=Path, default=None,
                        help="JSON target map from build_utility_targets.py (train/validation only)")
    parser.add_argument("--utility-alpha", type=float, default=0.0,
                        help="advantage-weighted BC strength: winning rows 1+a, losing rows 1-a (default: 0)")
    parser.add_argument("--continuation-targets", type=Path, default=None,
                        help="development-only public-continuation sidecar (train/validation only)")
    parser.add_argument("--continuation-lambda", type=float, default=0.0,
                        help="pairwise continuation compatibility loss weight (default: 0)")
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1:
        raise ValueError("epochs and batch size must be positive")
    if not math.isfinite(args.multi_weight) or args.multi_weight < 1.0:
        raise ValueError("multi-weight must be finite and at least 1")
    if not math.isfinite(args.pass_mass_beta) or args.pass_mass_beta < 0.0:
        raise ValueError("pass-mass-beta must be finite and nonnegative")
    if not math.isfinite(args.rank_weight) or args.rank_weight < 0.0:
        raise ValueError("rank-weight must be finite and nonnegative")
    if not math.isfinite(args.utility_alpha) or not 0.0 <= args.utility_alpha < 1.0:
        raise ValueError("utility-alpha must be finite and in [0, 1)")
    if args.utility_alpha and args.utility_targets is None:
        raise ValueError("utility-targets is required when utility-alpha is nonzero")
    if not math.isfinite(args.continuation_lambda) or not 0.0 <= args.continuation_lambda <= 0.2:
        raise ValueError("continuation-lambda must be finite and in [0, 0.2]")
    if args.continuation_lambda and args.continuation_targets is None:
        raise ValueError("continuation-targets is required when continuation-lambda is nonzero")
    if (args.output / "best.pt").exists():
        raise FileExistsError("choose a new output directory; existing checkpoint is preserved")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(4)
    device = torch.device(args.device)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available; do not silently train on CPU")
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cuda.matmul.allow_tf32 = False
    train_files = shard_paths(args.data, "train")
    validation_files = shard_paths(args.data, "validation")
    # Verify selected inputs against the prepared manifest through their hashes
    # and persist the exact set. Test files deliberately remain untouched.
    data_manifest = args.data / "manifest.json"
    if not data_manifest.is_file():
        raise ValueError("prepared data manifest is required")
    source_manifest = json.loads(data_manifest.read_text(encoding="utf-8"))
    if source_manifest.get("status") != "complete" or source_manifest.get("feature_version") != FEATURE_VERSION:
        raise ValueError("prepared manifest is incomplete or has incompatible features")
    expected_hashes = {item["path"]: item["sha256"]
                       for split in ("train", "validation")
                       for item in source_manifest["splits"][split]["shards"]}
    input_hashes = {path.relative_to(args.data).as_posix(): sha256_file(path) for path in train_files + validation_files}
    if input_hashes != expected_hashes:
        raise ValueError("prepared shard set or SHA256 does not match the manifest")
    expected_alignment = prepared_sample_alignment({"train": train_files, "validation": validation_files})
    utility_targets = None
    utility_targets_sha256 = None
    utility_payload = None
    if args.utility_targets is not None:
        if not args.utility_targets.is_file():
            raise ValueError(f"utility target file does not exist: {args.utility_targets}")
        utility_payload = json.loads(args.utility_targets.read_text(encoding="utf-8"))
        if utility_payload.get("schema") != "oxbot-action-utility-targets-v1" or utility_payload.get("test_used"):
            raise ValueError("utility target file has incompatible schema or test provenance")
        if utility_payload.get("splits_read") != ["train", "validation"]:
            raise ValueError("utility target file must be built from train and validation only")
        if utility_payload.get("source_manifest_sha256") != source_manifest.get("source_manifest_sha256"):
            raise ValueError("utility target source manifest does not match prepared data provenance")
        expected_source_splits = {
            split: source_manifest.get("source_splits_sha256", {}).get(split)
            for split in ("train", "validation")
        }
        if utility_payload.get("source_split_sha256") != expected_source_splits:
            raise ValueError("utility target source split SHA256 does not match prepared data provenance")
        utility_targets = utility_payload.get("targets")
        if not isinstance(utility_targets, dict) or not utility_targets:
            raise ValueError("utility target map is empty")
        if any(value not in (0, 1) for value in utility_targets.values()):
            raise ValueError("utility targets must be binary team-win indicators")
        utility_targets_sha256 = sha256_file(args.utility_targets)
    continuation_targets = None
    continuation_targets_sha256 = None
    if args.continuation_targets is not None:
        if not args.continuation_targets.is_file():
            raise ValueError(f"continuation target file does not exist: {args.continuation_targets}")
        continuation_targets, continuation_targets_sha256 = load_continuation_targets(
            args.continuation_targets, data_manifest, source_manifest, expected_alignment)
    provenance = {"purpose": "njupt_behavioral_cloning", "seed": args.seed,
                  "data_manifest_sha256": sha256_file(data_manifest), "input_shards": input_hashes,
                  "source_manifest_schema": source_manifest.get("schema"),
                  "train_script_sha256": sha256_file(__file__), "model_script_sha256": sha256_file(Path(__file__).with_name("model.py")),
                  "torch": str(torch.__version__), "device": str(device), "epochs_requested": args.epochs,
                  "batch_size": args.batch_size, "learning_rate": args.lr,
                  "multi_weight": args.multi_weight,
                  "multi_weight_definition": "positive action feature length > 1 card",
                  "pass_mass_beta": args.pass_mass_beta,
                  "pass_mass_definition": "BCE(logsumexp(pass)-logsumexp(play)) on rows with finite pass and play candidates",
                  "rank_weight": args.rank_weight,
                  "rank_definition": "softplus(score_same_kind_same_size_negative-score_positive), max 32 deterministic negatives per row",
                  "utility_targets": str(args.utility_targets) if args.utility_targets else None,
                  "utility_targets_sha256": utility_targets_sha256,
                  "utility_source_manifest_sha256": utility_payload.get("source_manifest_sha256") if utility_payload else None,
                  "utility_source_split_sha256": utility_payload.get("source_split_sha256") if utility_payload else None,
                  "utility_alpha": args.utility_alpha,
                  "utility_definition": "row weight 1+alpha for eventual winning team and 1-alpha for losing team; target is not a model feature",
                  "continuation_targets": str(args.continuation_targets) if args.continuation_targets else None,
                  "continuation_targets_sha256": continuation_targets_sha256,
                  "continuation_lambda": args.continuation_lambda,
                  "continuation_definition": "public continuation compatibility pairwise loss; not a Q value or team outcome",
                  "loss": "negative_log_probability_of_all_compatible_claims plus optional hard-negative pairwise ranking",
                  "test_used": False}
    model = CandidateModel().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=.01)
    args.output.mkdir(parents=True, exist_ok=True)
    initial = measure(model, validation_files, args.batch_size, device)
    report = {"status": "running", "provenance": provenance, "initial_validation": initial, "epochs": []}
    atomic_json(args.output / "training.json", report)
    print(json.dumps({"event": "initial_validation", **initial}), flush=True)
    started, best, global_step = time.monotonic(), float("inf"), 0
    mixed = device.type == "cuda" and not args.fp32
    for epoch in range(args.epochs):
        model.train()
        loss_sum, weighted_loss_sum, weight_sum, multi_rows, samples = 0., 0., 0., 0, 0
        pass_mass_sum, pass_mass_samples, pass_mass_positive_rows, pass_mass_correct = 0., 0, 0, 0
        rank_sum, rank_pairs, rank_rows = 0., 0, 0
        continuation_sum, continuation_pairs, continuation_rows, continuation_correct = 0., 0, 0, 0
        epoch_started = time.monotonic()
        for batch_index, batch in enumerate(batches(
                train_files, args.batch_size, args.seed + epoch, True,
                include_ids=bool(args.utility_alpha or args.continuation_lambda))):
            row_weights = None
            sample_ids = batch.pop("ids", None)
            if args.utility_alpha:
                if sample_ids is None:
                    raise ValueError("utility training requires sample IDs")
                try:
                    labels = [utility_targets[sample_id] for sample_id in sample_ids]
                except KeyError as error:
                    raise ValueError(f"utility target missing prepared sample ID: {error.args[0]}") from error
                row_weights = torch.as_tensor(
                    [1.0 + args.utility_alpha if label else 1.0 - args.utility_alpha for label in labels],
                    dtype=torch.float32, device=device)
            continuation_labels = None
            if args.continuation_lambda:
                if sample_ids is None:
                    raise ValueError("continuation training requires sample IDs")
                counts = [int(value) for value in batch["mask"].sum(-1).tolist()]
                max_count = int(batch["mask"].shape[1])
                label_array = np.zeros((len(sample_ids), max_count), dtype=np.int8)
                for row, (sample_id, count) in enumerate(zip(sample_ids, counts, strict=True)):
                    target = continuation_targets["train"].get(sample_id)
                    if target is None:
                        continue
                    if target["candidate_count"] != count:
                        raise ValueError(f"continuation candidate count mismatch for {sample_id}")
                    label_array[row, :count] = np.asarray(target["labels"], dtype=np.int8)
                continuation_labels = torch.from_numpy(label_array)
            batch = {key: value.to(device) for key, value in batch.items()}
            if continuation_labels is not None:
                continuation_labels = continuation_labels.to(device)
            positives = batch.pop("positives")
            optimizer.zero_grad(set_to_none=True)
            warmup = min(1., (global_step + 1) / 100)
            decay = .15 + .85 * .5 * (1 + math.cos(math.pi * epoch / args.epochs))
            for group in optimizer.param_groups:
                group["lr"] = args.lr * warmup * decay
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16) if mixed else nullcontext():
                scores = model(**batch)
                loss, row_losses, selected_multi = weighted_marginal_loss(
                    scores, positives, batch["actions"], args.multi_weight, row_weights)
                if args.rank_weight:
                    rank_aux, rank_pair_count, rank_row_count, _ = hard_pairwise_loss(
                        scores, batch["actions"], positives, batch["mask"])
                    loss = loss + args.rank_weight * rank_aux
                else:
                    rank_aux, rank_pair_count, rank_row_count = scores.new_zeros(()), 0, 0
                pass_aux, pass_eligible, pass_positive_count = pass_mass_loss(
                    scores, batch["actions"], positives)
                if args.pass_mass_beta:
                    loss = loss + args.pass_mass_beta * pass_aux
                if bool(pass_eligible.any()):
                    pass_logits, _, pass_labels = pass_mass_terms(scores, batch["actions"], positives)
                    eligible_count = int(pass_eligible.sum())
                    pass_mass_sum += float(pass_aux.detach()) * eligible_count
                    pass_mass_samples += eligible_count
                    pass_mass_positive_rows += pass_positive_count
                    pass_mass_correct += int(((pass_logits[pass_eligible] > 0) == pass_labels[pass_eligible].bool()).sum())
                if args.rank_weight:
                    rank_sum += float(rank_aux.detach()) * rank_pair_count
                    rank_pairs += rank_pair_count
                    rank_rows += rank_row_count
                if args.continuation_lambda:
                    continuation_aux, continuation_pair_count, continuation_row_count, continuation_pair_correct = continuation_pairwise_loss(
                        scores, continuation_labels, batch["mask"])
                    loss = loss + args.continuation_lambda * continuation_aux
                    continuation_sum += float(continuation_aux.detach()) * continuation_pair_count
                    continuation_pairs += continuation_pair_count
                    continuation_rows += continuation_row_count
                    continuation_correct += continuation_pair_correct
            if not torch.isfinite(loss):
                raise RuntimeError("nonfinite BC loss")
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
            count = len(positives)
            loss_sum += float(row_losses.detach().sum())
            weighted_loss_sum += float(loss.detach()) * count
            effective_weights = torch.where(selected_multi,
                                            torch.as_tensor(args.multi_weight, device=selected_multi.device),
                                            torch.ones_like(selected_multi, dtype=torch.float32))
            if row_weights is not None:
                effective_weights = effective_weights * row_weights
            weight_sum += float(effective_weights.sum())
            multi_rows += int(selected_multi.sum())
            samples += count
            global_step += 1
            if batch_index % 100 == 0:
                print(json.dumps({"event": "train", "epoch": epoch+1, "batch": batch_index,
                                  "samples": samples, "mean_nll": loss_sum/samples,
                                  "weighted_nll": weighted_loss_sum/max(1, samples),
                                  "multi_rows": multi_rows,
                                  "pass_mass_bce": pass_mass_sum/max(1, pass_mass_samples),
                                  "pass_mass_samples": pass_mass_samples,
                                  "pass_mass_accuracy": pass_mass_correct/max(1, pass_mass_samples),
                                  "rank_pairwise": rank_sum/max(1, rank_pairs),
                                  "rank_pairs": rank_pairs,
                                  "rank_rows": rank_rows,
                                  "continuation_pairwise": continuation_sum/max(1, continuation_pairs),
                                  "continuation_pairs": continuation_pairs,
                                  "continuation_rows": continuation_rows,
                                  "continuation_accuracy": continuation_correct/max(1, continuation_pairs),
                                  "grad_norm": float(grad_norm), "seconds": time.monotonic()-epoch_started}), flush=True)
        validation = measure(model, validation_files, args.batch_size, device)
        row = {"epoch": epoch+1, "train_samples": samples, "train_nll": loss_sum/max(1,samples),
               "train_weighted_nll": weighted_loss_sum/max(1, samples),
               "train_weight_sum": weight_sum, "train_multi_rows": multi_rows,
               "train_multi_fraction": multi_rows/max(1, samples),
               "train_pass_mass_bce": pass_mass_sum/max(1, pass_mass_samples),
               "train_pass_mass_samples": pass_mass_samples,
               "train_pass_mass_positive_rows": pass_mass_positive_rows,
               "train_pass_mass_accuracy": pass_mass_correct/max(1, pass_mass_samples),
               "train_rank_pairwise": rank_sum/max(1, rank_pairs),
               "train_rank_pairs": rank_pairs,
               "train_rank_rows": rank_rows,
               "train_continuation_pairwise": continuation_sum/max(1, continuation_pairs),
               "train_continuation_pairs": continuation_pairs,
               "train_continuation_rows": continuation_rows,
               "train_continuation_accuracy": continuation_correct/max(1, continuation_pairs),
               "validation": validation, "seconds": time.monotonic()-epoch_started, "steps": global_step}
        if validation["nll"] < best:
            best = validation["nll"]
            checkpoint = {"state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
                          "config": asdict(ModelConfig()), "provenance": dict(provenance, selected_epoch=epoch+1,
                                                                                 validation=validation, optimizer_steps=global_step)}
            temp = args.output / "best.pt.tmp"
            torch.save(checkpoint, temp)
            temp.replace(args.output / "best.pt")
            report["best_epoch"] = epoch+1
            row["new_best"] = True
        report["epochs"].append(row)
        report["seconds"] = time.monotonic()-started
        atomic_json(args.output / "training.json", report)
        print(json.dumps({"event": "epoch_done", **row}), flush=True)
    report["status"] = "completed"
    report["checkpoint_sha256"] = sha256_file(args.output / "best.pt")
    report["peak_cuda_mebibytes"] = torch.cuda.max_memory_allocated() / 2**20 if device.type == "cuda" else None
    atomic_json(args.output / "training.json", report)
    print(json.dumps({"event": "completed", "best_epoch": report["best_epoch"], "best_validation_nll": best,
                      "seconds": report["seconds"], "checkpoint_sha256": report["checkpoint_sha256"]}), flush=True)


if __name__ == "__main__":
    main()
