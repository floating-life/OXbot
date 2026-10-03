"""Validation-only diagnosis of strategic candidate grouping.

The v1 checkpoint is a scorer over concrete legal candidates.  This report
keeps that model and its feature contract unchanged, but asks what happens if
concrete candidates are marginalized into a strategic group.  A group keeps
kind/key/secondary, the actual rank multiset and wildcard count; straight
flush groups additionally keep the concrete face/suit vector.  The report is
for diagnosis and frozen-policy experiments only: it never reads the test
split and never changes weights or training artifacts.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "train")]
from features import FEATURE_VERSION, KINDS
from model import CandidateModel, ModelConfig


HISTORY_FEATURES = 128
ACTION_FEATURES = 128
RANK_NAMES = tuple("A234567890JQK") + ("small_joker", "big_joker")
GROUP_KINDS = tuple(KINDS)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def logsumexp(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    largest = float(np.max(values))
    return largest + math.log(float(np.exp(values - largest).sum()))


def softmax(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    largest = np.max(values)
    result = np.exp(values - largest)
    return result / result.sum()


def group_descriptor(action: np.ndarray, kind: str, key: int, secondary: int,
                     level: int) -> tuple:
    """Return a hashable strategic key and serializable descriptor fields."""
    # action_features stores half-counts by canonical face (two decks).  A
    # physical rank multiset is therefore recovered by multiplying by two.
    face_counts = np.rint(np.asarray(action[:54], dtype=np.float64) * 2).astype(np.int64)
    if np.any(face_counts < 0) or np.any(face_counts > 2):
        raise ValueError("action face count is outside the two-deck contract")
    if not np.allclose(action[:54] * 2, face_counts, atol=1e-5):
        raise ValueError("action face count is not an integral half-count")
    rank_counts = [int(face_counts[4 * rank:4 * rank + 4].sum()) for rank in range(13)]
    rank_counts.extend((int(face_counts[52]), int(face_counts[53])))
    wild_count = int(face_counts[4 * level]) if 0 <= level < 13 else 0
    # For ordinary actions suits are intentionally marginalized.  A straight
    # flush's suit pattern is part of the action's strategic identity; keeping
    # the complete face vector is conservative and preserves all required suit
    # information, including a possible heart wildcard.
    suit_signature = tuple(int(value) for value in face_counts) if kind == "straight_flush" else None
    group_key = (kind, int(round(action[120] * 10)), int(key), int(secondary),
                 tuple(rank_counts), wild_count, suit_signature)
    return group_key, {
        "kind": kind,
        "length": int(round(action[120] * 10)),
        "key": int(key),
        "secondary": int(secondary),
        "rank_counts": rank_counts,
        "actual_ranks": [RANK_NAMES[index] for index, count in enumerate(rank_counts) for _ in range(count)],
        "wild_count": wild_count,
        "suit_signature": list(suit_signature) if suit_signature is not None else None,
    }


def action_metadata(actions: np.ndarray, state: np.ndarray) -> tuple[list[tuple], list[dict]]:
    level = int(np.argmax(state[112:125]))
    kinds = np.argmax(actions[:, 108:120], axis=1)
    keys = np.rint(actions[:, 121] * 14).astype(np.int64)
    secondaries = np.rint(actions[:, 122] * 14).astype(np.int64)
    keys_out, descriptors = [], []
    for action, kind_index, key, secondary in zip(actions, kinds, keys, secondaries, strict=True):
        kind = GROUP_KINDS[int(kind_index)]
        descriptor_key, descriptor = group_descriptor(action, kind, int(key), int(secondary), level)
        keys_out.append(descriptor_key)
        descriptors.append(descriptor)
    return keys_out, descriptors


def collated_batches(state: np.ndarray, tokens: np.ndarray, lengths: np.ndarray,
                     actions: np.ndarray, offsets: np.ndarray, positives: np.ndarray,
                     batch_size: int, cell_limit: int = 16384):
    """Yield validation rows in bounded padded batches, preserving row order."""
    pending: list[int] = []
    max_count = 0
    for row in range(len(lengths)):
        count = int(offsets[row + 1] - offsets[row])
        if not count or not positives[offsets[row]:offsets[row + 1]].any():
            raise ValueError("validation row has no candidate or positive label")
        if pending and (len(pending) >= batch_size or max(max_count, count) * (len(pending) + 1) > cell_limit):
            yield _collate(pending, state, tokens, lengths, actions, offsets, positives)
            pending, max_count = [], 0
        pending.append(row)
        max_count = max(max_count, count)
    if pending:
        yield _collate(pending, state, tokens, lengths, actions, offsets, positives)


def _collate(rows, state, tokens, lengths, actions, offsets, positives):
    max_length = int(np.max(lengths[rows]))
    counts = [int(offsets[row + 1] - offsets[row]) for row in rows]
    width = max(counts)
    batch_actions = np.zeros((len(rows), width, ACTION_FEATURES), dtype=np.float32)
    batch_positive = np.zeros((len(rows), width), dtype=np.bool_)
    mask = np.zeros((len(rows), width), dtype=np.bool_)
    for batch_row, row in enumerate(rows):
        begin, end = int(offsets[row]), int(offsets[row + 1])
        batch_actions[batch_row, :end - begin] = actions[begin:end]
        batch_positive[batch_row, :end - begin] = positives[begin:end]
        mask[batch_row, :end - begin] = True
    return (rows, {"tokens": torch.from_numpy(tokens[rows, :max_length].astype(np.int64)),
                   "lengths": torch.from_numpy(lengths[rows].astype(np.int64)),
                   "state": torch.from_numpy(state[rows].astype(np.float32)),
                   "actions": torch.from_numpy(batch_actions),
                   "mask": torch.from_numpy(mask),
                   "positives": torch.from_numpy(batch_positive)})


def summarize_distribution(counter: Counter) -> dict[str, int]:
    return {str(key): int(value) for key, value in sorted(counter.items())}


def run(args) -> dict:
    data = args.data.resolve()
    checkpoint_path = args.checkpoint.resolve()
    checkpoint_sha = sha256_file(checkpoint_path)
    if args.expected_checkpoint_sha and checkpoint_sha != args.expected_checkpoint_sha:
        raise ValueError("checkpoint does not match the expected frozen SHA256")
    manifest_path = data / "manifest.json"
    manifest_sha = sha256_file(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "complete" or manifest.get("feature_version") != FEATURE_VERSION:
        raise ValueError("prepared manifest is incomplete or feature-incompatible")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if checkpoint.get("config") != asdict(ModelConfig()):
        raise ValueError("checkpoint architecture differs from the fixed v1 model")
    if checkpoint.get("provenance", {}).get("data_manifest_sha256") != manifest_sha:
        raise ValueError("checkpoint and validation data use different manifests")
    descriptors = manifest["splits"]["validation"]["shards"]
    expected = {item["path"]: item["sha256"] for item in descriptors}
    actual_paths = sorted((data / "validation").glob("shard-*.npz"))
    actual = {path.relative_to(data).as_posix(): sha256_file(path) for path in actual_paths}
    if actual != expected:
        raise ValueError("validation shard set or SHA256 differs from the prepared manifest")
    if any(path.startswith("test/") for path in actual):
        raise ValueError("test split must never be opened by this diagnostic")

    torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
    model = CandidateModel(ModelConfig(**checkpoint["config"]))
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.float().to(device).eval()

    rows_seen = candidates_seen = groups_seen = 0
    predicted_raw = Counter()
    predicted_sum = Counter()
    predicted_lme = Counter()
    predicted_mean = Counter()
    demonstrated = Counter()
    predicted_by_context = defaultdict(Counter)
    demonstrated_by_context = defaultdict(Counter)
    group_kind = defaultdict(lambda: {"groups": 0, "candidates": 0, "positive_groups": 0,
                                      "positive_candidates": 0, "rows": 0})
    group_sizes_by_kind = defaultdict(list)
    positive_group_sizes_by_kind = defaultdict(list)
    row_stats = Counter()
    positive_probability = defaultdict(list)
    group_size_samples = []
    sum_lme_switches = 0
    lme_switches = 0
    mean_switches = 0
    group_map_correct = Counter()
    method_metrics = defaultdict(Counter)
    context_metrics = defaultdict(Counter)
    context_predictions = defaultdict(Counter)
    context_demonstrations = defaultdict(Counter)
    transition_counts = defaultdict(Counter)
    changed_examples = defaultdict(list)
    pairwise = defaultdict(Counter)
    top_groups: list[dict] = []
    started = time.monotonic()

    def keep_top(item: dict, limit: int = 200):
        top_groups.append(item)
        top_groups.sort(key=lambda value: (value["group_logsumexp"], value["group_max_score"]), reverse=True)
        if len(top_groups) > limit:
            top_groups.pop()

    with torch.inference_mode():
        for descriptor in descriptors:
            relative = descriptor["path"]
            with np.load(data / relative, allow_pickle=False) as shard:
                required = {"state", "tokens", "lengths", "actions", "offsets", "positives", "ids"}
                if not required.issubset(shard.files):
                    raise ValueError(f"missing keys in {relative}")
                state, tokens, lengths = shard["state"], shard["tokens"], shard["lengths"]
                actions, offsets, positives, ids = (shard[key] for key in ("actions", "offsets", "positives", "ids"))
                for rows, batch in collated_batches(state, tokens, lengths, actions, offsets, positives, args.batch_size):
                    model_batch = {key: value.to(device) for key, value in batch.items() if key != "positives"}
                    scores = model(**model_batch).detach().cpu().numpy()
                    for local, row in enumerate(rows):
                        begin, end = int(offsets[row]), int(offsets[row + 1])
                        count = end - begin
                        row_scores = scores[local, :count].astype(np.float64)
                        row_actions = actions[begin:end]
                        row_positive = positives[begin:end]
                        row_state = state[row]
                        group_keys, group_descriptors = action_metadata(row_actions, row_state)
                        groups = defaultdict(list)
                        for index, group_key in enumerate(group_keys):
                            groups[group_key].append(index)
                        group_items = []
                        all_logsumexp = logsumexp(row_scores)
                        for group_key, indices in groups.items():
                            values = row_scores[indices]
                            group_logsumexp = logsumexp(values)
                            group_logmeanexp = group_logsumexp - math.log(len(indices))
                            group_mean = float(values.mean())
                            pos_count = int(row_positive[indices].sum())
                            descriptor_data = dict(group_descriptors[indices[0]])
                            group_items.append({"key": group_key, "indices": indices,
                                                "descriptor": descriptor_data,
                                                "candidate_count": len(indices),
                                                "positive_count": pos_count,
                                                "group_logsumexp": group_logsumexp,
                                                "group_logmeanexp": group_logmeanexp,
                                                "group_mean": group_mean,
                                                "group_max_score": float(values.max())})
                            kind = descriptor_data["kind"]
                            row_kind = "lead" if bool(row_state[125]) else "follow"
                            stat = group_kind[(row_kind, kind)]
                            stat["groups"] += 1
                            stat["candidates"] += len(indices)
                            stat["positive_groups"] += int(pos_count > 0)
                            stat["positive_candidates"] += pos_count
                            stat["rows"] += 1
                            group_sizes_by_kind[(row_kind, kind)].append(len(indices))
                            if pos_count:
                                positive_group_sizes_by_kind[(row_kind, kind)].append(len(indices))
                            group_size_samples.append(len(indices))
                        group_values = np.asarray([item["group_logsumexp"] for item in group_items])
                        lme_values = np.asarray([item["group_logmeanexp"] for item in group_items])
                        mean_values = np.asarray([item["group_mean"] for item in group_items])
                        group_sum_probabilities = softmax(group_values - 0.0)
                        group_lme_probabilities = softmax(lme_values)
                        group_mean_probabilities = softmax(mean_values)
                        for item_index, item in enumerate(group_items):
                            item["group_sum_probability"] = float(group_sum_probabilities[item_index])
                            item["group_logmeanexp_probability"] = float(group_lme_probabilities[item_index])
                            item["group_mean_probability"] = float(group_mean_probabilities[item_index])
                        raw_index = int(np.argmax(row_scores))
                        sum_index = int(np.argmax(group_values))
                        lme_index = int(np.argmax(lme_values))
                        mean_index = int(np.argmax(mean_values))
                        raw_group = next(i for i, item in enumerate(group_items) if raw_index in item["indices"])
                        positive_groups = {i for i, item in enumerate(group_items) if item["positive_count"] > 0}
                        lead_follow = "lead" if bool(row_state[125]) else "follow"
                        target_indices = np.flatnonzero(row_positive)
                        target_kinds = {group_descriptors[int(index)]["kind"] for index in target_indices}
                        demonstrated[lead_follow + ":" + "/".join(sorted(target_kinds))] += 1
                        predicted_raw[lead_follow + ":" + group_items[raw_group]["descriptor"]["kind"]] += 1
                        predicted_sum[lead_follow + ":" + group_items[sum_index]["descriptor"]["kind"]] += 1
                        predicted_lme[lead_follow + ":" + group_items[lme_index]["descriptor"]["kind"]] += 1
                        predicted_mean[lead_follow + ":" + group_items[mean_index]["descriptor"]["kind"]] += 1
                        demonstrated_by_context[lead_follow].update(target_kinds)
                        predicted_by_context["raw:" + lead_follow][group_items[raw_group]["descriptor"]["kind"]] += 1
                        predicted_by_context["sum:" + lead_follow][group_items[sum_index]["descriptor"]["kind"]] += 1
                        predicted_by_context["logmeanexp:" + lead_follow][group_items[lme_index]["descriptor"]["kind"]] += 1
                        predicted_by_context["mean:" + lead_follow][group_items[mean_index]["descriptor"]["kind"]] += 1
                        raw_correct = bool(row_positive[raw_index])
                        hand_size = int(round(float(row_state[:54].sum()) * 2))
                        hand_bin = "1-5" if hand_size <= 5 else "6-13" if hand_size <= 13 else "14-27"
                        detailed_context = lead_follow + ":hand" + hand_bin
                        target_kind = "/".join(sorted(target_kinds))
                        context_demonstrations[detailed_context][target_kind] += 1
                        finishing = row_actions[:, 127] > .5
                        for method, chosen_group in (("raw", raw_group), ("sum", sum_index),
                                                     ("logmeanexp", lme_index), ("mean", mean_index)):
                            members = group_items[chosen_group]["indices"]
                            concrete = int(members[int(np.argmax(row_scores[members]))])
                            chosen_kind = group_items[chosen_group]["descriptor"]["kind"]
                            correct_group = chosen_group in positive_groups
                            correct_face = bool(row_positive[concrete])
                            for meter in (method_metrics[method], context_metrics[method + ":" + detailed_context]):
                                meter["rows"] += 1
                                meter["group_correct"] += int(correct_group)
                                meter["face_correct"] += int(correct_face)
                                meter["choices"] += int(count > 1)
                                meter["choice_group_correct"] += int(count > 1 and correct_group)
                                meter["choice_face_correct"] += int(count > 1 and correct_face)
                                meter["has_finishing_candidate"] += int(finishing.any())
                                meter["selects_finish"] += int(finishing[concrete])
                                meter["demo_finish"] += int(finishing[row_positive].any())
                            context_predictions[method + ":" + detailed_context][chosen_kind] += 1
                            if method != "raw":
                                raw_kind = group_items[raw_group]["descriptor"]["kind"]
                                transition_counts[method][lead_follow + ":" + raw_kind + "->" + chosen_kind] += 1
                                if (raw_group != chosen_group and len(changed_examples[method]) < 24
                                        and (chosen_kind != raw_kind or correct_group != (raw_group in positive_groups))):
                                    example_groups = []
                                    example_indexes = list(dict.fromkeys([raw_group, chosen_group, *sorted(positive_groups)]))
                                    for gi in example_indexes:
                                        item = group_items[gi]
                                        example_groups.append({key: value for key, value in item.items() if key not in ("key", "indices")})
                                    changed_examples[method].append({
                                        "source_id": str(ids[row]), "shard": relative, "row": int(row),
                                        "hand_size": hand_size, "leading": bool(row_state[125]),
                                        "raw_kind": raw_kind, "chosen_kind": chosen_kind,
                                        "raw_group_correct": raw_group in positive_groups,
                                        "chosen_group_correct": correct_group,
                                        "chosen_face_correct": correct_face,
                                        "raw_score": float(row_scores[raw_index]),
                                        "chosen_member_score": float(row_scores[concrete]),
                                        "groups_raw_chosen_then_positive": example_groups,
                                    })
                        method_correct = {}
                        method_group_correct = {}
                        for method, chosen_group in (("raw", raw_group), ("sum", sum_index),
                                                     ("logmeanexp", lme_index), ("mean", mean_index)):
                            members = group_items[chosen_group]["indices"]
                            concrete = int(members[int(np.argmax(row_scores[members]))])
                            method_correct[method] = bool(row_positive[concrete])
                            method_group_correct[method] = chosen_group in positive_groups
                        for left, right in (("raw", "sum"), ("raw", "logmeanexp"), ("raw", "mean")):
                            pairwise[left + "_vs_" + right]["rows"] += 1
                            pairwise[left + "_vs_" + right]["left_face_right_wrong"] += int(method_correct[left] and not method_correct[right])
                            pairwise[left + "_vs_" + right]["left_face_wrong_right_correct"] += int(not method_correct[left] and method_correct[right])
                            pairwise[left + "_vs_" + right]["left_group_right_wrong"] += int(method_group_correct[left] and not method_group_correct[right])
                            pairwise[left + "_vs_" + right]["left_group_wrong_right_correct"] += int(not method_group_correct[left] and method_group_correct[right])
                        sum_correct = sum_index in positive_groups
                        lme_correct = lme_index in positive_groups
                        mean_correct = mean_index in positive_groups
                        group_map_correct.update({"raw_candidate": raw_correct, "group_sum": sum_correct,
                                                  "group_logmeanexp": lme_correct, "group_mean": mean_correct})
                        sum_lme_switches += int(raw_group != sum_index)
                        lme_switches += int(raw_group != lme_index)
                        mean_switches += int(raw_group != mean_index)
                        row_stats["rows"] += 1
                        row_stats["positive_groups_multiple"] += int(len(positive_groups) > 1)
                        row_stats["groups"] += len(group_items)
                        row_stats["candidates"] += count
                        row_stats["raw_positive_candidates"] += int(raw_correct)
                        row_stats["sum_positive_groups"] += int(sum_correct)
                        row_stats["logmeanexp_positive_groups"] += int(lme_correct)
                        row_stats["mean_positive_groups"] += int(mean_correct)
                        row_stats["group_count_sum"] += len(group_items)
                        row_stats["largest_group"] = max(row_stats["largest_group"], max(len(item["indices"]) for item in group_items))
                        denominator = all_logsumexp
                        positive_logsumexp = logsumexp(row_scores[row_positive])
                        positive_probability["raw_candidate_sum"].append(float(math.exp(positive_logsumexp - denominator)))
                        for name, values in (("group_sum", group_values), ("group_logmeanexp", lme_values), ("group_mean", mean_values)):
                            probabilities = softmax(values)
                            positive_probability[name].append(float(probabilities[[i in positive_groups for i in range(len(group_items))]].sum()))
                        for item in group_items:
                            keep_top({"source_id": str(ids[row]), "shard": relative, "row": int(row),
                                      "leading": bool(row_state[125]), "descriptor": item["descriptor"],
                                      "candidate_count": item["candidate_count"], "positive_count": item["positive_count"],
                                      "group_max_score": item["group_max_score"], "group_mean": item["group_mean"],
                                      "group_logsumexp": item["group_logsumexp"], "group_logmeanexp": item["group_logmeanexp"],
                                      "group_sum_probability": item["group_sum_probability"],
                                      "group_logmeanexp_probability": item["group_logmeanexp_probability"],
                                      "group_mean_probability": item["group_mean_probability"],
                                      "top_member_indices": [int(item["indices"][index]) for index in np.argsort(row_scores[item["indices"]])[::-1][:3]]})
                        rows_seen += 1
                        candidates_seen += count
                        groups_seen += len(group_items)

    # Hashes are checked again after inference so the report cannot silently
    # combine scores from a moving validation set with a different manifest.
    if sha256_file(manifest_path) != manifest_sha or any(sha256_file(data / path) != digest for path, digest in actual.items()):
        raise ValueError("validation inputs changed during diagnosis")
    if sha256_file(checkpoint_path) != checkpoint_sha:
        raise ValueError("checkpoint changed during diagnosis")

    def metric_report(meter):
        out = dict(meter)
        out.update(group_accuracy=meter["group_correct"] / max(1, meter["rows"]),
                   face_accuracy=meter["face_correct"] / max(1, meter["rows"]),
                   choice_group_accuracy=meter["choice_group_correct"] / max(1, meter["choices"]),
                   choice_face_accuracy=meter["choice_face_correct"] / max(1, meter["choices"]))
        return out
    result = {
        "schema": "oxbot-candidate-group-diagnosis-v1",
        "status": "complete",
        "scope": "frozen v1 checkpoint; validation-only strategic grouping; not test evaluation or gameplay strength",
        "split": "validation", "test_split_read": False, "all_candidates": True,
        "checkpoint": str(checkpoint_path), "checkpoint_sha256": checkpoint_sha,
        "data_manifest_sha256": manifest_sha, "validation_shards": actual,
        "feature_version": FEATURE_VERSION, "model_config": checkpoint["config"],
        "torch": str(torch.__version__), "device": str(device), "threads": args.threads,
        "script_sha256": sha256_file(Path(__file__)), "seconds": time.monotonic() - started,
        "rows": int(rows_seen), "candidates": int(candidates_seen), "strategic_groups": int(groups_seen),
        "group_definition": {
            "key": "kind,length,key,secondary,actual rank-count multiset,wildcard count",
            "rank_names": list(RANK_NAMES),
            "wildcard": "heart suit (suit index 0) at current level; counted from actual face cards",
            "straight_flush": "full canonical face-count/suit signature retained",
            "candidate_count": "all legal validation candidates; no sampling",
            "positive_count": "prepared BC positive labels in the group; no extra same-rank labels invented",
            "probability": "raw_candidate_sum counts only positive face candidates; group probabilities count every member of a group containing a positive",
        },
        "row_summary": {key: int(value) for key, value in row_stats.items()},
        "argmax": {
            "raw_candidate": {"correct_rows": int(group_map_correct["raw_candidate"]), "accuracy": group_map_correct["raw_candidate"] / max(1, rows_seen)},
            "group_sum_logsumexp": {"correct_rows": int(group_map_correct["group_sum"]), "accuracy": group_map_correct["group_sum"] / max(1, rows_seen)},
            "group_logmeanexp": {"correct_rows": int(group_map_correct["group_logmeanexp"]), "accuracy": group_map_correct["group_logmeanexp"] / max(1, rows_seen)},
            "group_mean_score": {"correct_rows": int(group_map_correct["group_mean"]), "accuracy": group_map_correct["group_mean"] / max(1, rows_seen)},
            "raw_to_group_sum_switches": int(sum_lme_switches),
            "raw_to_group_logmeanexp_switches": int(lme_switches),
            "raw_to_group_mean_switches": int(mean_switches),
            "note": "sum/logsumexp preserves candidate multiplicity; logmeanexp and mean neutralize it",
        },
        "positive_group_probability": {
            name: {"mean": float(np.mean(values)), "p50": float(np.percentile(values, 50)),
                   "p10": float(np.percentile(values, 10)), "p90": float(np.percentile(values, 90))}
            for name, values in positive_probability.items()
        },
        "method_metrics": {name: metric_report(meter) for name, meter in method_metrics.items()},
        "hand_context_metrics": {name: metric_report(meter) for name, meter in sorted(context_metrics.items())},
        "hand_context_predicted": {name: summarize_distribution(counts) for name, counts in sorted(context_predictions.items())},
        "hand_context_demonstrated": {name: summarize_distribution(counts) for name, counts in sorted(context_demonstrations.items())},
        "transitions": {name: summarize_distribution(counts) for name, counts in sorted(transition_counts.items())},
        "pairwise": {name: {key: int(value) for key, value in counts.items()}
                     for name, counts in sorted(pairwise.items())},
        "changed_examples": dict(changed_examples),
        "predicted_raw": summarize_distribution(predicted_raw),
        "predicted_group_sum": summarize_distribution(predicted_sum),
        "predicted_group_logmeanexp": summarize_distribution(predicted_lme),
        "predicted_group_mean": summarize_distribution(predicted_mean),
        "demonstrated": summarize_distribution(demonstrated),
        "predicted_by_context": {key: summarize_distribution(value) for key, value in sorted(predicted_by_context.items())},
        "demonstrated_by_context": {key: summarize_distribution(value) for key, value in sorted(demonstrated_by_context.items())},
        "group_kind_summary": {f"{context}:{kind}": values for (context, kind), values in sorted(group_kind.items())},
        "group_size_by_kind": {
            f"{context}:{kind}": {
                "groups": len(values), "mean": float(np.mean(values)),
                "p50": float(np.percentile(values, 50)), "p95": float(np.percentile(values, 95)),
                "max": int(max(values)),
            }
            for (context, kind), values in sorted(group_sizes_by_kind.items())
        },
        "positive_group_size_by_kind": {
            f"{context}:{kind}": {
                "positive_groups": len(values), "mean": float(np.mean(values)),
                "p50": float(np.percentile(values, 50)), "p95": float(np.percentile(values, 95)),
                "max": int(max(values)),
            }
            for (context, kind), values in sorted(positive_group_sizes_by_kind.items())
        },
        "group_size": {"mean": float(np.mean(group_size_samples)), "p50": float(np.percentile(group_size_samples, 50)),
                       "p95": float(np.percentile(group_size_samples, 95)), "max": int(max(group_size_samples))},
        "top_groups_by_logsumexp": top_groups,
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "models/bc-v1/best.pt")
    parser.add_argument("--expected-checkpoint-sha")
    parser.add_argument("--data", type=Path, default=ROOT / "data/processed/bc-v1")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--report", type=Path, default=ROOT / "reports/bc-v1-candidate-groups.json")
    args = parser.parse_args()
    if args.threads < 1 or args.batch_size < 1:
        raise ValueError("threads and batch size must be positive")
    report = run(args)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "rows": report["rows"], "candidates": report["candidates"],
                      "groups": report["strategic_groups"], "argmax": report["argmax"],
                      "seconds": report["seconds"]}), flush=True)


if __name__ == "__main__":
    main()
