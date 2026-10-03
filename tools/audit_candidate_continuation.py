"""Audit a train/validation candidate-continuation sidecar without training.

Only the sidecar's ``train`` and ``validation`` maps and matching prepared
shards are opened.  The script deliberately does not enumerate or open a
``test`` directory.  It checks candidate-count alignment, labels the prepared
candidate features by kind/size, and measures whether the demonstrated BC
positive agrees with the observational continuation label.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


SPLITS = ("train", "validation")
KINDS = ("pass", "invalid", "single", "pair", "three", "straight", "set",
         "three_straight", "triple_pairs", "bomb", "straight_flush", "rocket")
SCHEMA = "oxbot-candidate-continuation-audit-v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _bin(count: int) -> str:
    if count == 1:
        return "1"
    if count <= 3:
        return "2-3"
    if count <= 8:
        return "4-8"
    if count <= 16:
        return "9-16"
    if count <= 32:
        return "17-32"
    if count <= 64:
        return "33-64"
    if count <= 128:
        return "65-128"
    if count <= 256:
        return "129-256"
    if count <= 512:
        return "257-512"
    return "513+"


def _q(values: list[int | float]) -> dict[str, int | float | None]:
    if not values:
        return {"count": 0, "min": None, "p50": None, "p95": None, "p99": None, "max": None}
    values = sorted(float(value) for value in values)
    def quantile(frac: float) -> float:
        pos = (len(values) - 1) * frac
        lo, hi = math.floor(pos), math.ceil(pos)
        if lo == hi:
            return values[lo]
        return values[lo] + (values[hi] - values[lo]) * (pos - lo)
    return {"count": len(values), "min": values[0], "p50": quantile(.5),
            "p95": quantile(.95), "p99": quantile(.99), "max": values[-1]}


def _feature_signature(row: np.ndarray) -> str:
    """Reconstruct build_candidate_credit_targets' signature from a shard row."""
    action = [int(value) for value in np.rint(row[:54] * 2).astype(np.int64)]
    claim = [int(value) for value in np.rint(row[54:108] * 2).astype(np.int64)]
    kind_id = int(np.argmax(row[108:120]))
    kind = KINDS[kind_id] if 0 <= kind_id < len(KINDS) else "unknown"
    key = -1 if kind == "pass" else int(round(float(row[121]) * 14))
    # action_features clamps negative secondary keys to zero.  Only ``set``
    # has a meaningful secondary rank, so recover -1 for all other kinds.
    secondary = int(round(float(row[122]) * 14)) if kind == "set" else -1
    payload = {"action": action, "claim": claim, "kind": kind,
               "key": key, "secondary": secondary}
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def _candidate_fingerprint(rows: np.ndarray) -> str:
    payload = "\n".join(_feature_signature(row) for row in rows).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def audit_split(prepared: Path, split: str, targets: dict[str, Any]) -> dict[str, Any]:
    files = sorted((prepared / split).glob("*.npz"))
    if not files:
        raise ValueError(f"no prepared {split} shards")
    counters: Counter[str] = Counter()
    count_bins: Counter[str] = Counter()
    horizon: list[int] = []
    label_by_kind: dict[str, Counter[str]] = defaultdict(Counter)
    label_by_size: dict[str, Counter[str]] = defaultdict(Counter)
    demo_labels: Counter[str] = Counter()
    row_margin: list[int] = []
    seen: set[str] = set()
    for path in files:
        with np.load(path, allow_pickle=False) as data:
            for key in ("ids", "offsets", "actions", "positives"):
                if key not in data.files:
                    raise ValueError(f"{path} lacks {key}")
            ids, offsets = data["ids"], data["offsets"]
            actions, positives = data["actions"], data["positives"]
            if len(offsets) != len(ids) + 1 or int(offsets[-1]) != len(actions) or len(positives) != len(actions):
                raise ValueError(f"{path} has malformed offsets")
            for index, raw_id in enumerate(ids):
                identity = str(raw_id)
                target = targets.get(identity)
                if target is None:
                    continue
                if identity in seen:
                    raise ValueError(f"duplicate target ID in prepared {split}: {identity}")
                seen.add(identity)
                begin, end = int(offsets[index]), int(offsets[index + 1])
                labels = target.get("labels") if isinstance(target, dict) else None
                if not isinstance(labels, list) or len(labels) != end - begin:
                    counters["candidate_count_mismatch"] += 1
                    continue
                if any(type(value) is not int or value not in (-1, 0, 1) for value in labels):
                    counters["invalid_label"] += 1
                    continue
                counters["target_rows"] += 1
                count = end - begin
                count_bins[_bin(count)] += 1
                expected_fingerprint = target.get("candidate_fingerprint")
                if isinstance(expected_fingerprint, str):
                    actual_fingerprint = _candidate_fingerprint(actions[begin:end])
                    if actual_fingerprint != expected_fingerprint:
                        counters["candidate_fingerprint_mismatch"] += 1
                        if counters["candidate_fingerprint_mismatch"] <= 10:
                            counters["candidate_fingerprint_mismatch_examples"] += 1
                if isinstance(target.get("horizon"), int):
                    horizon.append(int(target["horizon"]))
                pos = sum(value > 0 for value in labels)
                neg = sum(value < 0 for value in labels)
                if pos and neg:
                    counters["pair_rows"] += 1
                    row_margin.append(pos - neg)
                counters["positive_labels"] += pos
                counters["negative_labels"] += neg
                counters["ignored_labels"] += sum(value == 0 for value in labels)
                kind_ids = np.argmax(actions[begin:end, 108:120], axis=1)
                sizes = np.rint(actions[begin:end, 120] * 10).astype(np.int64)
                row_positive = positives[begin:end]
                for label, kind_id, size, is_demo in zip(labels, kind_ids, sizes, row_positive, strict=True):
                    kind = KINDS[int(kind_id)] if 0 <= int(kind_id) < len(KINDS) else "unknown"
                    label_name = "+1" if label > 0 else "-1" if label < 0 else "0"
                    label_by_kind[kind][label_name] += 1
                    label_by_size[str(int(size))][label_name] += 1
                    counters["demo_positive_candidates"] += int(bool(is_demo))
                    if is_demo:
                        demo_labels[label_name] += 1
    missing = set(targets) - seen
    counters["target_rows_missing_prepared"] = len(missing)
    return {
        "source": str(prepared / split),
        "shard_count": len(files),
        "counters": dict(counters),
        "candidate_count_bins": dict(count_bins),
        "horizon": _q(horizon),
        "row_label_margin_pos_minus_neg": _q(row_margin),
        "demonstrated_positive_label": dict(demo_labels),
        "labels_by_kind": {key: dict(value) for key, value in sorted(label_by_kind.items())},
        "labels_by_size": {key: dict(value) for key, value in sorted(label_by_size.items(), key=lambda item: int(item[0]))},
        "missing_target_examples": sorted(missing)[:10],
    }


def run(sidecar: Path, prepared: Path, output: Path) -> dict[str, Any]:
    payload = json.loads(sidecar.read_text(encoding="utf-8"))
    if payload.get("schema") != "oxbot-candidate-continuation-targets-v1":
        raise ValueError("unexpected sidecar schema")
    if payload.get("test_used") is not False or payload.get("splits_read") != list(SPLITS):
        raise ValueError("sidecar provenance is not train/validation-only")
    target_maps = payload.get("targets")
    if not isinstance(target_maps, dict):
        raise ValueError("sidecar has no targets")
    prepared = prepared.resolve()
    results = {split: audit_split(prepared, split, target_maps.get(split, {})) for split in SPLITS}
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    result = {
        "schema": SCHEMA,
        "status": "complete",
        "sidecar": str(sidecar.resolve()),
        "sidecar_sha256": sha256_file(sidecar),
        "prepared": str(prepared),
        "prepared_manifest_sha256": sha256_file(prepared / "manifest.json"),
        "splits_read": list(SPLITS),
        "test_used": False,
        "target_definition": payload.get("target_definition"),
        "splits": results,
    }
    temp = output.with_suffix(output.suffix + ".tmp")
    temp.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temp.replace(output)
    result["output_sha256"] = sha256_file(output)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parent.parent
    parser.add_argument("--sidecar", type=Path, default=root / "reports" / "candidate_continuation_targets_v1.json")
    parser.add_argument("--prepared", type=Path, default=root / "data" / "processed" / "bc-v2-full")
    parser.add_argument("--output", type=Path, default=root / "reports" / "candidate-continuation-audit-full.json")
    args = parser.parse_args()
    result = run(args.sidecar, args.prepared, args.output)
    print(json.dumps({"status": result["status"], "output": str(args.output.resolve()),
                      "sha256": result["output_sha256"], "test_used": result["test_used"],
                      "splits": {key: value["counters"] for key, value in result["splits"].items()}},
               ensure_ascii=False))


if __name__ == "__main__":
    main()
