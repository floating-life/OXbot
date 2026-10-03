"""Small, read-only residual-feature audit on development splits.

This intentionally opens only ``train.jsonl`` and ``validation.jsonl``.  It
samples a fixed number of valid play rows per split and (by default) at most
32 candidates per row, so an unusually large legal-move set cannot turn an
audit into an accidental full training-data pass.  No model is loaded and no
held-out test file is opened.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import random
import sys
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
from probe import Probe  # noqa: E402
try:  # Running as ``python train/audit...`` versus importing as ``train.*``.
    from residual_features import (  # type: ignore  # noqa: E402
        residual_features,
        summarize_lead_moves,
    )
except ModuleNotFoundError:  # pragma: no cover - import-mode compatibility
    from train.residual_features import residual_features, summarize_lead_moves  # noqa: E402


SPLITS = ("train", "validation")
DECISION_SCHEMA = "njupt-decision-v3"
AUDIT_SCHEMA = "oxbot-candidate-residual-audit-v1"
RANKS = "A234567890JQK"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _valid_row(record: Any) -> bool:
    if not isinstance(record, dict) or record.get("schema") != DECISION_SCHEMA:
        return False
    if record.get("stage") != "play":
        return False
    feature = record.get("features")
    if not isinstance(feature, dict):
        return False
    hand = feature.get("own_hand")
    if (not isinstance(hand, list) or not hand or len(hand) > 27 or
            any(type(card) is not int or not 0 <= card < 108 for card in hand) or
            len(set(hand)) != len(hand)):
        return False
    level = feature.get("level_label")
    if level == "10":
        level = "0"
    if not isinstance(level, str) or level not in RANKS:
        return False
    leading = feature.get("leading")
    if type(leading) is not bool:
        return False
    if not leading:
        previous_cards = feature.get("last_cards")
        previous_claim = feature.get("last_claim")
        if (not isinstance(previous_cards, list) or not previous_cards or
                not isinstance(previous_claim, list) or len(previous_cards) != len(previous_claim) or
                any(type(card) is not int or not 0 <= card < 108 for card in previous_cards + previous_claim)):
            return False
    provenance = record.get("provenance")
    return (isinstance(provenance, dict) and isinstance(provenance.get("data_sha256"), str) and
            len(provenance["data_sha256"]) == 64 and type(record.get("deal_id")) is int and
            type(record.get("event_index")) is int)


def _observation(record: dict[str, Any]) -> dict[str, Any]:
    feature = record["features"]
    level = "0" if feature["level_label"] == "10" else feature["level_label"]
    previous = None
    if not feature["leading"]:
        previous = [list(feature["last_cards"]), list(feature["last_claim"])]
    return {
        "hand": list(feature["own_hand"]),
        "level": level,
        "leading": feature["leading"],
        "previous": previous,
    }


def _identity(record: dict[str, Any], line: int) -> str:
    provenance = record.get("provenance") or {}
    digest = provenance.get("data_sha256", "")
    return f"{digest[:16]}:d{record.get('deal_id')}:e{record.get('event_index')}@{line}"


def sample_rows(path: Path, limit: int, seed: int) -> tuple[list[tuple[int, dict[str, Any]]], Counter[str]]:
    """Reservoir-sample valid rows without opening any other split."""
    rng = random.Random(seed)
    rows: list[tuple[int, dict[str, Any]]] = []
    counts: Counter[str] = Counter()
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            counts["lines"] += 1
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                counts["malformed"] += 1
                continue
            if not _valid_row(record):
                counts["invalid_or_nonplay"] += 1
                continue
            counts["valid_play"] += 1
            item = (line_number, record)
            if len(rows) < limit:
                rows.append(item)
            else:
                slot = rng.randrange(counts["valid_play"])
                if slot < limit:
                    rows[slot] = item
    rows.sort(key=lambda item: item[0])
    return rows, counts


def _quantiles(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "min": None, "p50": None, "p95": None, "p99": None, "max": None}
    values = sorted(values)
    def q(frac: float) -> float:
        position = (len(values) - 1) * frac
        low = math.floor(position)
        high = math.ceil(position)
        if low == high:
            return float(values[low])
        return float(values[low] + (values[high] - values[low]) * (position - low))
    return {"count": len(values), "min": float(values[0]), "p50": q(.50),
            "p95": q(.95), "p99": q(.99), "max": float(values[-1])}


def audit_split(path: Path, probe: Probe, rows: list[tuple[int, dict[str, Any]]], max_candidates: int) -> dict[str, Any]:
    counters: Counter[str] = Counter()
    metrics: dict[str, list[float]] = {name: [] for name in (
        "base_lead_total", "residual_lead_total", "residual_lead_power",
        "lead_total_delta", "lead_log_total_delta", "residual_bomb_max_length",
    )}
    candidate_kinds: Counter[str] = Counter()
    examples: list[dict[str, Any]] = []
    for line_number, record in rows:
        counters["rows_selected"] += 1
        identity = _identity(record, line_number)
        obs = _observation(record)
        generated = probe.call(command="generate", hand=obs["hand"], level=obs["level"],
                               leading=obs["leading"], previous=obs["previous"], metadata=True)
        if "error" in generated:
            counters["rows_probe_error"] += 1
            continue
        moves = generated.get("moves", [])
        types = generated.get("types", [])
        if not isinstance(moves, list) or not isinstance(types, list) or len(moves) != len(types):
            counters["rows_probe_shape_error"] += 1
            continue
        counters["rows_with_candidates"] += 1
        counters["candidates_available"] += len(moves)
        if not moves:
            continue
        lead = probe.call(command="generate", hand=obs["hand"], level=obs["level"],
                          leading=True, metadata=True)
        if "error" in lead:
            counters["rows_future_probe_error"] += 1
            continue
        try:
            base = summarize_lead_moves(lead.get("moves", []), lead.get("types", []))
        except (TypeError, ValueError, KeyError):
            counters["rows_future_summary_error"] += 1
            continue
        indices = list(range(len(moves)))
        if len(indices) > max_candidates:
            # Deterministic prefix is intentionally avoided: generated moves
            # are sorted by kind, which would bias this audit toward singles.
            local = random.Random(f"{identity}:{max_candidates}")
            indices = sorted(local.sample(indices, max_candidates))
            counters["rows_candidate_sampled"] += 1
        for index in indices:
            counters["candidate_rows_evaluated"] += 1
            move = moves[index]
            meta = types[index] if index < len(types) else {}
            kind = meta.get("kind") if isinstance(meta, dict) else "unknown"
            candidate_kinds[str(kind)] += 1
            action = move[0] if isinstance(move, list) and len(move) == 2 else []
            if not action:
                counters["pass_candidates"] += 1
            validation = probe.call(command="validate", hand=obs["hand"], level=obs["level"],
                                    leading=obs["leading"], previous=obs["previous"], move=move)
            if not validation.get("ok", False):
                counters["candidate_validate_errors"] += 1
            try:
                row = residual_features(probe, obs["hand"], obs["level"], action, base_summary=base)
            except ValueError as exc:
                if "sub-multiset" in str(exc):
                    counters["physical_deletion_errors"] += 1
                else:
                    counters["candidate_feature_errors"] += 1
                if len(examples) < 10:
                    examples.append({"id": identity, "candidate": index, "error": str(exc)})
                continue
            counters["unknown_metadata"] += int(str(kind) == "unknown")
            for name in metrics:
                metrics[name].append(float(row[name]))
    return {
        "source": path.name,
        "counters": dict(counters),
        "candidate_kinds": dict(sorted(candidate_kinds.items())),
        "metrics": {name: _quantiles(values) for name, values in metrics.items()},
        "errors": examples,
    }


def run(source: Path, output: Path, probe_path: Path, *, rows: int = 128,
        max_candidates: int = 32, seed: int = 20261002, timeout: float = 20) -> dict[str, Any]:
    if rows <= 0 or max_candidates <= 0:
        raise ValueError("rows and max_candidates must be positive")
    source = source.resolve()
    output = output.resolve()
    probe_path = probe_path.resolve()
    split_sha: dict[str, str] = {}
    sampled: dict[str, list[tuple[int, dict[str, Any]]]] = {}
    sampling_counts: dict[str, dict[str, int]] = {}
    for offset, split in enumerate(SPLITS):
        path = source / f"{split}.jsonl"
        if not path.is_file():
            raise ValueError(f"missing development split: {path}")
        split_sha[split] = sha256_file(path)
        sampled[split], counts = sample_rows(path, rows, seed + offset)
        sampling_counts[split] = dict(counts)
        if len(sampled[split]) < rows:
            raise ValueError(f"only {len(sampled[split])} valid rows in {split}; need {rows}")
    with Probe(probe_path, timeout=timeout) as probe:
        split_results = {
            split: audit_split(source / f"{split}.jsonl", probe, sampled[split], max_candidates)
            for split in SPLITS
        }
    payload = {
        "schema": AUDIT_SCHEMA,
        "status": "complete",
        "source": str(source),
        "probe": str(probe_path),
        "probe_sha256": sha256_file(probe_path),
        "scripts_sha256": {
            "audit_residual_features.py": sha256_file(Path(__file__).resolve()),
            "residual_features.py": sha256_file(Path(__file__).with_name("residual_features.py")),
        },
        "source_split_sha256": split_sha,
        "splits_read": list(SPLITS),
        "test_used": False,
        "row_sample": {"rows_per_split": rows, "max_candidates_per_row": max_candidates,
                       "seed": seed, "sampling": "reservoir valid play rows; per-row candidate sample"},
        "sampling_counts": sampling_counts,
        "splits": split_results,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(output)
    payload["output_sha256"] = sha256_file(output)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "data" / "processed" / "njupt")
    parser.add_argument("--output", type=Path, default=ROOT / "reports" / "candidate-local-residual-audit-v1.json")
    parser.add_argument("--probe", type=Path, default=ROOT / "bin" / "core_probe_rules")
    parser.add_argument("--rows", type=int, default=128)
    parser.add_argument("--max-candidates", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20261002)
    parser.add_argument("--timeout", type=float, default=20)
    args = parser.parse_args()
    payload = run(args.source, args.output, args.probe, rows=args.rows,
                  max_candidates=args.max_candidates, seed=args.seed, timeout=args.timeout)
    print(json.dumps({"status": payload["status"], "output": str(args.output.resolve()),
                      "sha256": payload["output_sha256"], "test_used": payload["test_used"],
                      "rows": {split: payload["splits"][split]["counters"] for split in SPLITS}},
               ensure_ascii=False))


if __name__ == "__main__":
    main()
