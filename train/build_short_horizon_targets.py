"""Build public-trace, short-horizon action-credit targets.

This is a development-only target generator.  It intentionally opens only
``train.jsonl`` and ``validation.jsonl`` from an ETL source directory.  For
each play observation it finds the next public lead in the same deal and
records whether the acting team retained that lead.  No opponent hand,
allocation, or final result is read.  The resulting values are diagnostic
targets; they are not causal counterfactual Q values.

The script is deliberately independent of the C++ rule core.  The core can
be used later to add candidate-local features (residual legal-move counts,
preserved bombs, etc.), but this first artifact establishes an auditable
public-only target and its leakage boundary before changing the model.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
from typing import Any


DECISION_SCHEMA = "njupt-decision-v3"
SPLITS = ("train", "validation")
OUTPUT_SCHEMA = "oxbot-short-horizon-targets-v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def source_id(record: dict[str, Any]) -> str:
    """Return the same stable ID used by prepared BC records."""
    provenance = record.get("provenance") or {}
    digest = provenance.get("data_sha256")
    deal = record.get("deal_id")
    event = record.get("event_index")
    if not isinstance(digest, str) or len(digest) != 64:
        raise ValueError("missing provenance data_sha256")
    if type(deal) is not int or type(event) is not int or deal < 0 or event < 0:
        raise ValueError("missing deal/event index")
    return f"{digest[:16]}:d{deal}:e{event}"


def _int(value: Any, name: str, low: int, high: int) -> int:
    if type(value) is not int or value < low or value > high:
        raise ValueError(f"{name} is outside {low}..{high}")
    return value


def _public_observation(record: dict[str, Any]) -> dict[str, Any]:
    """Whitelist exactly the public fields needed for the credit target.

    In particular this function never reads ``result`` and never copies the
    raw record.  ``own_hand`` and ``remaining_counts`` are part of the
    established online observation contract; the latter are public card
    counts, not opponent card identities.
    """
    features = record.get("features")
    if not isinstance(features, dict):
        raise ValueError("features is not an object")
    seat = _int(features.get("seat"), "features.seat", 0, 3)
    team = _int(features.get("team"), "features.team", 0, 1)
    if team != seat % 2:
        raise ValueError("seat/team invariant failed")
    hand = features.get("own_hand")
    if not isinstance(hand, list) or any(type(card) is not int or not 0 <= card < 108 for card in hand):
        raise ValueError("features.own_hand is invalid")
    remaining = features.get("remaining_counts")
    if (not isinstance(remaining, list) or len(remaining) != 4 or
            any(type(count) is not int or not 0 <= count <= 27 for count in remaining)):
        raise ValueError("features.remaining_counts is invalid")
    if remaining[seat] != len(hand):
        raise ValueError("own hand/count invariant failed")
    leading = features.get("leading")
    if type(leading) is not bool:
        raise ValueError("features.leading is invalid")
    return {
        "seat": seat,
        "team": team,
        "own_hand_size": len(hand),
        "remaining_counts": list(remaining),
        "leading": leading,
    }


def _action_meta(record: dict[str, Any]) -> tuple[str, int]:
    label = record.get("label")
    if not isinstance(label, dict):
        return "invalid", 0
    cards = label.get("cards")
    if not isinstance(cards, list) or any(type(card) is not int or not 0 <= card < 108 for card in cards):
        return "invalid", 0
    if not cards:
        return "pass", 0
    candidates = label.get("claim_candidates")
    if isinstance(candidates, list):
        for candidate in candidates:
            if isinstance(candidate, dict) and isinstance(candidate.get("kind"), str):
                return candidate["kind"], len(cards)
    # An unknown kind is retained for audit; dropping it would bias the
    # public-control estimate towards rows with reconstructed claims.
    return "unknown", len(cards)


def _group_key(record: dict[str, Any]) -> tuple[str, str, str, int]:
    provenance = record.get("provenance") or {}
    data_hash = provenance.get("data_sha256")
    match_id = record.get("match_id")
    game_id = record.get("game_id")
    deal_id = record.get("deal_id")
    if not isinstance(data_hash, str) or not isinstance(match_id, str) or not isinstance(game_id, str):
        raise ValueError("missing public deal identity")
    if type(deal_id) is not int or deal_id < 0:
        raise ValueError("invalid deal_id")
    # The source hash is included so a coincidentally repeated match/game key
    # cannot merge data from two source archives.
    return data_hash, match_id, game_id, deal_id


def _future_lead(rows: list[dict[str, Any]], index: int) -> tuple[int, dict[str, Any]] | None:
    for future_index in range(index + 1, len(rows)):
        observation = _public_observation(rows[future_index])
        if observation["leading"]:
            return future_index, observation
    return None


def _target(rows: list[dict[str, Any]], index: int) -> dict[str, Any]:
    current = _public_observation(rows[index])
    kind, action_size = _action_meta(rows[index])
    future = _future_lead(rows, index)
    result: dict[str, Any] = {
        "action_kind": kind,
        "action_size": action_size,
        "acting_seat": current["seat"],
        "acting_team": current["team"],
        "was_leading": current["leading"],
        "credit_available": False,
        "credit": None,
        "next_lead_seat": None,
        "next_lead_team": None,
        "plays_to_next_lead": None,
        "lead_relation": "unknown",
        "remaining_delta": None,
    }
    if future is None:
        result["terminal_or_missing_next_lead"] = True
        return result
    future_index, next_observation = future
    next_team = next_observation["team"]
    relation = "same_actor" if next_observation["seat"] == current["seat"] else (
        "partner" if next_team == current["team"] else "opponent")
    # A pass does not claim a trick.  Keep its public handoff as context but
    # do not turn it into a positive/negative action-credit label.
    credit_available = action_size > 0 and kind != "invalid"
    result.update({
        "credit_available": credit_available,
        "credit": (1 if next_team == current["team"] else -1) if credit_available else None,
        "next_lead_seat": next_observation["seat"],
        "next_lead_team": next_team,
        "plays_to_next_lead": future_index - index,
        "lead_relation": relation,
        "terminal_or_missing_next_lead": False,
        # Counts are public and are included only as a local-transition audit
        # signal.  They are not model features or a private allocation.
        "remaining_delta": [
            next_observation["remaining_counts"][seat] - current["remaining_counts"][seat]
            for seat in range(4)
        ],
    })
    return result


def _summary(targets: dict[str, dict[str, Any]]) -> dict[str, Any]:
    by_kind: dict[str, Counter[str]] = defaultdict(Counter)
    by_size: dict[str, Counter[str]] = defaultdict(Counter)
    totals = Counter()
    for value in targets.values():
        kind = str(value["action_kind"])
        size = str(value["action_size"])
        totals["rows"] += 1
        by_kind[kind]["rows"] += 1
        by_size[size]["rows"] += 1
        if value["credit_available"]:
            totals["credit_rows"] += 1
            by_kind[kind]["credit_rows"] += 1
            by_size[size]["credit_rows"] += 1
            if value["credit"] > 0:
                totals["positive"] += 1
                by_kind[kind]["positive"] += 1
                by_size[size]["positive"] += 1
            else:
                totals["negative"] += 1
                by_kind[kind]["negative"] += 1
                by_size[size]["negative"] += 1
        else:
            totals["unavailable"] += 1
    def finish(table: dict[str, Counter[str]]) -> dict[str, dict[str, Any]]:
        out: dict[str, dict[str, Any]] = {}
        for key in sorted(table, key=lambda x: (len(x), x)):
            row = dict(table[key])
            row["control_rate"] = (row.get("positive", 0) / row["credit_rows"]
                                    if row.get("credit_rows", 0) else None)
            out[key] = row
        return out
    return {"totals": dict(totals), "by_kind": finish(by_kind), "by_size": finish(by_size)}


def read_split(path: Path, split: str, summary: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Read one development split and return only that split's targets."""
    groups: dict[tuple[str, str, str, int], list[dict[str, Any]]] = defaultdict(list)
    split_targets: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            if not isinstance(record, dict) or record.get("schema") != DECISION_SCHEMA:
                raise ValueError(f"{path}:{line_number}: unexpected decision schema")
            if record.get("stage") != "play":
                summary["non_play"] += 1
                continue
            # This reads no result field.  The target is computed solely from
            # current public fields and later public play rows in this split.
            _public_observation(record)
            key = source_id(record)
            if key in split_targets:
                raise ValueError(f"duplicate target key {key} in {split}")
            groups[_group_key(record)].append(record)
            summary["play_rows"] += 1
    for rows in groups.values():
        rows.sort(key=lambda row: _int(row.get("event_index"), "event_index", 0, 10**9))
        for index, record in enumerate(rows):
            key = source_id(record)
            split_targets[key] = _target(rows, index)
    return split_targets
    summary["groups"] += len(groups)


def run(source: Path, output: Path) -> dict[str, Any]:
    source = source.resolve()
    manifest_path = source / "manifest.json"
    if not manifest_path.is_file():
        raise ValueError(f"source manifest does not exist: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != "njupt-etl-manifest-v3" or manifest.get("decision_schema") != DECISION_SCHEMA:
        raise ValueError("source manifest schema is incompatible")
    manifest_outputs = manifest.get("output_files")
    if not isinstance(manifest_outputs, dict):
        raise ValueError("source manifest has no output_files metadata")
    manifest_sha = sha256_file(manifest_path)
    split_sha: dict[str, str] = {}
    paths: dict[str, Path] = {}
    for split in SPLITS:
        path = source / f"{split}.jsonl"
        if not path.is_file():
            raise ValueError(f"required development split does not exist: {path}")
        metadata = manifest_outputs.get(f"{split}.jsonl")
        expected = metadata.get("sha256") if isinstance(metadata, dict) else None
        actual = sha256_file(path)
        if not isinstance(expected, str) or actual != expected:
            raise ValueError(f"{split}.jsonl SHA256 differs from source manifest")
        split_sha[split] = actual
        paths[split] = path
    targets: dict[str, dict[str, Any]] = {}
    split_summaries: dict[str, Any] = {}
    for split in SPLITS:
        counts: Counter[str] = Counter()
        split_targets = read_split(paths[split], split, counts)
        # Merge only after computing the split-local summary.  Provenance
        # data_sha256 is an archive hash, not the JSONL file hash, so it must
        # never be used as a prefix to infer split membership.
        overlap = set(targets).intersection(split_targets)
        if overlap:
            raise ValueError(f"target IDs overlap across development splits: {sorted(overlap)[:3]}")
        targets.update(split_targets)
        summary = _summary(split_targets)
        summary["counts"] = dict(counts)
        summary["target_count"] = len(split_targets)
        split_summaries[split] = summary
    # The explicit test_used field and split list are machine-checkable guard
    # rails used by future training code.
    payload = {
        "schema": OUTPUT_SCHEMA,
        "status": "complete",
        "source": str(source),
        "source_manifest_sha256": manifest_sha,
        "source_split_sha256": split_sha,
        "splits_read": list(SPLITS),
        "test_used": False,
        "target_definition": {
            "credit": "+1 when the acting non-pass action's team owns the next public lead, -1 otherwise",
            "pass": "credit unavailable; handoff retained as context",
            "horizon": "first later play row with leading=true in the same source/deal group",
            "hidden_information": "no result/final winner/opponent hand identity is read",
        },
        "split_summaries": split_summaries,
        "target_count": len(targets),
        "targets": targets,
    }
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temp = output.with_suffix(output.suffix + ".tmp")
    temp.write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n", encoding="utf-8")
    temp.replace(output)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("data/processed/njupt"))
    parser.add_argument("--output", type=Path, default=Path("reports/short_horizon_targets_v1.json"))
    args = parser.parse_args()
    payload = run(args.source, args.output)
    digest = sha256_file(args.output.resolve())
    print(json.dumps({
        "status": payload["status"],
        "output": str(args.output.resolve()),
        "target_count": payload["target_count"],
        "sha256": digest,
        "train": payload["split_summaries"]["train"]["totals"],
        "validation": payload["split_summaries"]["validation"]["totals"],
        "test_used": payload["test_used"],
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
