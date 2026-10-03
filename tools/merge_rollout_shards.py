"""Merge audited self-play rollout JSONL shards without hiding provenance.

Each input must have the sidecar report emitted by
``tools/generate_selfplay.py`` (same basename with ``.json`` suffix).  The
merger checks report/output SHA256, schema/contracts, non-overlapping rollout
identities, and counter totals before publishing one atomic JSONL plus an
aggregate report.  It never reads a held-out training split; ``test_used`` is
carried as a fail-closed field and must be false on every shard.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any


SCHEMA = "oxbot-selfplay-candidate-v1"
FEATURE_VERSION = "oxbot-observation-v1"
RULES_CONTRACT = "botzone-corrected-fa63589d-v1"
OFFICIAL_ROLLOUT_RULES_CONTRACT = "botzone-official-910cba94-v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", dir=path.parent, prefix=path.name + ".", suffix=".tmp", delete=False
        ) as stream:
            temporary = Path(stream.name)
            json.dump(payload, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def row_key(row: dict[str, Any]) -> tuple[int, int, int, int]:
    values = (row.get("game_seed"), row.get("game_index"), row.get("seat"), row.get("event_index"))
    if any(type(value) is not int for value in values):
        raise ValueError("row identity is missing or not an integer")
    return values


def read_shard(path: Path, identities: set[tuple[int, int, int, int]], report_path: Path | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
    path = path.resolve()
    if not path.is_file():
        raise ValueError(f"shard does not exist: {path}")
    report_path = (report_path or path.with_suffix(".json")).resolve()
    if not report_path.is_file():
        raise ValueError(f"shard sidecar report is missing: {report_path}")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("schema") != SCHEMA or report.get("status") != "complete":
        raise ValueError(f"{report_path}: report schema/status is incompatible")
    if report.get("feature_version") != FEATURE_VERSION or report.get("rules_contract") != RULES_CONTRACT:
        raise ValueError(f"{report_path}: feature/rules contract is incompatible")
    rollout_contract = report.get("rollout_rules_contract")
    oracle_sha256 = report.get("oracle_sha256")
    if not isinstance(rollout_contract, str) or not isinstance(oracle_sha256, str):
        raise ValueError(f"{report_path}: rollout judge provenance is missing")
    if report.get("test_used") is not False:
        raise ValueError(f"{report_path}: test_used must be false")
    expected_sha = report.get("output_sha256")
    actual_sha = sha256_file(path)
    if expected_sha != actual_sha:
        raise ValueError(f"{path}: output SHA256 differs from sidecar report")

    rows = 0
    labels = 0
    games: set[tuple[int, int]] = set()
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict) or row.get("schema") != SCHEMA or row.get("stage") != "play":
                raise ValueError(f"{path}:{line_number}: invalid rollout row")
            if row.get("rollout_rules_contract") != rollout_contract or row.get("oracle_sha256") != oracle_sha256:
                raise ValueError(f"{path}:{line_number}: row/report rollout provenance differs")
            key = row_key(row)
            if key in identities:
                raise ValueError(f"duplicate rollout identity across shards: {key}")
            identities.add(key)
            games.add((key[0], key[1]))
            candidate_q = row.get("candidate_q")
            if not isinstance(candidate_q, dict) or not candidate_q:
                raise ValueError(f"{path}:{line_number}: candidate_q is missing/empty")
            rows += 1
            labels += len(candidate_q)
    counters = report.get("counters") or {}
    if counters.get("play_records") != rows or counters.get("counterfactual_labels") != labels:
        raise ValueError(f"{path}: row/label counters differ from report")
    summary = {
        "path": str(path),
        "report": str(report_path),
        "sha256": actual_sha,
        "rows": rows,
        "labels": labels,
        "games": len(games),
        "seed_values": sorted(seed for seed, _ in games),
        "policy": report.get("policy"),
        "model": report.get("model"),
        "model_sha256": report.get("model_sha256"),
        "rollout_rules_contract": rollout_contract,
        "oracle_sha256": oracle_sha256,
    }
    return summary, report


def merge(inputs: list[Path], output: Path, report_path: Path, reports: list[Path] | None = None) -> dict[str, Any]:
    if not inputs:
        raise ValueError("at least one shard is required")
    resolved_inputs = [path.resolve() for path in inputs]
    if len(set(resolved_inputs)) != len(resolved_inputs):
        raise ValueError("duplicate shard path")
    if output.resolve() in resolved_inputs:
        raise ValueError("output must differ from every shard")
    resolved_reports = None if reports is None else [path.resolve() for path in reports]
    if resolved_reports is not None and len(resolved_reports) != len(resolved_inputs):
        raise ValueError("--reports must have exactly one path per --inputs shard")
    identities: set[tuple[int, int, int, int]] = set()
    shard_summaries: list[dict[str, Any]] = []
    shard_reports: list[dict[str, Any]] = []
    for index, path in enumerate(resolved_inputs):
        summary, report = read_shard(path, identities, None if resolved_reports is None else resolved_reports[index])
        shard_summaries.append(summary)
        shard_reports.append(report)
    baseline = shard_reports[0]
    for report in shard_reports[1:]:
        for field in ("feature_version", "rules_contract", "rollout_rules_contract", "oracle_sha256",
                      "policy", "model", "model_sha256", "selection_strategy"):
            if report.get(field) != baseline.get(field):
                raise ValueError(f"shard reports disagree on {field}")

    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "wb", dir=output.parent, prefix=output.name + ".", suffix=".tmp", delete=False
        ) as stream:
            temporary = Path(stream.name)
            for path in resolved_inputs:
                with path.open("rb") as source:
                    for block in iter(lambda: source.read(1 << 20), b""):
                        stream.write(block)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, output)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()

    total_rows = sum(item["rows"] for item in shard_summaries)
    total_labels = sum(item["labels"] for item in shard_summaries)
    seed_values = sorted({seed for item in shard_summaries for seed in item["seed_values"]})
    aggregate = {
        "schema": SCHEMA,
        "status": "complete",
        "aggregate": True,
        "feature_version": baseline.get("feature_version"),
        "rules_contract": baseline.get("rules_contract"),
        "rollout_rules_contract": baseline.get("rollout_rules_contract"),
        "oracle_sha256": baseline.get("oracle_sha256"),
        "oracle_sha256_values": sorted({report.get("oracle_sha256") for report in shard_reports}),
        "probe_sha256": baseline.get("probe_sha256"),
        "probe_sha256_values": sorted({report.get("probe_sha256") for report in shard_reports}),
        "policy": baseline.get("policy"),
        "model": baseline.get("model"),
        "model_sha256": baseline.get("model_sha256"),
        "model_team": baseline.get("model_team"),
        "selection_strategy": baseline.get("selection_strategy"),
        "test_used": False,
        "source_shards": shard_summaries,
        "seed_values": seed_values,
        "games": len({(key[0], key[1]) for key in identities}),
        "play_records": total_rows,
        "counterfactual_labels": total_labels,
        "output": str(output),
        "output_sha256": sha256_file(output),
        "identity_key": "(game_seed,game_index,seat,event_index)",
    }
    atomic_json(report_path.resolve(), aggregate)
    return aggregate


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", nargs="+", type=Path, required=True)
    parser.add_argument("--reports", nargs="*", type=Path,
                        help="optional sidecar reports in the same order; defaults to each input's .json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    try:
        aggregate = merge(args.inputs, args.output, args.report, args.reports if args.reports else None)
    except Exception as exc:
        print(json.dumps({"status": "failed", "error": str(exc)}, ensure_ascii=False))
        return 1
    print(json.dumps({"status": aggregate["status"], "output": aggregate["output"],
                      "report": str(args.report.resolve()), "games": aggregate["games"],
                      "play_records": aggregate["play_records"],
                      "counterfactual_labels": aggregate["counterfactual_labels"],
                      "output_sha256": aggregate["output_sha256"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
