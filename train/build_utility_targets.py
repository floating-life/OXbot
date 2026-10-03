"""Build train/validation outcome targets for advantage-weighted BC.

Only the two development splits are opened.  The target is deliberately a
coarse, game-level utility proxy: +1 for a decision made by the eventual
winning team and -1 for the other team.  It is never included in the model's
state/history features and the held-out test split is never touched.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path


def source_id(record: dict) -> str:
    provenance = record.get("provenance") or {}
    digest = provenance.get("data_sha256")
    deal = record.get("deal_id")
    event = record.get("event_index")
    if not isinstance(digest, str) or len(digest) != 64:
        raise ValueError("missing provenance data_sha256")
    if type(deal) is not int or type(event) is not int or deal < 0 or event < 0:
        raise ValueError("missing deal/event index")
    return f"{digest[:16]}:d{deal}:e{event}"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read(path: Path, split: str, targets: dict, seen: set[str], counts: Counter) -> None:
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("schema") != "njupt-decision-v3":
                raise ValueError(f"{path}:{line_number}: unexpected decision schema")
            if record.get("stage") != "play":
                counts["non_play"] += 1
                continue
            features = record.get("features") or {}
            single = (record.get("result") or {}).get("single_game") or {}
            seat, team, winner_seat = features.get("seat"), features.get("team"), single.get("winner_seat")
            if not all(type(x) is int for x in (seat, team, winner_seat)):
                counts["invalid"] += 1
                continue
            if team != seat % 2 or not 0 <= winner_seat < 4:
                counts["team_invariant_failure"] += 1
                continue
            key = source_id(record)
            if key in seen:
                raise ValueError(f"duplicate target key {key} in {split} at line {line_number}")
            seen.add(key)
            win = int(team == winner_seat % 2)
            targets[key] = win
            counts["rows"] += 1
            counts["wins"] += win
            counts["losses"] += 1 - win


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=Path("data/processed/njupt"))
    parser.add_argument("--output", type=Path, default=Path("reports/action_utility_targets_v2.json"))
    args = parser.parse_args()
    source = args.source.resolve()
    manifest_path = source / "manifest.json"
    if not manifest_path.is_file():
        raise ValueError(f"source manifest does not exist: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != "njupt-etl-manifest-v3":
        raise ValueError("unexpected NJUPT source manifest schema")
    if manifest.get("decision_schema") != "njupt-decision-v3":
        raise ValueError("source manifest decision schema is incompatible")
    manifest_outputs = manifest.get("output_files")
    if not isinstance(manifest_outputs, dict):
        raise ValueError("source manifest has no output_files metadata")
    manifest_sha256 = sha256_file(manifest_path)
    source_split_sha256 = {}
    split_paths = {}
    for split in ("train", "validation"):
        path = source / f"{split}.jsonl"
        if not path.is_file():
            raise ValueError(f"required development split does not exist: {path}")
        metadata = manifest_outputs.get(f"{split}.jsonl")
        expected = metadata.get("sha256") if isinstance(metadata, dict) else None
        if not isinstance(expected, str) or len(expected) != 64:
            raise ValueError(f"manifest has no SHA256 for {split}.jsonl")
        actual = sha256_file(path)
        if actual != expected:
            raise ValueError(f"{split}.jsonl SHA256 differs from source manifest")
        source_split_sha256[split] = actual
        split_paths[split] = path
    targets = {}
    seen = set()
    split_counts = {}
    for split in ("train", "validation"):
        counts = Counter()
        read(split_paths[split], split, targets, seen, counts)
        split_counts[split] = dict(counts)
    payload = {
        "schema": "oxbot-action-utility-targets-v1",
        "status": "complete",
        "source": str(source),
        "source_manifest_sha256": manifest_sha256,
        "source_split_sha256": source_split_sha256,
        "source_manifest_schema": manifest.get("schema"),
        "decision_schema": manifest.get("decision_schema"),
        "splits_read": ["train", "validation"],
        "test_used": False,
        "target_definition": "team_win = (features.team == result.single_game.winner_seat % 2)",
        "split_counts": split_counts,
        "target_count": len(targets),
        "targets": targets,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temp = args.output.with_suffix(args.output.suffix + ".tmp")
    temp.write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n", encoding="utf-8")
    temp.replace(args.output)
    digest = hashlib.sha256(args.output.read_bytes()).hexdigest()
    print(json.dumps({"status": payload["status"], "output": str(args.output),
                      "target_count": len(targets), "sha256": digest}, ensure_ascii=False))


if __name__ == "__main__":
    main()
