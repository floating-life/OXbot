"""Audit outcome/action correlations using only NJUPT train/validation.

This is a diagnostic for an advantage-weighted behavioural-cloning proposal.
It deliberately never opens the held-out ``test.jsonl``.  The final game
result is not an observation feature and is only used to estimate whether a
demonstrated action came from the winning or losing side.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path


def action_kind(record: dict) -> str:
    label = record.get("label") or {}
    cards = label.get("cards")
    if not isinstance(cards, list):
        return "invalid"
    if not cards:
        return "pass"
    candidates = label.get("claim_candidates") or []
    if candidates and isinstance(candidates[0], dict):
        kind = candidates[0].get("kind")
        if isinstance(kind, str):
            return kind
    # A small number of old records have no reconstructed claim metadata.
    return str(label.get("kind") or "unknown")


def read_split(path: Path, split: str, by_kind: dict, by_size: dict,
               summary: dict) -> None:
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("stage") != "play":
                continue
            features = record.get("features") or {}
            result = record.get("result") or {}
            single = result.get("single_game") or {}
            seat = features.get("seat")
            team = features.get("team")
            winner_seat = single.get("winner_seat")
            if not all(isinstance(x, int) for x in (seat, team, winner_seat)):
                summary["invalid_rows"] += 1
                continue
            # NJUPT's team assignment is seat parity; check the supplied
            # feature as an audit invariant rather than trusting it blindly.
            if team != seat % 2 or winner_seat < 0 or winner_seat > 3:
                summary["team_invariant_failures"] += 1
                continue
            win = int(team == winner_seat % 2)
            kind = action_kind(record)
            cards = record.get("label", {}).get("cards") or []
            size = len(cards)
            summary["rows"] += 1
            summary["wins"] += win
            summary["losses"] += 1 - win
            by_kind[kind]["rows"] += 1
            by_kind[kind]["wins"] += win
            by_kind[kind]["losses"] += 1 - win
            by_size[str(size)]["rows"] += 1
            by_size[str(size)]["wins"] += win
            by_size[str(size)]["losses"] += 1 - win


def finalize(table: dict) -> dict:
    out = {}
    for key in sorted(table, key=lambda x: (len(x), x)):
        row = dict(table[key])
        row["win_rate"] = row["wins"] / max(1, row["rows"])
        out[key] = row
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=Path("data/processed/njupt"))
    parser.add_argument("--output", type=Path, default=Path("reports/action_utility_audit.json"))
    args = parser.parse_args()
    source = args.source.resolve()
    result = {
        "schema": "oxbot-action-utility-audit-v1",
        "status": "running",
        "source": str(source),
        "test_used": False,
        "splits": {},
    }
    for split in ("train", "validation"):
        summary = Counter()
        by_kind = defaultdict(Counter)
        by_size = defaultdict(Counter)
        read_split(source / f"{split}.jsonl", split, by_kind, by_size, summary)
        result["splits"][split] = {
            "summary": dict(summary),
            "by_kind": finalize(by_kind),
            "by_size": finalize(by_size),
        }
    result["status"] = "complete"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temp = args.output.with_suffix(args.output.suffix + ".tmp")
    temp.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temp.replace(args.output)
    print(json.dumps({"status": result["status"], "output": str(args.output),
                      "train_rows": result["splits"]["train"]["summary"].get("rows", 0),
                      "validation_rows": result["splits"]["validation"]["summary"].get("rows", 0)},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
