"""Read-only audit of row-level kind/size labels in prepared development shards.

The proposed factorized loss would aggregate candidate scores by semantic kind
and action size.  This script checks whether the demonstrated positive set has
one unambiguous kind/size and reports train/validation distribution shift.  It
opens only prepared ``train`` and ``validation`` shards; no test path is
enumerated or opened and no checkpoint is loaded.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


SPLITS = ("train", "validation")
KINDS = ("pass", "invalid", "single", "pair", "three", "straight", "set",
         "three_straight", "triple_pairs", "bomb", "straight_flush", "rocket")
SCHEMA = "oxbot-factorized-label-audit-v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def audit_split(prepared: Path, split: str) -> dict[str, Any]:
    files = sorted((prepared / split).glob("*.npz"))
    if not files:
        raise ValueError(f"no prepared {split} shards")
    rows = positives = 0
    kind_rows: Counter[str] = Counter()
    size_rows: Counter[str] = Counter()
    candidate_kinds: Counter[str] = Counter()
    candidate_sizes: Counter[str] = Counter()
    ambiguity: Counter[str] = Counter()
    for path in files:
        with np.load(path, allow_pickle=False) as data:
            for key in ("ids", "actions", "offsets", "positives"):
                if key not in data.files:
                    raise ValueError(f"{path} lacks {key}")
            ids = data["ids"]
            actions, offsets, positive = data["actions"], data["offsets"], data["positives"]
            if len(offsets) != len(ids) + 1 or int(offsets[-1]) != len(actions):
                raise ValueError(f"{path} has malformed offsets")
            for index in range(len(ids)):
                begin, end = int(offsets[index]), int(offsets[index + 1])
                row_actions = actions[begin:end]
                row_positive = positive[begin:end]
                kinds = np.argmax(row_actions[:, 108:120], axis=1)
                sizes = np.rint(row_actions[:, 120] * 10).astype(np.int64)
                candidate_kinds.update(KINDS[int(value)] for value in kinds)
                candidate_sizes.update(str(int(value)) for value in sizes)
                pos_kinds = {KINDS[int(value)] for value in kinds[row_positive]}
                pos_sizes = {str(int(value)) for value in sizes[row_positive]}
                rows += 1
                positives += int(row_positive.sum())
                ambiguity["kind_single" if len(pos_kinds) == 1 else "kind_multi"] += 1
                ambiguity["size_single" if len(pos_sizes) == 1 else "size_multi"] += 1
                kind_rows.update(pos_kinds)
                size_rows.update(pos_sizes)
    return {
        "rows": rows,
        "positive_candidates": positives,
        "kind_rows": dict(kind_rows),
        "size_rows": dict(size_rows),
        "candidate_kinds": dict(candidate_kinds),
        "candidate_sizes": dict(candidate_sizes),
        "ambiguity": dict(ambiguity),
        "kind_row_rate": {key: value / rows for key, value in kind_rows.items()},
        "size_row_rate": {key: value / rows for key, value in size_rows.items()},
    }


def run(prepared: Path, output: Path) -> dict[str, Any]:
    prepared = prepared.resolve()
    results = {split: audit_split(prepared, split) for split in SPLITS}
    payload = {
        "schema": SCHEMA,
        "status": "complete",
        "prepared": str(prepared),
        "prepared_manifest_sha256": sha256_file(prepared / "manifest.json"),
        "splits_read": list(SPLITS),
        "test_used": False,
        "splits": results,
    }
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temp = output.with_suffix(output.suffix + ".tmp")
    temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temp.replace(output)
    payload["output_sha256"] = sha256_file(output)
    return payload


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, default=root / "data" / "processed" / "bc-v2-full")
    parser.add_argument("--output", type=Path, default=root / "reports" / "factorized-label-audit.json")
    args = parser.parse_args()
    payload = run(args.prepared, args.output)
    print(json.dumps({"status": payload["status"], "output": str(args.output.resolve()),
                      "sha256": payload["output_sha256"], "test_used": payload["test_used"],
                      "splits": {split: payload["splits"][split]["ambiguity"] for split in SPLITS}},
               ensure_ascii=False))


if __name__ == "__main__":
    main()
