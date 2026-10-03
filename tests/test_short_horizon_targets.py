import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from train.build_short_horizon_targets import run


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _row(data_hash: str, deal: int, event: int, seat: int, hand: list[int],
         counts: list[int], leading: bool, cards: list[int]) -> dict:
    return {
        "schema": "njupt-decision-v3",
        "stage": "play",
        "deal_id": deal,
        "event_index": event,
        "match_id": f"m{deal}",
        "game_id": f"m{deal}__game0",
        "provenance": {"data_sha256": data_hash},
        "features": {
            "seat": seat, "team": seat % 2, "own_hand": hand,
            "remaining_counts": counts, "leading": leading,
        },
        "label": {"cards": cards, "claim_candidates": (
            [{"kind": "single"}] if cards else []
        )},
    }


class ShortHorizonTargetTests(unittest.TestCase):
    def test_public_only_and_never_opens_test_split(self):
        with tempfile.TemporaryDirectory() as name:
            source = Path(name) / "njupt"
            source.mkdir()
            digest = "a" * 64
            train = [
                _row(digest, 0, 1, 0, [0], [1, 1, 1, 1], True, [0]),
                _row(digest, 0, 2, 1, [1], [0, 1, 1, 1], True, []),
            ]
            validation = [
                _row(digest, 1, 1, 0, [2], [1, 1, 1, 1], True, [2]),
                _row(digest, 1, 2, 1, [3], [0, 1, 1, 1], True, []),
            ]
            for split, rows in (("train", train), ("validation", validation)):
                (source / f"{split}.jsonl").write_text(
                    "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
            # Deliberately malformed.  A target run must not open it.
            (source / "test.jsonl").write_text("this must not be read\n", encoding="utf-8")
            manifest = {
                "schema": "njupt-etl-manifest-v3",
                "decision_schema": "njupt-decision-v3",
                "output_files": {
                    f"{split}.jsonl": {"sha256": _sha(source / f"{split}.jsonl")}
                    for split in ("train", "validation")
                },
            }
            (source / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            output = Path(name) / "targets.json"
            payload = run(source, output)
            self.assertFalse(payload["test_used"])
            self.assertEqual(payload["target_count"], 4)
            first = payload["targets"][f"{digest[:16]}:d0:e1"]
            self.assertEqual(first["credit"], -1)
            self.assertEqual(first["lead_relation"], "opponent")
            self.assertEqual(payload["split_summaries"]["train"]["target_count"], 2)
            self.assertEqual(payload["split_summaries"]["validation"]["target_count"], 2)


if __name__ == "__main__":
    unittest.main()
