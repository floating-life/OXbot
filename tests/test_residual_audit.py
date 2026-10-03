import json
import tempfile
import unittest
from pathlib import Path

from train.audit_residual_features import _valid_row, sample_rows


def _row(line: int) -> dict:
    return {
        "schema": "njupt-decision-v3",
        "stage": "play",
        "deal_id": line,
        "event_index": 1,
        "provenance": {"data_sha256": f"{line:064x}"},
        "features": {
            "own_hand": [line % 54],
            "level_label": "2",
            "leading": True,
        },
    }


class ResidualAuditTests(unittest.TestCase):
    def test_sampling_reads_only_the_requested_development_file(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            development = root / "train.jsonl"
            held_out = root / "test.jsonl"
            development.write_text("".join(json.dumps(_row(i)) + "\n" for i in range(8)), encoding="utf-8")
            held_out.write_text("this is deliberately not JSON\n", encoding="utf-8")
            rows, counts = sample_rows(development, limit=4, seed=7)
            self.assertEqual(len(rows), 4)
            self.assertEqual(counts["valid_play"], 8)
            self.assertTrue(all(_valid_row(record) for _, record in rows))


if __name__ == "__main__":
    unittest.main()
