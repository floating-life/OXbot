"""Fail-closed checks for sparse continuation sidecar provenance."""
from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "train"))
from train_bc import alignment_digest, load_continuation_targets, sha256_file


class ContinuationAlignmentTests(unittest.TestCase):
    def _payload(self, directory: Path, targets: dict, expected: dict) -> tuple[Path, Path, dict]:
        data_manifest = directory / "manifest.json"
        data_manifest.write_text("{}\n", encoding="utf-8")
        source_manifest = {
            "source_manifest_sha256": "source-manifest",
            "source_splits_sha256": {"train": "train-source", "validation": "validation-source"},
        }
        payload = {
            "schema": "oxbot-candidate-continuation-targets-v1",
            "test_used": False,
            "splits_read": ["train", "validation"],
            "audit_only": False,
            "trainable": True,
            "prepared_manifest_sha256": sha256_file(data_manifest),
            "source_manifest_sha256": source_manifest["source_manifest_sha256"],
            "source_split_sha256": source_manifest["source_splits_sha256"],
            "target_coverage": "paired_rows_only",
            "prepared_alignment": {
                split: {"record_count": len(expected[split]), "sha256": alignment_digest(expected[split])}
                for split in ("train", "validation")
            },
            "targets": targets,
        }
        path = directory / "targets.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        return path, data_manifest, source_manifest

    @staticmethod
    def _target(count: int = 2) -> dict:
        return {"candidate_count": count, "labels": [1, -1] + [0] * (count - 2)}

    def test_sparse_targets_accept_only_prepared_ids_and_counts(self):
        expected = {"train": {"train-id": 2, "unused-id": 3}, "validation": {"validation-id": 2}}
        targets = {"train": {"train-id": self._target()}, "validation": {"validation-id": self._target()}}
        with tempfile.TemporaryDirectory() as temporary:
            path, manifest, source = self._payload(Path(temporary), targets, expected)
            loaded, _ = load_continuation_targets(path, manifest, source, expected)
            self.assertEqual(loaded["train"]["train-id"]["candidate_count"], 2)
            # Sparse sidecars intentionally omit unsupported/non-pair rows.
            self.assertNotIn("unused-id", loaded["train"])

    def test_extra_target_id_fails_closed(self):
        expected = {"train": {"prepared": 2}, "validation": {"validation": 2}}
        targets = {"train": {"not-prepared": self._target()}, "validation": {"validation": self._target()}}
        with tempfile.TemporaryDirectory() as temporary:
            path, manifest, source = self._payload(Path(temporary), targets, expected)
            with self.assertRaisesRegex(ValueError, "not in prepared"):
                load_continuation_targets(path, manifest, source, expected)

    def test_candidate_count_mismatch_fails_closed(self):
        expected = {"train": {"prepared": 3}, "validation": {"validation": 2}}
        targets = {"train": {"prepared": self._target(2)}, "validation": {"validation": self._target()}}
        with tempfile.TemporaryDirectory() as temporary:
            path, manifest, source = self._payload(Path(temporary), targets, expected)
            with self.assertRaisesRegex(ValueError, "candidate count mismatch"):
                load_continuation_targets(path, manifest, source, expected)


if __name__ == "__main__":
    unittest.main()
