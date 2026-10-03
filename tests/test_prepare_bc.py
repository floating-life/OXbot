"""BC shard contract checks; no raw pickle or hidden state is used."""
from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "train"))
from prepare_bc import (DECISION_SCHEMA, DEFAULT_PROBE, PreparationError, ShardWriter, SkipRecord, encode_record,
                        ensure_safe_output, observation_from_record, run, select_candidates)


def record():
    return {
        "schema": DECISION_SCHEMA, "stage": "play", "match_id": "fixture-match", "game_id": "fixture-game", "deal_id": 0,
        "event_index": 12, "provenance": {"data_sha256": "a" * 64, "source_data": "fixture.data"},
        "features": {"own_hand": [4, 8], "seat": 0, "level_label": "2", "leading": True,
                     "remaining_counts": [2, 3, 3, 3], "played_face_counts": [0] * 54,
                     "history": [], "history_total": 0, "tribute": 0, "resist": False,
                     "last_player": -1, "last_cards": [], "last_claim": None},
        "label": {"cards": [4]},
        "result": {"winner": 3},
        "opponent_hands": [[53], [52], [107]],
    }


class FakeProbe:
    def __init__(self):
        self.calls = []

    def call(self, **request):
        self.calls.append(request)
        if request["command"] == "classify":
            return {"kind": "single", "key": 0, "secondary": -1}
        assert request["command"] == "generate"
        return {
            "moves": [[[4], [0]], [[4], [48]], [[8], [8]]],
            "types": [{"kind": "single", "key": 0, "secondary": -1},
                      {"kind": "single", "key": 12, "secondary": -1},
                      {"kind": "single", "key": 2, "secondary": -1}],
        }


class HiddenFieldsPoisoned(dict):
    def get(self, key, default=None):
        if key in ("result", "opponent_hands"):
            raise AssertionError("attempted to read hidden or future information")
        return super().get(key, default)

    def __getitem__(self, key):
        if key in ("result", "opponent_hands"):
            raise AssertionError("attempted to read hidden or future information")
        return super().__getitem__(key)


class PrepareBCTests(unittest.TestCase):
    def test_multi_positive_and_information_set_whitelist(self):
        poisoned = HiddenFieldsPoisoned(record())
        encoded, stats = encode_record(poisoned, "train", FakeProbe(), seed=11, max_negatives=0)
        self.assertEqual(encoded["positives"].tolist(), [True, True])
        self.assertEqual(encoded["actions"].shape, (2, 128))
        self.assertEqual(stats["all_candidates"], 3)
        self.assertEqual(stats["positive_candidates"], 2)
        self.assertEqual(encoded["length"], 2)
        self.assertEqual(encoded["tokens"].tolist()[:3], [1, 118, 0])
        self.assertEqual(encoded["state"].dtype, np.float32)

    def test_evaluation_retains_every_candidate(self):
        for split in ("validation", "test"):
            encoded, _ = encode_record(record(), split, FakeProbe(), seed=11, max_negatives=0)
            self.assertEqual(encoded["positives"].tolist(), [True, True, False])
            self.assertEqual(encoded["actions"].shape[0], 3)

    def test_deterministic_sampling_never_drops_positives(self):
        positive = [0, 3, 200, 499]
        first = select_candidates(500, positive, "train", 123, "same-id", 128)
        self.assertEqual(first, select_candidates(500, positive, "train", 123, "same-id", 128))
        self.assertEqual(len(first), 132)
        self.assertTrue(set(positive).issubset(first))
        self.assertNotEqual(first, select_candidates(500, positive, "train", 123, "other-id", 128))

    def test_uncertain_previous_is_countable_skip(self):
        item = record()
        item["features"]["leading"] = False
        item["features"]["last_cards"] = [0]
        item["features"]["last_claim"] = None
        with self.assertRaises(SkipRecord) as error:
            observation_from_record(item)
        self.assertEqual(error.exception.reason, "uncertain_previous")

    def test_unknown_demonstration_is_countable_skip(self):
        item = record()
        item["label"]["cards"] = [12]
        with self.assertRaises(SkipRecord) as error:
            encode_record(item, "train", FakeProbe(), seed=11, max_negatives=128)
        self.assertEqual(error.exception.reason, "demonstration_not_legal")

    def test_history_is_windowed_explicitly_not_silently(self):
        item = record()
        item["features"]["history"] = [{"player": i % 4, "action": [], "claim": []} for i in range(128)]
        item["features"]["history_total"] = 180
        encoded, stats = encode_record(item, "test", FakeProbe(), seed=11, max_negatives=128)
        self.assertEqual(encoded["length"], 256)
        self.assertEqual(stats["token_window_truncated"], 1)
        self.assertEqual(stats["etl_history_window_truncated"], 1)
        item["features"]["history"].append({"player": 0, "action": [], "claim": []})
        with self.assertRaises(SkipRecord):
            observation_from_record(item)

    def test_shard_ragged_candidates_preserve_labels_and_offsets(self):
        first, _ = encode_record(record(), "train", FakeProbe(), seed=11, max_negatives=0)
        second_record = record()
        second_record["event_index"] = 13
        second, _ = encode_record(second_record, "test", FakeProbe(), seed=11, max_negatives=128)
        with tempfile.TemporaryDirectory() as temporary:
            writer = ShardWriter(Path(temporary), "train", 2)
            writer.add(first)
            writer.add(second)
            self.assertEqual(len(writer.files), 1)
            path = Path(temporary) / writer.files[0]["path"]
            with np.load(path, allow_pickle=False) as data:
                self.assertEqual(data["offsets"].tolist(), [0, 2, 5])
                self.assertEqual(data["positives"].tolist(), [True, True, True, True, False])
                self.assertEqual(data["state"].shape, (2, 128))
                self.assertEqual(data["tokens"].dtype, np.uint8)
                self.assertEqual(data["lengths"].dtype, np.int16)
                self.assertEqual(len(data["ids"]), 2)

    def test_source_and_output_cannot_overlap(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            source = base / "source"
            source.mkdir()
            for output in (source, source / "child", base):
                with self.assertRaises(PreparationError):
                    ensure_safe_output(source, output)
            self.assertEqual(ensure_safe_output(source, base / "output")[1], (base / "output").resolve())

    @unittest.skipUnless(DEFAULT_PROBE.exists(), "C++ rule probe not built")
    def test_real_probe_end_to_end_manifest_and_arrays(self):
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "input"
            source.mkdir()
            (source / "manifest.json").write_text(json.dumps({"decision_schema": DECISION_SCHEMA}), encoding="utf-8")
            for index, split in enumerate(("train", "validation", "test")):
                item = record()
                item["match_id"] = "fixture-" + split
                item["provenance"]["data_sha256"] = str(index + 1) * 64
                (source / (split + ".jsonl")).write_text(json.dumps(item) + "\n", encoding="utf-8")
            target = Path(temporary) / "output"
            manifest = run(source, target, DEFAULT_PROBE, workers=1, max_negatives=1)
            self.assertEqual(manifest["status"], "complete")
            self.assertTrue((target / "manifest.json").exists())
            for split in ("train", "validation", "test"):
                info = manifest["splits"][split]
                self.assertEqual(info["counts"]["accepted_records"], 1)
                self.assertEqual(len(info["shards"]), 1)
                self.assertEqual(len(info["shards"][0]["sha256"]), 64)
                with np.load(target / info["shards"][0]["path"], allow_pickle=False) as data:
                    self.assertEqual(data["state"].shape, (1, 128))
                    self.assertTrue(data["positives"].any())
                    self.assertEqual(int(data["offsets"][-1]), len(data["actions"]))


if __name__ == "__main__":
    unittest.main()
