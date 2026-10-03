"""Regression checks for the read-only NUPT import and information boundary."""
from __future__ import annotations

from collections import Counter
import copy
import json
from pathlib import Path
import pickle
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from train.etl_njupt import (
    DEFAULT_MAX_EVENTS, DEFAULT_MAX_FILE_BYTES, ETLError, MatchRecord, SchemaError,
    UnsafePickleError, _candidate_claims, _find_result_marker, _parse_game,
    assign_splits, iter_safe_pickle, run_etl, sha256_file, source_id_to_face,
)


def write_stream(path, events):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as stream:
        for event in events:
            pickle.dump(event, stream, protocol=3)


def initial_events():
    deck = list(range(2, 56)) * 2
    return [("R", -1, 2), ("R", 0, 2), ("R", 1, 2)] + [
        ("I", seat, deck[seat * 27 : (seat + 1) * 27]) for seat in range(4)
    ]


def victory():
    return ("V", ("Alpha队", 0, 2, 14, 4, 14))


class SafePickleTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.path = self.root / "input.data"

    def tearDown(self):
        self.temp.cleanup()

    def test_concatenated_primitive_events(self):
        events = [("R", -1, 2), ("P", 0, [3, 16]), victory()]
        write_stream(self.path, events)
        self.assertEqual(list(iter_safe_pickle(self.path)), events)

    def test_globals_and_persistent_objects_are_rejected(self):
        self.path.write_bytes(pickle.dumps(eval, protocol=3))
        with self.assertRaisesRegex(UnsafePickleError, "global"):
            list(iter_safe_pickle(self.path))
        self.path.write_bytes(b"Pidentifier\n.")
        with self.assertRaisesRegex(UnsafePickleError, "persistent"):
            list(iter_safe_pickle(self.path))

    def test_resource_limits(self):
        write_stream(self.path, [("R", -1, 2)] * 3)
        with self.assertRaisesRegex(UnsafePickleError, "bytes"):
            list(iter_safe_pickle(self.path, max_file_bytes=1))
        with self.assertRaisesRegex(UnsafePickleError, "event count"):
            list(iter_safe_pickle(self.path, max_events=2))
        write_stream(self.path, [[[[[1]]]]])
        with self.assertRaisesRegex(UnsafePickleError, "nesting"):
            list(iter_safe_pickle(self.path, max_depth=2))
        write_stream(self.path, [[1, 2, 3]])
        with self.assertRaisesRegex(UnsafePickleError, "sequence"):
            list(iter_safe_pickle(self.path, max_sequence=2))
        with self.assertRaisesRegex(UnsafePickleError, "value count"):
            list(iter_safe_pickle(self.path, max_nodes=2))
        write_stream(self.path, ["long"])
        with self.assertRaisesRegex(UnsafePickleError, "string"):
            list(iter_safe_pickle(self.path, max_string=2))

    def test_cycles_and_nonprimitive_values_are_rejected(self):
        recursive = []
        recursive.append(recursive)
        for value in (recursive, {"tag": "R"}, 3.14, b"bytes", 2**65):
            with self.subTest(value_type=type(value).__name__):
                write_stream(self.path, [value])
                with self.assertRaises(UnsafePickleError):
                    list(iter_safe_pickle(self.path))

    def test_truncated_final_pickle_is_not_silent_eof(self):
        write_stream(self.path, [("R", -1, 2), ("P", 0, [2, 3])])
        self.path.write_bytes(self.path.read_bytes()[:-2])
        with self.assertRaises(UnsafePickleError):
            list(iter_safe_pickle(self.path))


class ReplayTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.match = MatchRecord("Alpha vs Beta", "", None, self.root)

    def tearDown(self):
        self.temp.cleanup()

    def parse(self, events, name="Alpha_Beta_0.data"):
        path = self.root / name
        write_stream(path, events)
        return _parse_game(self.match, 0, path, max_file_bytes=DEFAULT_MAX_FILE_BYTES, max_events=DEFAULT_MAX_EVENTS)

    def test_source_mapping_uses_suit_blocks_and_heart_block_one(self):
        self.assertEqual(source_id_to_face(2), 5)    # diamond 2
        self.assertEqual(source_id_to_face(15), 4)   # heart 2 / level-2 wildcard
        self.assertEqual(source_id_to_face(28), 6)   # spade 2
        self.assertEqual(source_id_to_face(41), 7)   # club 2
        self.assertEqual(source_id_to_face(14), 1)   # diamond A
        self.assertEqual(source_id_to_face(27), 0)   # heart A
        self.assertEqual([source_id_to_face(c) for c in (54, 55)], [52, 53])
        self.assertEqual(set(source_id_to_face(c) for c in range(2, 56)), set(range(54)))
        for invalid in (1, 56, True, "2"):
            with self.assertRaises(SchemaError):
                source_id_to_face(invalid)

    def test_decisions_are_before_action_and_transfers_do_not_count_as_played(self):
        events = initial_events() + [
            ("T", 0, 1, 28), ("B", 1, 0, 41),
            ("P", 0, [3]), ("P", 1, [31]), ("P", 2, 1),
            ("P", 3, 1), ("P", 0, 1), ("P", 1, [32]), victory(),
        ]
        parsed = self.parse(events)
        tribute, returning, first, second = parsed.decisions[:4]
        self.assertEqual(tribute["features"]["remaining_counts"], [27] * 4)
        self.assertEqual(returning["features"]["remaining_counts"], [26, 28, 27, 27])
        self.assertEqual(first["features"]["remaining_counts"], [27] * 4)
        self.assertEqual(first["features"]["played_face_counts"], [0] * 54)
        self.assertEqual(second["features"]["remaining_counts"], [26, 27, 27, 27])
        self.assertEqual(sum(second["features"]["played_face_counts"]), 1)
        self.assertEqual(first["features"]["tribute"], 1)
        self.assertTrue(first["features"]["leading"])
        self.assertFalse(second["features"]["leading"])
        self.assertEqual(second["features"]["last_player"], 0)
        self.assertEqual(second["features"]["last_claim"], [source_id_to_face(3)])
        self.assertTrue(parsed.decisions[-1]["features"]["leading"])
        self.assertEqual(parsed.decisions[-1]["features"]["last_cards"], [])
        self.assertEqual(sum("return_current_level_conflicts_oracle" in row["quality_flags"] for row in parsed.decisions), 1)
        for decision in parsed.decisions:
            self.assertFalse(Counter(decision["label"]["cards"]) - Counter(decision["features"]["own_hand"]))

    def test_opponent_allocation_cannot_change_decision_features(self):
        original = initial_events()
        changed = copy.deepcopy(original)
        # Move one duplicate face from a lower-numbered opponent to a higher-
        # numbered opponent. Global replay IDs change; seat 2's observations do not.
        hand0, hand3 = changed[3][2], changed[6][2]
        hand0[hand0.index(3)], hand3[hand3.index(29)] = 29, 3
        a = self.parse(original + [("P", 2, [3]), victory()], "original_0.data")
        b = self.parse(changed + [("P", 2, [3]), victory()], "changed_0.data")
        self.assertEqual(a.decisions[0]["features"], b.decisions[0]["features"])
        self.assertEqual(a.decisions[0]["label"], b.decisions[0]["label"])
        self.assertNotEqual(a.events[-3]["cards"], b.events[-3]["cards"])

    def test_catch_wind_resets_previous_play_and_preserves_done_order(self):
        events = initial_events() + [
            ("P", 0, list(range(2, 29))), ("P", 1, 1),
            ("P", 2, 1), ("P", 3, 1), ("C",), ("P", 2, [3]), victory(),
        ]
        parsed = self.parse(events)
        resumed = parsed.decisions[-1]["features"]
        self.assertTrue(resumed["leading"])
        self.assertEqual(resumed["done_order"], [0])
        self.assertEqual(resumed["remaining_counts"], [0, 27, 27, 27])
        self.assertEqual(resumed["last_player"], -1)

    def test_missing_wildcard_claim_stays_unknown(self):
        parsed = self.parse(initial_events() + [("P", 0, [15]), ("P", 1, [54]), victory()])
        self.assertEqual(parsed.decisions[0]["label"]["claim_status"], "pending_reconstruction")
        self.assertIsNone(parsed.decisions[0]["label"]["claim"])
        self.assertIsNone(parsed.decisions[1]["features"]["last_claim"])
        self.assertIsNone(parsed.decisions[1]["features"]["history"][-1]["claim"])

    def test_exchange_context_uses_public_prior_finish_and_prefix_only(self):
        previous = initial_events() + [
            ("P", 0, list(range(2, 29))), ("P", 1, 1),
            ("P", 2, list(range(2, 29))),
        ]
        next_deal = [("R", -1, 5), ("R", 0, 5), ("R", 1, 2)] + initial_events()[3:]
        resisted = self.parse(previous + next_deal + [("P", 3, [29]), victory()])
        feature = resisted.decisions[-1]["features"]
        self.assertEqual((feature["tribute"], feature["resist"], feature["exchange_context_known"]), (2, True, True))
        normal = self.parse(previous + next_deal + [
            ("T", 1, 0, 54), ("T", 3, 2, 55),
            ("B", 0, 1, 3), ("B", 2, 3, 4), ("P", 1, [30]), victory(),
        ])
        feature = normal.decisions[-1]["features"]
        self.assertEqual((feature["tribute"], feature["resist"], feature["exchange_context_known"]), (2, False, True))
        self.assertFalse(normal.decisions[0]["features"]["exchange_context_known"])

    def test_history_window_exceeds_short_pass_heavy_64_event_tail(self):
        events = initial_events()
        for card in range(2, 28):
            events.extend([("P", 0, [card]), ("P", 1, 1), ("P", 2, 1), ("P", 3, 1)])
        parsed = self.parse(events + [victory()])
        feature = parsed.decisions[-1]["features"]
        self.assertEqual(feature["history_total"], 103)
        self.assertEqual(len(feature["history"]), 103)
        self.assertEqual(len(feature["public_history"]), 64)

    def test_result_marker_final_score_and_post_v_actions(self):
        path = self.root / "Alpha_Beta_0.data"
        marker = path.with_name(path.name + "_14_4")
        marker.touch()
        parsed = self.parse(initial_events() + [("P", 0, [3]), victory(), ("F", [2, 1, 2, 1]), ("P", 0, [55])])
        self.assertEqual(_find_result_marker(path), (14, 4))
        self.assertEqual(len(parsed.decisions), 1)
        self.assertEqual(parsed.result["single_game"]["marker_levels"], [14, 4])
        self.assertEqual(parsed.result["series_game_score"], ["F", [2, 1, 2, 1]])
        self.assertEqual([e[0] for e in parsed.trailing_events], ["F", "P"])

    def test_invalid_ownership_and_turn_order_are_rejected(self):
        for actions in ([('P', 0, [54])], [('P', 0, [3]), ('P', 2, [3])], [('P', 0, 1)]):
            with self.subTest(actions=actions), self.assertRaises(SchemaError):
                self.parse(initial_events() + actions + [victory()])

    def test_ace_high_natural_claim_is_reconstructed(self):
        cards = [source_id_to_face(c) for c in (10, 11, 12, 13, 14)]
        candidates, status = _candidate_claims(cards, 2)
        self.assertEqual(status, "natural_reconstructed")
        self.assertEqual(candidates[0]["kind"], "straight_flush")
        self.assertEqual(candidates[0]["key"], 9)


class PipelineTests(unittest.TestCase):
    def test_deterministic_match_split_is_32_7_7(self):
        records = [MatchRecord(f"team{i:02} vs other", "", None, Path("unused")) for i in range(46)]
        forward = assign_splits(records)
        self.assertEqual(forward, assign_splits(list(reversed(records))))
        self.assertEqual(Counter(forward.values()), {"train": 32, "validation": 7, "test": 7})

    def test_source_is_read_only_and_incomplete_duplicate_is_quarantined(self):
        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp) / "source"
            match = source / "已解压" / "Alpha vs Beta"
            complete, truncated = match / "complete_0.data", match / "truncated_0.data"
            write_stream(complete, initial_events() + [("P", 0, [3]), victory()])
            write_stream(truncated, initial_events())
            before = {p: sha256_file(p) for p in (complete, truncated)}
            output = Path(temp) / "processed"
            manifest = run_etl(source, output)
            self.assertEqual(manifest["counts"]["games"], 1)
            self.assertEqual(manifest["counts"]["quarantined_files"], 1)
            quarantine = json.loads((output / "quarantine.jsonl").read_text(encoding="utf-8"))
            self.assertEqual(quarantine["reason"], "missing_terminal_v")
            self.assertEqual(quarantine["event_count"], 7)
            output_hashes = {p.name: sha256_file(p) for p in output.iterdir()}
            run_etl(source, output)
            self.assertEqual(output_hashes, {p.name: sha256_file(p) for p in output.iterdir()})
            self.assertEqual(before, {p: sha256_file(p) for p in before})
            self.assertFalse(list(output.glob("*.tmp")))
            with self.assertRaisesRegex(ETLError, "read-only"):
                run_etl(source, source / "accidental-output")
            self.assertFalse((source / "accidental-output").exists())


if __name__ == "__main__":
    unittest.main()
