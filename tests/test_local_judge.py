"""Evaluation contracts: teams, duplicate pairing, artifact identity and failures."""
import hashlib
import json
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from local_judge import (GameFailure, game_specs, model_for_seat, model_identity, paired_scores,
                         parse_process_response, score_teams)


def fabledan_container_fixture(dtype=2):
    """Small valid container; network tensor semantics belong to the C++ loader."""
    item_bytes = 2 if dtype == 1 else 4
    payload = bytes(64 * item_bytes)
    table = bytearray()
    for index in range(64):
        name = f"fixture.{index:02}".encode()
        table.extend(struct.pack("<HBBIQQ", len(name), 1, 0, 1, index * item_bytes, item_bytes))
        table.extend(name)
    header = (b"FBDN001\0" + struct.pack("<IIII", 1, dtype, 116 + len(table), 64)
              + struct.pack("<13I", 128, 4, 4, 64, 64, 512, 512, 3, 1024, 3, 512, 48, 80)
              + struct.pack("<fI", 1e-6, len(payload)) + hashlib.sha256(payload).digest())
    return header + table + payload


class EvaluationContracts(unittest.TestCase):
    def test_process_response_accepts_one_json_with_optional_keep_running_marker(self):
        response = {"response": [], "debug": "policy=stage_rules"}
        line = json.dumps(response)
        for stdout in (line, line + "\n", line + "\n>>>BOTZONE_REQUEST_KEEP_RUNNING<<<\n"):
            with self.subTest(stdout=stdout):
                self.assertEqual(parse_process_response(0, stdout, ""), response)

    def test_process_response_rejects_extra_output_or_failed_process(self):
        line = '{"response":[]}\n'
        marker = ">>>BOTZONE_REQUEST_KEEP_RUNNING<<<\n"
        for stdout in ("", line + line, line + "diagnostic\n", line + "\n",
                       "\n" + line, marker + line, line + " " + marker,
                       line + marker + marker, line + marker + "extra\n"):
            with self.subTest(stdout=stdout):
                with self.assertRaises(GameFailure) as caught:
                    parse_process_response(0, stdout, "")
                self.assertEqual(caught.exception.reason, "bot_process_contract_failed")
        for returncode, stderr in ((1, ""), (0, "unexpected stderr")):
            with self.subTest(returncode=returncode, stderr=stderr):
                with self.assertRaises(GameFailure) as caught:
                    parse_process_response(returncode, line, stderr,
                                           failure_reason="benchmark_process_contract_failed")
                self.assertEqual(caught.exception.reason, "benchmark_process_contract_failed")
        with self.assertRaises(json.JSONDecodeError):
            parse_process_response(0, "not JSON\n" + marker, "")

    def test_partner_points_are_not_double_counted(self):
        self.assertEqual(score_teams({"0": 3, "1": 0, "2": 3, "3": 0}), [3, 0])
        self.assertEqual(score_teams({"0": 0, "1": 2, "2": 0, "3": 2}), [0, 2])
        with self.assertRaises(GameFailure):
            score_teams({"0": 3, "1": 0, "2": 2, "3": 0})

    def test_team_selection_and_explicit_rule_seats(self):
        self.assertEqual([model_for_seat("weights.bin", "0", i) for i in range(4)],
                         ["weights.bin", "", "weights.bin", ""])
        self.assertEqual([model_for_seat("weights.bin", "1", i) for i in range(4)],
                         ["", "weights.bin", "", "weights.bin"])
        self.assertTrue(all(model_for_seat("weights.bin", "all", i) for i in range(4)))
        self.assertFalse(model_for_seat(None, "all", 0))

    def test_paired_points_follow_model_perspective(self):
        left = [{"deal_key": "a", "model_team": "0", "team_scores": [3, 0]},
                {"deal_key": "b", "model_team": "0", "team_scores": [0, 2]}]
        right = [{"deal_key": "a", "model_team": "1", "team_scores": [0, 1]},
                 {"deal_key": "b", "model_team": "1", "team_scores": [2, 0]}]
        result = paired_scores(left, right)
        self.assertEqual(result["pairs"], 2)
        self.assertEqual(result["total_model_point_margin"], 0)
        self.assertEqual((result["positive_pairs"], result["negative_pairs"]), (1, 1))
        with self.assertRaises(GameFailure):
            paired_scores(left, left)
        with self.assertRaises(GameFailure):
            paired_scores(left + [left[0]], right)

    def test_missing_pair_is_explicit(self):
        left = [{"deal_key": "a", "model_team": "0", "team_scores": [1, 0]}]
        result = paired_scores(left, [])
        self.assertEqual(result["pairs"], 0)
        self.assertEqual(result["unmatched_first"], 1)
        self.assertIsNone(result["mean_model_point_margin_per_pair"])

    def test_seed_schedule_does_not_depend_on_worker_count(self):
        specs = list(game_specs(40, 20261001))
        self.assertEqual(specs, list(game_specs(40, 20261001)))
        self.assertEqual([item["seed"] for item in specs], list(range(20261001, 20261041)))
        self.assertEqual(set(item["level"] for item in specs), set("234567890JQKA"))
        self.assertEqual(set(item["tribute"] for item in specs), {0, 1, 2})
        self.assertTrue(all(item["first"] % 2 != item["last"] % 2 for item in specs))

    def test_model_payload_is_verified(self):
        payload = b"fixed model fixture"
        header = json.dumps({"payload_sha256": hashlib.sha256(payload).hexdigest()}).encode()
        content = b"OXGDQ001" + len(header).to_bytes(4, "little") + header + payload
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "model.bin"
            path.write_bytes(content)
            identity = model_identity(path)
            self.assertEqual(identity["payload_sha256"], hashlib.sha256(payload).hexdigest())
            path.write_bytes(content[:-1] + b"x")
            with self.assertRaises(GameFailure):
                model_identity(path)

    def test_fabledan_container_identity_supports_both_dtypes(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "model.fbd"
            for dtype, name in ((1, "fp16"), (2, "fp32")):
                with self.subTest(dtype=name):
                    content = fabledan_container_fixture(dtype)
                    path.write_bytes(content)
                    identity = model_identity(path)
                    self.assertEqual(identity["format"], "FBDN001")
                    self.assertEqual(identity["format_version"], 1)
                    self.assertEqual(identity["dtype"], name)
                    self.assertEqual(identity["manifest_selection_default"], "raw")
                    self.assertEqual(identity["config"]["feat_dim"], 80)
                    self.assertEqual(identity["tensor_count"], 64)
                    self.assertEqual(identity["bytes"], len(content))
                    self.assertEqual(identity["file_sha256"], hashlib.sha256(content).hexdigest())
                    self.assertEqual(identity["payload_sha256"], content[84:116].hex())

    def test_fabledan_corrupt_container_is_rejected(self):
        content = fabledan_container_fixture()
        corruptions = [
            ("magic", content[:7] + b"x" + content[8:], "model_header_magic_invalid"),
            ("short_header", content[:115], "model_header_length_invalid"),
            ("truncated_payload", content[:-1], "model_payload_length_invalid"),
            ("trailing_payload", content + b"x", "model_payload_length_invalid"),
            ("payload_sha", content[:-1] + b"x", "model_payload_sha_invalid"),
        ]
        # Include metadata changes that leave the payload and its SHA intact.
        for name, offset, format_, value, reason in (
                ("version", 8, "I", 2, "model_version_invalid"),
                ("dtype", 12, "I", 3, "model_dtype_invalid"),
                ("short_header_bytes", 16, "I", 115, "model_header_length_invalid"),
                ("long_header_bytes", 16, "I", len(content) + 1, "model_header_length_invalid"),
                ("tensor_count", 20, "I", 63, "model_tensor_count_invalid"),
                ("architecture", 24, "I", 64, "model_config_invalid"),
                ("rms_epsilon", 76, "f", 0, "model_config_invalid"),
                ("rms_nan", 76, "f", float("nan"), "model_config_invalid"),
                ("payload_bytes", 80, "I", 0, "model_payload_length_invalid"),
                ("table_name_length", 116, "H", 513, "model_tensor_table_invalid"),
                ("table_rank", 118, "B", 0, "model_tensor_table_invalid"),
                ("table_dimension", 120, "I", 0, "model_tensor_shape_invalid"),
                ("table_offset", 124, "Q", 256, "model_tensor_bounds_invalid"),
                ("table_item_length", 132, "Q", 3, "model_tensor_length_invalid"),
                ("table_overlap", 158, "Q", 0, "model_tensor_payload_gap")):
            damaged = bytearray(content)
            struct.pack_into("<" + format_, damaged, offset, value)
            corruptions.append((name, damaged, reason))
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "model.fbd"
            for name, damaged, reason in corruptions:
                with self.subTest(corruption=name):
                    path.write_bytes(damaged)
                    with self.assertRaises(GameFailure) as caught:
                        model_identity(path)
                    self.assertEqual(caught.exception.reason, reason)

    def test_startup_failure_leaves_valid_failure_report(self):
        with tempfile.TemporaryDirectory() as temporary:
            report = Path(temporary) / "result.json"
            process = subprocess.run([sys.executable, str(ROOT / "tools" / "local_judge.py"),
                                      "--games", "1", "--probe", str(Path(temporary) / "absent-probe"),
                                      "--report", str(report)], cwd=ROOT, text=True, capture_output=True)
            self.assertEqual(process.returncode, 1)
            result = json.loads(report.read_text(encoding="utf-8"))
            self.assertEqual(result["status"], "failed")
            failure = json.loads(Path(result["failure"]["path"]).read_text(encoding="utf-8"))
            self.assertEqual(failure["reason"], "FileNotFoundError")
            self.assertEqual(result["game_records"], [])


if __name__ == "__main__":
    unittest.main()
