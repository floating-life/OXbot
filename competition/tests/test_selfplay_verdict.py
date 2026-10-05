# -*- coding: utf-8 -*-
import hashlib
import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tools"))

import selfplay_verdict as V  # noqa: E402


class SelfplayVerdictTest(unittest.TestCase):
    seed = 20261101

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = self.tmp.name
        self.candidate = self._write_bytes("candidate.npz", b"candidate-model")
        self.baseline = self._write_bytes("baseline.npz", b"real-v2-model")
        self.champion = self._write_bytes("previous_champion.npz", b"champion-model")
        self.candidate_sha = self._sha(self.candidate)
        self.baseline_sha = self._sha(self.baseline)
        self.champion_sha = self._sha(self.champion)
        self._write_json("training_status.json", {
            "training_ok": True,
            "exit_code": 0,
            "optimizer_steps": 12,
            "stop_reason": "max_hours",
            "candidate_sha256": self.candidate_sha,
        })

    def tearDown(self):
        self.tmp.cleanup()

    def _write_bytes(self, name, data):
        path = os.path.join(self.root, name)
        with open(path, "wb") as stream:
            stream.write(data)
        return path

    def _write_json(self, name, value):
        with open(os.path.join(self.root, name), "w", encoding="utf-8") as stream:
            json.dump(value, stream)

    @staticmethod
    def _sha(path):
        h = hashlib.sha256()
        with open(path, "rb") as stream:
            h.update(stream.read())
        return h.hexdigest()

    def _duplicate(self, name, deals=1000, lo=0.1, hi=0.2,
                   reference="baseline", verdict=None):
        if verdict is None:
            verdict = V._expected_verdict(lo, hi)
        if name == "rule":
            b = {"spec": "rule"}
        elif reference == "baseline":
            b = {"spec": "baseline.npz", "sha256": self.baseline_sha}
        else:
            b = {"spec": "previous_champion.npz", "sha256": self.champion_sha}
        self._write_json("vs_%s.json" % name, {
            "a": {"spec": "candidate.npz", "sha256": self.candidate_sha},
            "b": b,
            "seed": self.seed,
            "ladder_frac": 0.5,
            "deals": deals,
            "games": 2 * deals,
            "avg_score_diff_per_game": (lo + hi) / 2.0,
            "ci95_per_game": [lo, hi],
            "verdict": verdict,
            "game_win_rate": 0.7,
        })

    def _judge(self, games=200, errors=0, require_model=True,
               exit_code=0):
        self._write_json("judge.json", {
            "games": games,
            "errors": errors,
            "require_model": require_model,
            "failures": [] if errors == 0 else [{"game": 1}],
            "candidate_sha256": self.candidate_sha,
            "baseline_sha256": self.baseline_sha,
            "champion_sha256": self.champion_sha,
            "seed": self.seed,
            "driver": "inproc",
            "exit_code_for_test_only": exit_code,
        })

    def _complete(self, deals=1000, real_lo=0.1, champ_lo=0.1,
                  judge_games=200, judge_errors=0, require_model=True,
                  judge_exit_code=0):
        self._duplicate("realv2", deals=deals, lo=real_lo, hi=0.3)
        self._duplicate("rule", deals=deals, lo=0.2, hi=0.4)
        self._duplicate("champion", deals=deals, lo=champ_lo, hi=0.3,
                        reference="champion")
        self._judge(games=judge_games, errors=judge_errors,
                    require_model=require_model, exit_code=judge_exit_code)

    def _run(self, *extra, **kwargs):
        args = [self.root]
        if kwargs.get("require_champion", True):
            args.append("--require-champion")
        args.extend(extra)
        return V.main(args)

    def test_twenty_all_wins_do_not_promote(self):
        self._complete(deals=20)
        self.assertEqual(self._run("--deals", "20"), V.NO_PROMOTE)
        with open(os.path.join(self.root, "summary.json"), encoding="utf-8") as stream:
            summary = json.load(stream)
        self.assertFalse(summary["promote"])
        self.assertTrue(any("minimum" in x for x in summary["gate_failures"]))

    def test_thousand_positive_ci_promotes(self):
        self._complete(deals=1000)
        self.assertEqual(self._run(), V.PROMOTE)
        with open(os.path.join(self.root, "summary.json"), encoding="utf-8") as stream:
            summary = json.load(stream)
        self.assertTrue(summary["promote"])
        self.assertEqual(summary["status"], "PROMOTED")

    def test_missing_champion_is_invalid_when_required(self):
        self._duplicate("realv2")
        self._duplicate("rule")
        self._judge()
        os.unlink(os.path.join(self.root, "previous_champion.npz"))
        self.assertEqual(self._run(), V.INVALID_EVIDENCE)
        self.assertFalse(self._summary()["promote"])

    def test_missing_judge_is_invalid(self):
        self._duplicate("realv2")
        self._duplicate("rule")
        self._duplicate("champion", reference="champion")
        self.assertEqual(self._run(), V.INVALID_EVIDENCE)
        self.assertFalse(self._summary()["promote"])

    def test_first_run_without_champion_can_pass(self):
        self._duplicate("realv2")
        self._duplicate("rule")
        self._judge()
        os.unlink(os.path.join(self.root, "previous_champion.npz"))
        self.assertEqual(self._run(require_champion=False), V.PROMOTE)
        self.assertFalse(self._summary()["champion_required"])

    def test_frozen_champion_requires_comparison_even_without_flag(self):
        self._duplicate("realv2")
        self._duplicate("rule")
        self._judge()
        self.assertEqual(self._run(require_champion=False), V.INVALID_EVIDENCE)
        self.assertTrue(self._summary()["champion_required"])

    def test_candidate_sha_mismatch_is_invalid(self):
        self._complete()
        path = os.path.join(self.root, "vs_realv2.json")
        with open(path, encoding="utf-8") as stream:
            report = json.load(stream)
        report["a"]["sha256"] = "0" * 64
        self._write_json("vs_realv2.json", report)
        self.assertEqual(self._run(), V.INVALID_EVIDENCE)

    def test_incomplete_judge_identity_is_invalid(self):
        self._complete()
        path = os.path.join(self.root, "judge.json")
        with open(path, encoding="utf-8") as stream:
            report = json.load(stream)
        del report["candidate_sha256"]
        self._write_json("judge.json", report)
        self.assertEqual(self._run(), V.INVALID_EVIDENCE)

    def test_bad_judge_does_not_promote(self):
        self._complete(judge_games=200, judge_errors=1, require_model=False)
        # The report is structurally valid, but it fails the official gate.
        self.assertEqual(self._run(), V.NO_PROMOTE)
        self.assertFalse(self._summary()["promote"])

    def test_negative_ci_does_not_promote(self):
        self._complete(real_lo=-0.3, champ_lo=0.1)
        self.assertEqual(self._run(), V.NO_PROMOTE)
        summary = self._summary()
        self.assertFalse(summary["promote"])
        self.assertTrue(any("real-v2" in x for x in summary["gate_failures"]))

    def _summary(self):
        with open(os.path.join(self.root, "summary.json"), encoding="utf-8") as stream:
            return json.load(stream)


if __name__ == "__main__":
    unittest.main()
