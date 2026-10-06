"""Exercise launcher failures, signals and completion without CUDA or models."""

import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tempfile
import textwrap
import time
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "selfplay_eval_5080_wsl.sh"


@unittest.skipUnless(
    os.name == "posix" and all(shutil.which(cmd) for cmd in ("bash", "flock", "setsid")),
    "the WSL launcher requires Linux bash, flock and setsid",
)
class SelfplayScriptTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="oxbot-selfplay-script-")
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.out = self.root / "run"
        self.out.mkdir()
        self.calls = self.root / "python-calls.jsonl"
        self.fake_python = self.root / "fake-python"
        self.fake_python.write_text("#!" + sys.executable + "\n" + textwrap.dedent(r'''
            import json, os, subprocess, sys, time, types
            from pathlib import Path
            args = sys.argv[1:]
            with open(os.environ['OXBOT_TEST_CALLS'], 'a') as log:
                log.write(json.dumps(args) + '\n')
            if args == ['-']:
                sys.stdin.read()  # bypass only the CUDA preflight
                sys.exit(int(os.environ.get('OXBOT_TEST_PREFLIGHT_RC', '0')))
            if args[:2] == ['-m', 'fabledan.train_fast']:
                behavior = os.environ.get('OXBOT_TEST_TRAIN', 'fail')
                if behavior == 'block':
                    while True:
                        time.sleep(1)
                if behavior == 'fail':
                    sys.exit(23)
                if behavior == 'fail-with-child':
                    subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(120)'])
                    sys.exit(23)
                if behavior != 'no-checkpoint':
                    meta = dict(training_kind='dmc', optimizer_steps=10, cycle=2,
                                training_args={'cycles': 2}, elapsed_seconds=360.,
                                stop_reason='time limit reached')
                    if behavior == 'interrupted':
                        meta.update(stop_reason='interrupted', elapsed_seconds=100.)
                    if behavior == 'periodic':
                        meta.update(stop_reason='cycles completed', elapsed_seconds=100.)
                    if behavior == 'underbudget':
                        meta.update(elapsed_seconds=100.)
                    out = Path(args[args.index('--out') + 1])
                    (out / 'latest.pt').write_text(json.dumps({'meta': meta}))
                sys.exit(0)
            if args and args[0] == '-':
                # Execute real launcher status/budget/finalization code while
                # replacing only torch/model I/O with tiny JSON checkpoints.
                torch = types.ModuleType('torch')
                torch.load = lambda path, **kwargs: json.loads(Path(path).read_text())
                torch.set_num_threads = lambda _: None
                class Finite:
                    def all(self): return self
                    def item(self): return True
                torch.isfinite = lambda _: Finite()
                sys.modules['torch'] = torch
                model_io = types.ModuleType('fabledan.model_torch')
                class Model:
                    def eval(self): return self
                    def state_dict(self): return {'weight': 1}
                model_io.load_ckpt = lambda path, **kwargs: (Model(), torch.load(path))
                model_io.export_npz = lambda model, path: Path(path).write_bytes(b'candidate weights')
                sys.modules['fabledan.model_torch'] = model_io
                sys.argv = args
                exec(compile(sys.stdin.read(), '<launcher-inline>', 'exec'), {'__name__': '__main__'})
                sys.exit(0)
            if args and args[0].endswith('duplicate_eval.py'):
                if os.environ.get('OXBOT_TEST_DUPLICATE') == 'block':
                    while True:
                        time.sleep(1)
                if os.environ.get('OXBOT_TEST_DUPLICATE') == 'fail':
                    sys.exit(31)
                report = Path(args[args.index('--report') + 1])
                report.write_text('{}')
                sys.exit(0)
            if args and args[0].endswith('judge_runner.py'):
                report = Path(args[args.index('--report') + 1])
                report.write_text(json.dumps({'games': 200, 'errors': 0,
                                              'require_model': True, 'failures': []}))
                sys.exit(0)
            if args and args[0].endswith('selfplay_verdict.py'):
                (Path(args[1]) / 'summary.json').write_text(json.dumps(
                    {'status': 'NOT_PROMOTED', 'promote': False}))
                sys.exit(3)
            sys.exit('unexpected invocation: ' + repr(args))
        '''), encoding="utf-8")
        self.fake_python.chmod(0o700)
        self.env = dict(os.environ)
        self.env.update(
            OXBOT_PYTHON=str(self.fake_python),
            OXBOT_TEST_CALLS=str(self.calls),
            OXBOT_OUT=str(self.out),
            OXBOT_WARM_START=str(self.root / "warm.pt"),
            OXBOT_BASELINE=str(self.root / "baseline.npz"),
            OXBOT_JUDGE=str(self.root / "judge.py"),
            OXBOT_HOURS="0.1",
            OXBOT_RUN_ID="cpu-lifecycle-test",
            OXBOT_SIGNAL_GRACE_SECONDS="1",
            OXBOT_DEALS="1000",
            OXBOT_JUDGE_GAMES="200",
            OXBOT_EVAL_SEED="20261101",
        )
        for name in ("warm.pt", "baseline.npz", "judge.py"):
            (self.root / name).write_bytes(b"test placeholder")

    def run_script(self, mode="all"):
        return subprocess.run(
            ["bash", str(SCRIPT), mode], env=self.env, text=True,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=20,
        )

    def read_status(self, name="last_pipeline.json"):
        return json.loads((self.out / name).read_text())

    def python_calls(self):
        return [json.loads(line) for line in self.calls.read_text().splitlines()]

    def write_resume(self, elapsed=150, reason="running"):
        checkpoint = self.out / "latest.pt"
        checkpoint.write_text(json.dumps({"meta": {
            "elapsed_seconds": elapsed, "training_kind": "dmc",
            "optimizer_steps": 10, "stop_reason": reason,
        }}))
        return checkpoint

    def assert_no_evaluation(self):
        self.assertFalse(any(c and c[0].endswith("duplicate_eval.py") for c in self.python_calls()))
        self.assertFalse((self.out / "champion").exists())

    def test_failed_training_never_evaluates_stale_checkpoint(self):
        checkpoint = self.write_resume()
        previous = checkpoint.read_bytes()
        (self.out / "latest.npz").write_bytes(b"stale weights")

        result = self.run_script()

        self.assertEqual(result.returncode, 23, result.stdout)
        self.assertIn("train_fast exited with 23", result.stdout)
        status = self.read_status("last_training.json")
        self.assertFalse(status["training_ok"])
        self.assertEqual(status["exit_code"], 23)
        self.assertEqual(status["status"], "failed")
        self.assertEqual(status["run_id"], self.env["OXBOT_RUN_ID"])
        self.assertEqual(self.read_status()["stage"], "training")
        self.assertIsNotNone(self.read_status()["finished_at"])
        self.assert_no_evaluation()
        self.assertFalse((self.out / "eval").exists(), result.stdout)
        self.assertEqual(checkpoint.read_bytes(), previous)
        self.assertEqual((self.out / "latest.npz").read_bytes(), b"stale weights")

    def test_resume_uses_only_remaining_cumulative_budget(self):
        self.write_resume(elapsed=150)

        result = self.run_script("train")

        self.assertEqual(result.returncode, 23, result.stdout)
        calls = [c for c in self.python_calls() if c[:2] == ["-m", "fabledan.train_fast"]]
        self.assertEqual(len(calls), 1)
        self.assertAlmostEqual(float(calls[0][calls[0].index("--max-hours") + 1]), 210 / 3600)
        self.assertEqual(calls[0][calls[0].index("--batch") + 1], "4096")
        self.assertEqual(calls[0][calls[0].index("--micro-batch") + 1], "128")

    def test_failed_trainer_does_not_leave_actors_holding_pipeline_open(self):
        self.env["OXBOT_TEST_TRAIN"] = "fail-with-child"

        # The orphan inherits stdout and fd 9; subprocess.run times out if it
        # survives. A second launcher also needs to be able to acquire the lock.
        result = self.run_script("train")
        self.assertEqual(result.returncode, 23, result.stdout)
        self.env["OXBOT_TEST_TRAIN"] = "fail"
        retry = self.run_script("train")
        self.assertEqual(retry.returncode, 23, retry.stdout)
        self.assertNotIn("another self-play/evaluation run", retry.stdout)

    def test_invalid_elapsed_never_grants_fresh_training_budget(self):
        for elapsed in (None, -1, float("nan"), "150", True):
            with self.subTest(elapsed=elapsed):
                checkpoint = self.write_resume(elapsed)
                previous = checkpoint.read_bytes()
                result = self.run_script()
                self.assertNotEqual(result.returncode, 0, result.stdout)
                self.assertIn("budget cannot be reset", result.stdout)
                self.assertEqual(self.read_status()["status"], "failed")
                self.assertEqual(self.read_status()["stage"], "training_budget")
                self.assertFalse(any(c[:2] == ["-m", "fabledan.train_fast"] for c in self.python_calls()))
                self.assert_no_evaluation()
                self.assertEqual(checkpoint.read_bytes(), previous)

    def test_exhausted_budget_does_not_restart_training_or_guess_completion(self):
        self.write_resume(elapsed=360, reason="cycles completed")

        result = self.run_script()

        self.assertNotEqual(result.returncode, 0, result.stdout)
        self.assertIn("total training budget exhausted", result.stdout)
        self.assertEqual(self.read_status()["status"], "failed")
        self.assertFalse(any(c[:2] == ["-m", "fabledan.train_fast"] for c in self.python_calls()))
        self.assert_no_evaluation()

    def test_preflight_failure_is_durable(self):
        self.env["OXBOT_TEST_PREFLIGHT_RC"] = "7"

        result = self.run_script()

        self.assertNotEqual(result.returncode, 0, result.stdout)
        status = self.read_status()
        self.assertEqual((status["stage"], status["status"]), ("preflight", "failed"))
        self.assertIn("PyTorch/CUDA", status["error"])
        self.assertIsNotNone(status["finished_at"])
        self.assert_no_evaluation()

    def test_train_mode_records_verified_completion_without_evaluation(self):
        self.env["OXBOT_TEST_TRAIN"] = "success"

        result = self.run_script("train")

        self.assertEqual(result.returncode, 0, result.stdout)
        status = self.read_status("last_training.json")
        self.assertTrue(status["training_ok"])
        self.assertEqual((status["status"], status["exit_code"]), ("completed", 0))
        self.assertEqual(status["checkpoint_sha256"], hashlib.sha256((self.out / "latest.pt").read_bytes()).hexdigest())
        self.assertEqual((self.read_status()["mode"], self.read_status()["status"]), ("train", "completed"))
        self.assert_no_evaluation()
        self.assertFalse((self.out / "eval").exists())

    def test_zero_exit_does_not_prove_final_training_completed(self):
        for behavior in ("interrupted", "periodic", "underbudget", "no-checkpoint"):
            with self.subTest(behavior=behavior):
                self.env["OXBOT_TEST_TRAIN"] = behavior
                result = self.run_script()
                self.assertNotEqual(result.returncode, 0, result.stdout)
                self.assertIn("verified final checkpoint", result.stdout)
                self.assertFalse(self.read_status("last_training.json")["training_ok"])
                self.assertEqual(self.read_status()["status"], "failed")
                self.assert_no_evaluation()

    def test_all_mode_reaches_eval_and_maintains_formal_parameters(self):
        self.env["OXBOT_TEST_TRAIN"] = "success"

        result = self.run_script()

        self.assertEqual(result.returncode, 0, result.stdout)
        status = self.read_status()
        self.assertEqual((status["mode"], status["status"], status["exit_code"]), ("all", "completed", 0))
        self.assertTrue(self.read_status("last_training.json")["training_ok"])
        self.assertEqual(json.loads((Path(status["eval_dir"]) / "summary.json").read_text())["status"], "NOT_PROMOTED")
        duplicate_calls = [c for c in self.python_calls() if c[0].endswith("duplicate_eval.py")]
        self.assertEqual(len(duplicate_calls), 2)
        for call in duplicate_calls:
            self.assertEqual(call[call.index("--deals") + 1], "1000")
            self.assertEqual(call[call.index("--seed") + 1], "20261101")
        judge = next(c for c in self.python_calls() if c[0].endswith("judge_runner.py"))
        self.assertEqual(judge[judge.index("--games") + 1], "200")
        self.assertIn("--require-model", judge)

    def test_eval_failure_retains_training_success_but_fails_pipeline(self):
        self.env["OXBOT_TEST_TRAIN"] = "success"
        result = self.run_script("train")
        self.assertEqual(result.returncode, 0, result.stdout)
        training = (self.out / "last_training.json").read_bytes()
        self.env["OXBOT_RUN_ID"] = "eval-failure-test"
        self.env["OXBOT_TEST_DUPLICATE"] = "fail"

        result = self.run_script("eval")

        self.assertNotEqual(result.returncode, 0, result.stdout)
        status = self.read_status()
        self.assertEqual((status["mode"], status["status"], status["stage"]), ("eval", "failed", "eval_realv2"))
        self.assertEqual(status["run_id"], "eval-failure-test")
        self.assertEqual((self.out / "last_training.json").read_bytes(), training)
        self.assertFalse((self.out / "champion").exists())

    def test_custom_champion_is_frozen_and_required_for_promotion(self):
        self.env["OXBOT_TEST_TRAIN"] = "success"
        champion_dir = self.root / "current champion"
        champion_dir.mkdir()
        champion = champion_dir / "champion.npz"
        champion.write_bytes(b"current champion weights")
        self.env["OXBOT_CHAMPION_DIR"] = str(champion_dir)

        result = self.run_script()

        self.assertEqual(result.returncode, 0, result.stdout)
        folder = Path(self.read_status()["eval_dir"])
        status = json.loads((folder / "training_status.json").read_text())
        self.assertTrue(status["champion_required"])
        self.assertEqual((folder / "previous_champion.npz").read_bytes(), champion.read_bytes())
        calls = self.python_calls()
        champion_duel = next(c for c in calls if c[0].endswith("duplicate_eval.py")
                             and c[c.index("--b") + 1] == str(folder / "previous_champion.npz"))
        self.assertEqual(champion_duel[champion_duel.index("--deals") + 1], "1000")
        verdict = next(c for c in calls if c[0].endswith("selfplay_verdict.py"))
        self.assertIn("--require-champion", verdict)

    def test_eval_rejects_checkpoint_changed_after_verified_completion(self):
        self.env["OXBOT_TEST_TRAIN"] = "success"
        result = self.run_script("train")
        self.assertEqual(result.returncode, 0, result.stdout)
        self.write_resume(elapsed=360, reason="cycles completed")

        result = self.run_script("eval")

        self.assertNotEqual(result.returncode, 0, result.stdout)
        self.assertIn("does not match the verified final training checkpoint", result.stdout)
        self.assertEqual(self.read_status()["status"], "failed")
        self.assert_no_evaluation()

    def test_eval_signal_preserves_verified_training_but_marks_pipeline_interrupted(self):
        self.env["OXBOT_TEST_TRAIN"] = "success"
        result = self.run_script("train")
        self.assertEqual(result.returncode, 0, result.stdout)
        training = (self.out / "last_training.json").read_bytes()
        self.env["OXBOT_TEST_DUPLICATE"] = "block"
        self.env["OXBOT_RUN_ID"] = "eval-signal-test"
        process = subprocess.Popen(
            ["bash", str(SCRIPT), "eval"], env=self.env, text=True,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        )
        try:
            self.wait_for_child("eval_realv2")
            process.send_signal(signal.SIGTERM)
            output, _ = process.communicate(timeout=10)
            self.assertEqual(process.returncode, 143, output)
            self.assertEqual(self.read_status()["status"], "interrupted")
            self.assertEqual(self.read_status()["stage"], "eval_realv2")
            self.assertEqual((self.out / "last_training.json").read_bytes(), training)
            self.assertFalse(any(c[0].endswith("judge_runner.py") for c in self.python_calls()))
        finally:
            if process.poll() is None:
                process.kill()
                process.communicate(timeout=5)

    def wait_for_child(self, stage):
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            try:
                status = self.read_status()
                if status["run_id"] == self.env["OXBOT_RUN_ID"] \
                        and status["stage"] == stage and status["child_pid"]:
                    return status
            except (FileNotFoundError, json.JSONDecodeError):
                pass
            time.sleep(0.03)
        self.fail("fake stage did not start: " + stage)

    def test_signals_record_interruption_and_terminate_trainer(self):
        self.env["OXBOT_TEST_TRAIN"] = "block"
        for name, signum in (("TERM", signal.SIGTERM), ("INT", signal.SIGINT), ("HUP", signal.SIGHUP)):
            with self.subTest(signal=name):
                self.env["OXBOT_RUN_ID"] = "signal-" + name
                process = subprocess.Popen(
                    ["bash", str(SCRIPT)], env=self.env, text=True,
                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                )
                try:
                    status = self.wait_for_child("training")
                    child_pid = status["child_pid"]
                    process.send_signal(signum)
                    output, _ = process.communicate(timeout=10)
                    self.assertEqual(process.returncode, 128 + signum, output)
                    status = self.read_status()
                    self.assertEqual((status["status"], status["signal"]), ("interrupted", name))
                    self.assertEqual(status["exit_code"], 128 + signum)
                    self.assertIsNone(status["child_pid"])
                    training = self.read_status("last_training.json")
                    self.assertFalse(training["training_ok"])
                    self.assertEqual(training["status"], "interrupted")
                    self.assertEqual(training["exit_code"], 128 + signum)
                    with self.assertRaises(ProcessLookupError):
                        os.kill(child_pid, 0)
                    self.assert_no_evaluation()
                finally:
                    if process.poll() is None:
                        process.kill()
                        process.communicate(timeout=5)

    def test_same_output_directory_rejects_concurrent_run_without_status_changes(self):
        import fcntl

        sentinel = b'{"run_id":"active-run","status":"running"}'
        for name in ("last_pipeline.json", "last_training.json"):
            (self.out / name).write_bytes(sentinel)
        with (self.out / ".selfplay.lock").open("w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            result = self.run_script()

        self.assertNotEqual(result.returncode, 0, result.stdout)
        self.assertIn("another self-play/evaluation run", result.stdout)
        self.assertFalse(self.calls.exists(), result.stdout)
        for name in ("last_pipeline.json", "last_training.json"):
            self.assertEqual((self.out / name).read_bytes(), sentinel)
        self.assertFalse((self.out / "eval").exists())
        self.assertFalse((self.out / "champion").exists())


if __name__ == "__main__":
    unittest.main()
