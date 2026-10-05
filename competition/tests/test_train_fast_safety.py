"""CPU-only guards for the long-running DMC trainer."""

import json
import os
import sys
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fabledan import train_fast
from fabledan.train_fast import TrainingProgress, _bounded_ranges, _validate_infer_batch


def _cfg():
    return SimpleNamespace(max_seq=8, vocab=48, feat_dim=80)


def _batch():
    return [np.asarray([1, 2, 3], dtype=np.int16)], \
        [np.zeros((2, 80), dtype=np.float32)], [2]


def test_inference_cpu_batch_guard_accepts_valid_request():
    _validate_infer_batch(*_batch(), _cfg())


@pytest.mark.parametrize("mutate, message", [
    (lambda toks, feats: toks.__setitem__(0, np.asarray([1, 48])),
     "out-of-range"),
    (lambda toks, feats: feats.__setitem__(0, np.full((2, 80), np.nan)),
     "non-finite"),
    (lambda toks, feats: feats.__setitem__(0, np.zeros((1, 80))),
     "feature shape"),
])
def test_inference_cpu_batch_guard_rejects_bad_request(mutate, message):
    toks, feats, counts = _batch()
    mutate(toks, feats)
    with pytest.raises(RuntimeError, match=message):
        _validate_infer_batch(toks, feats, counts, _cfg())


def test_inference_cpu_batch_guard_rejects_oversized_tokens():
    toks, feats, counts = _batch()
    toks[0] = np.arange(9, dtype=np.int16)
    with pytest.raises(RuntimeError, match="token length"):
        _validate_infer_batch(toks, feats, counts, _cfg())


def test_safe_inference_ranges_bound_each_forward_and_preserve_order():
    ranges = _bounded_ranges(97, 32)
    assert ranges == [(0, 32), (32, 64), (64, 96), (96, 97)]
    assert all(0 < end - start <= 32 for start, end in ranges)
    assert ranges[0][0] == 0 and ranges[-1][1] == 97


def test_responsive_loop_without_samples_does_not_fake_progress(tmp_path, monkeypatch):
    clock = [1000.0]
    monkeypatch.setattr(train_fast.time, "time", lambda: clock[0])
    monkeypatch.setattr(train_fast.time, "monotonic", lambda: clock[0])
    path = tmp_path / "training_progress.json"
    writer = TrainingProgress(str(path), "recovery-run")
    state = dict(phase="collect", cycle=260, total_samples=34352494,
                 optimizer_steps=4164, elapsed_seconds=27187.5)
    writer.update(**state)
    clock[0] += 31
    writer.update(**state)
    record = json.loads(path.read_text())
    assert record["updated_at"] == 1031
    assert record["progress_at"] == 1000
    assert record["stop_reason"] == "running"
    assert record["run_id"] == "recovery-run"
    clock[0] += 31
    state["total_samples"] += 40
    writer.update(**state)
    assert json.loads(path.read_text())["progress_at"] == 1062


def test_progress_write_failure_keeps_last_valid_record(tmp_path, monkeypatch, capsys):
    path = tmp_path / "training_progress.json"
    writer = TrainingProgress(str(path), "recovery-run")
    state = dict(phase="learn", cycle=260, total_samples=34352494,
                 optimizer_steps=4164, elapsed_seconds=27187.5)
    writer.update(**state, force=True)
    old = path.read_bytes()

    def fail_replace(*_args):
        raise OSError("simulated disk error")

    monkeypatch.setattr(train_fast.os, "replace", fail_replace)
    writer.update(**state, force=True, error="original CUDA error")
    assert path.read_bytes() == old
    assert "could not write training progress" in capsys.readouterr().err


def test_final_failure_is_written_even_before_heartbeat_interval(tmp_path):
    path = tmp_path / "training_progress.json"
    writer = TrainingProgress(str(path), "recovery-run")
    state = dict(phase="learn", cycle=260, total_samples=34352494,
                 optimizer_steps=4164, elapsed_seconds=27187.5)
    writer.update(**state)
    state["phase"] = "failed"
    writer.update(**state, stop_reason="training failed", error="worker stopped", force=True)
    record = json.loads(path.read_text())
    assert record["phase"] == "failed"
    assert record["error"] == "worker stopped"
