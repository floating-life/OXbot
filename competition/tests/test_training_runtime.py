"""Checks for fair paired evaluation and recoverable checkpoint writes."""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fabledan.agents import RuleAgent
from fabledan.evaluate import evaluate, make_agent, validate_evaluation


def test_identical_deterministic_policies_cancel_on_paired_deals():
    # A deal may heavily favor one team. Replaying it with the policies swapped
    # must cancel that advantage when both policies are identical.
    assert evaluate(RuleAgent, RuleAgent, games=20, seed=81,
                    duplicate=True, ladder_frac=0.5) == (0.5, 0.0)


def test_seeded_random_baseline_repeats():
    def run():
        return evaluate(RuleAgent, make_agent("random", seed=82), games=20,
                        seed=83, duplicate=True)
    assert run() == run()


@pytest.mark.parametrize("games,duplicate,fraction", [
    (0, False, 0), (3, True, 0), (2, True, -0.1), (2, True, 1.1),
    (2, True, float("nan")),
])
def test_reject_incomplete_pairs_and_invalid_distribution(games, duplicate, fraction):
    with pytest.raises(ValueError):
        validate_evaluation(games, duplicate, fraction)


def test_failed_checkpoint_write_preserves_previous_checkpoint(tmp_path, monkeypatch):
    pytest.importorskip("torch")
    from fabledan import train_fast
    checkpoint = tmp_path / "latest.pt"
    checkpoint.write_bytes(b"previous valid checkpoint")

    def interrupted_write(model, optimizer, meta, path):
        with open(path, "wb") as f:
            f.write(b"incomplete next checkpoint")
        raise OSError("simulated disk write failure")

    monkeypatch.setattr(train_fast, "save_ckpt", interrupted_write)
    with pytest.raises(OSError, match="disk write failure"):
        train_fast.atomic_checkpoint(None, None, {}, str(checkpoint))
    assert checkpoint.read_bytes() == b"previous valid checkpoint"
    assert not (tmp_path / "latest.pt.tmp").exists()


def test_evaluation_can_stop_between_games():
    calls = 0

    def stop():
        nonlocal calls
        calls += 1
        if calls == 2:
            raise TimeoutError("training deadline")

    with pytest.raises(TimeoutError, match="deadline"):
        evaluate(RuleAgent, RuleAgent, games=20, duplicate=True, check_stop=stop)
    assert calls == 2
