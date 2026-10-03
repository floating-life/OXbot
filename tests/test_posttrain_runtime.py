"""DMC warm start, normalized team credit, and recoverable learner state."""
from __future__ import annotations

import hashlib
import itertools
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "competition"))

from fabledan.encode import FEAT_DIM
from fabledan.engine import GuandanRound
from fabledan.model_torch import FableDanNet, ModelConfig, save_ckpt
from fabledan.ring import RingRunner
from fabledan.train import DMC_REWARD_SCHEMA, Replay, initialize_training


def tiny_checkpoint(path, kind="real_bc"):
    torch.set_num_threads(1)
    torch.manual_seed(91)
    cfg = ModelConfig(d_model=16, n_blocks=1, n_heads=1, qk_dim=8, v_dim=8,
                      ffn_hidden=32, hand_hidden=32, n_hand_layers=1,
                      q_hidden=32, n_q_layers=1, ntp_weight=0)
    model = FableDanNet(cfg)
    opt = torch.optim.Adam(model.parameters(), lr=0.009)
    # Populate source Adam moments; a warm start must not inherit them.
    model.q_head[-1].bias.grad = torch.ones_like(model.q_head[-1].bias)
    opt.step()
    with torch.no_grad():
        model.q_head[-1].bias.fill_(14)
    meta = {"training_kind": kind, "cycle": 0, "total_samples": 339972,
            "epoch": 4, "real_data_manifest_sha256": "synthetic-fixture"}
    save_ckpt(model, opt, meta, str(path))
    return model


def args_for(tmp_path, **changes):
    args = dict(out=str(tmp_path / "new-run"), device="cpu", resume="", warm_start="",
                warm_start_q_scale=None, n_blocks=None, ntp_weight=None, lr=None,
                trainer="train_fast")
    args.update(changes)
    return SimpleNamespace(**args)


def test_bc_warm_start_preserves_ranking_scales_output_and_resets_optimizer(tmp_path):
    source = tmp_path / "bc.pt"
    original = tiny_checkpoint(source)
    before = hashlib.sha256(source.read_bytes()).hexdigest()
    loaded, opt, prior, origin = initialize_training(args_for(tmp_path, warm_start=str(source)))
    tokens = torch.tensor([[1, 3, 15, 20, 35]])
    lengths = torch.tensor([tokens.shape[1]])
    features = torch.randn(1, 12, FEAT_DIM)
    with torch.no_grad():
        old_q = original(tokens, lengths, features)[0]
        new_q = loaded(tokens, lengths, features)[0]
    torch.testing.assert_close(new_q, old_q * 0.05, atol=1e-6, rtol=1e-6)
    assert torch.argmax(new_q).item() == torch.argmax(old_q).item()
    assert not opt.state and opt.param_groups[0]["lr"] == 1e-4
    assert prior == {} and origin["q_output_scale"] == 0.05
    assert origin["checkpoint_sha256"] == before
    assert origin["source_training_kind"] == "real_bc"
    assert hashlib.sha256(source.read_bytes()).hexdigest() == before


def test_credit_checkpoint_can_warm_start_without_reapplying_bc_scale(tmp_path):
    source = tmp_path / "credit.pt"
    original = tiny_checkpoint(source, kind="posttrain_credit")
    loaded, opt, prior, origin = initialize_training(args_for(tmp_path, warm_start=str(source),
                                                            warm_start_q_scale=1.0))
    for name, value in original.state_dict().items():
        torch.testing.assert_close(loaded.state_dict()[name], value)
    assert origin["source_training_kind"] == "posttrain_credit"
    assert origin["q_output_scale"] == 1 and not opt.state and not prior


@pytest.mark.parametrize("kind", ["real_bc", "posttrain_credit"])
def test_resume_does_not_treat_other_objectives_as_dmc(tmp_path, kind):
    source = tmp_path / "other.pt"
    tiny_checkpoint(source, kind=kind)
    with pytest.raises(ValueError, match="requires a DMC checkpoint"):
        initialize_training(args_for(tmp_path, resume=str(source)))


def test_warm_start_cannot_overwrite_source_or_an_existing_run(tmp_path):
    source = tmp_path / "bc.pt"
    tiny_checkpoint(source)
    with pytest.raises(ValueError, match="preserve its source"):
        initialize_training(args_for(tmp_path, out=str(tmp_path), warm_start=str(source)))
    out = tmp_path / "new-run"
    out.mkdir()
    (out / "latest.pt").write_bytes(b"an existing run")
    with pytest.raises(ValueError, match="already contains checkpoints"):
        initialize_training(args_for(tmp_path, warm_start=str(source)))


@pytest.mark.parametrize("scale", [0, -0.01, float("inf"), float("nan"), 2])
def test_invalid_warm_start_scale_fails_before_loading(tmp_path, scale):
    with pytest.raises(ValueError, match="scale must"):
        initialize_training(args_for(tmp_path, warm_start="unused.pt", warm_start_q_scale=scale))


def test_fifo_replay_roundtrip_preserves_next_seeded_batch_and_targets():
    replay = Replay(3, belief_dim=45)
    for i in range(5):
        replay.add([1, 3, 32 + i], np.full(FEAT_DIM, i / 5, dtype=np.float32),
                   (i - 2) / 3, np.full(45, i / 8, dtype=np.float32))
    loaded = Replay(3, belief_dim=45)
    loaded.load_state_dict(replay.state_dict())
    assert (loaded.n, loaded.ptr) == (3, 2)
    a = replay.sample(16, np.random.default_rng(7))
    b = loaded.sample(16, np.random.default_rng(7))
    for x, y in zip(a, b):
        np.testing.assert_array_equal(x, y)
    # Overwrite the same next physical slot after resume.
    for buf in (replay, loaded):
        buf.add([1, 3], np.zeros(FEAT_DIM), -1, np.zeros(45))
    np.testing.assert_array_equal(replay.targ, loaded.targ)


def test_replay_rejects_wrong_reward_scale_missing_labels_and_incompatible_resume():
    replay = Replay(2, belief_dim=45)
    with pytest.raises(ValueError, match="normalized terminal"):
        replay.add([1, 3], np.zeros(FEAT_DIM), 3, np.zeros(45))
    with pytest.raises(ValueError, match="belief label"):
        replay.add([1, 3], np.zeros(FEAT_DIM), 1)
    with pytest.raises(ValueError, match="capacity/belief_dim"):
        Replay(4, belief_dim=45).load_state_dict(replay.state_dict())


def test_terminal_reward_credits_both_partners_and_values_third_place():
    rnd = GuandanRound.__new__(GuandanRound)
    for ranking in itertools.permutations(range(4)):
        reward = rnd._rewards(list(ranking))
        assert reward[0] == reward[2] == -reward[1] == -reward[3]
        assert abs(reward[ranking[0]]) in (1, 2, 3)
    assert rnd._rewards([3, 2, 1, 0])[0] / 3 == -2 / 3
    assert rnd._rewards([3, 2, 0, 1])[0] / 3 == -1 / 3
    runner = RingRunner.__new__(RingRunner)
    samples = [(np.array([1, 3]), np.zeros(FEAT_DIM), p, np.zeros(45)) for p in range(4)]
    runner.finished = [(samples, rnd._rewards([3, 2, 0, 1]))]
    targets = [float(s[2]) for s in runner.pop_episodes()]
    np.testing.assert_allclose(targets, [-1 / 3, 1 / 3, -1 / 3, 1 / 3])
    assert not runner.finished


@pytest.mark.parametrize("trainer", ["train_fast", "train"])
def test_one_cycle_cpu_warm_start_then_resume_restores_dmc_checkpoint(tmp_path, trainer):
    source = tmp_path / "bc.pt"
    tiny_checkpoint(source)
    original_sha = hashlib.sha256(source.read_bytes()).hexdigest()
    out = tmp_path / "dmc"
    command = [sys.executable, "-m", "fabledan." + trainer, "--out", str(out),
               "--device", "cpu", "--actors", "1", "--buffer", "16", "--diversity", "1",
               "--steps-per-cycle", "1", "--batch", "2", "--eps", "0.15", "--top-k", "1",
               "--eval-cycles", "99", "--ckpt-cycles", "1", "--seed", "23"]
    if trainer == "train_fast":
        command += ["--infer-device", "cpu", "--ring", "1", "--micro-batch", "2",
                    "--belief-weight", "0", "--eval-games", "2", "--export-cycles", "0",
                    "--snapshot-cycles", "99", "--ladder-frac", "1", "--max-hours", "0.02"]
    first = subprocess.run(command + ["--cycles", "1", "--warm-start", str(source)],
                           cwd=ROOT / "competition", capture_output=True, text=True, timeout=90)
    assert first.returncode == 0, first.stdout + first.stderr
    ck = torch.load(out / "latest.pt", map_location="cpu", weights_only=False)
    assert ck["meta"]["training_kind"] == "dmc"
    assert ck["meta"]["reward_schema"] == DMC_REWARD_SCHEMA
    assert ck["meta"]["cycle"] == 1 and ck["meta"]["optimizer_steps"] == 1
    assert ck["meta"]["replay"]["n"] == 16
    assert ck["meta"]["initialization"]["checkpoint_sha256"] == original_sha
    assert ck["meta"]["total_samples"] < 339972
    assert (out / "latest.npz").exists()
    second = subprocess.run(command + ["--cycles", "2", "--resume", str(out / "latest.pt")],
                            cwd=ROOT / "competition", capture_output=True, text=True, timeout=90)
    assert second.returncode == 0, second.stdout + second.stderr
    resumed = torch.load(out / "latest.pt", map_location="cpu", weights_only=False)
    assert resumed["meta"]["cycle"] == 2 and resumed["meta"]["optimizer_steps"] == 2
    assert resumed["meta"]["initialization"] == ck["meta"]["initialization"]
    assert resumed["meta"]["startup"]["mode"] == "resume"
    assert resumed["meta"]["total_samples"] > ck["meta"]["total_samples"]
    assert hashlib.sha256(source.read_bytes()).hexdigest() == original_sha
