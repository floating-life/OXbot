"""Feature-v2 model migration and runtime contracts, separate from encoding QA."""
from __future__ import annotations

import hashlib
from pathlib import Path
import random
import subprocess
import sys
from types import SimpleNamespace
import zipfile

import numpy as np
import pytest

torch = pytest.importorskip("torch")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "competition"))

from fabledan.agents import NumpyAgent, TorchAgent
from fabledan.combos import classify_claim, gen_moves
from fabledan.encode import FEAT_DIM, STRUCTURE_FEAT_DIM, encode_decision
from fabledan.model_np import NumpyModel
from fabledan.model_torch import FableDanNet, ModelConfig, export_npz, save_ckpt
from fabledan.packaging import pack
from fabledan.ring import RingRunner
from fabledan.train import DMC_REWARD_SCHEMA, Replay, _ActorPolicy, initialize_training


def config(feat_dim=FEAT_DIM):
    return ModelConfig(d_model=16, n_blocks=1, n_heads=1, qk_dim=8, v_dim=8,
                       ffn_hidden=32, hand_hidden=32, n_hand_layers=1,
                       q_hidden=32, n_q_layers=1, ntp_weight=0, feat_dim=feat_dim)


def initialize_args(tmp_path, **changes):
    values = dict(out=str(tmp_path / "new-run"), device="cpu", resume="", warm_start="",
                  warm_start_q_scale=1.0, n_blocks=None, ntp_weight=None, lr=None,
                  trainer="train_fast", feature_dim=STRUCTURE_FEAT_DIM)
    values.update(changes)
    return SimpleNamespace(**values)


def migrated(tmp_path):
    torch.set_num_threads(1)
    torch.manual_seed(731)
    old = FableDanNet(config())
    source = tmp_path / "legacy.pt"
    save_ckpt(old, None, {"training_kind": "real_bc"}, str(source))
    model, opt, previous, origin = initialize_training(
        initialize_args(tmp_path, warm_start=str(source)))
    return old, model, opt, previous, origin, source


def test_feature_version_is_explicit_in_new_configs_and_legacy_defaults_to_v1():
    assert FEAT_DIM == 80 and STRUCTURE_FEAT_DIM == 224
    assert ModelConfig().feature_version == 1
    assert ModelConfig(feat_dim=224).feature_version == 2
    assert ModelConfig.from_dict({"feat_dim": 80}).feature_version == 1
    assert ModelConfig.from_dict({"feat_dim": 224, "feature_version": 2}).feat_dim == 224
    for values in ({"feat_dim": 224}, {"feat_dim": 80, "feature_version": 2},
                   {"feat_dim": 224, "feature_version": 1}, {"feat_dim": 81}):
        with pytest.raises(ValueError, match="feature"):
            ModelConfig.from_dict(values)


def test_zero_column_migration_preserves_all_weights_logits_and_order(tmp_path):
    old, model, opt, previous, origin, source = migrated(tmp_path)
    before_sha = hashlib.sha256(source.read_bytes()).hexdigest()
    for name, old_value in old.state_dict().items():
        value = model.state_dict()[name]
        if name == "hand_mlp.0.weight":
            torch.testing.assert_close(value[:, :80], old_value, rtol=0, atol=0)
            assert torch.count_nonzero(value[:, 80:]).item() == 0
        else:
            torch.testing.assert_close(value, old_value, rtol=0, atol=0)
    toks = torch.tensor([[1, 3, 15, 20, 36]])
    lengths = torch.tensor([5])
    feats = torch.randn(1, 21, 224)
    with torch.no_grad():
        old_q = old(toks, lengths, feats[:, :, :80])[0]
        new_q = model(toks, lengths, feats)[0]
    torch.testing.assert_close(new_q, old_q, atol=1e-6, rtol=1e-6)
    assert old_q.argmax().item() == new_q.argmax().item()
    assert not previous and not opt.state
    assert origin["feature_migration"]["method"] == "zero_pad_hand_mlp_first_layer"
    assert origin["source_feature_dim"] == 80 and model.cfg.feature_version == 2
    assert hashlib.sha256(source.read_bytes()).hexdigest() == before_sha


def test_new_feature_columns_receive_gradients_and_update(tmp_path):
    _, model, opt, _, _, _ = migrated(tmp_path)
    toks = torch.tensor([[1, 3, 15, 20, 36]])
    feats = torch.rand(1, 4, 224)
    q = model(toks, torch.tensor([5]), feats)[0]
    (q - 1).square().mean().backward()
    gradient = model.hand_mlp[0].weight.grad[:, 80:]
    assert torch.isfinite(gradient).all() and gradient.abs().sum().item() > 0
    opt.step()
    assert torch.count_nonzero(model.hand_mlp[0].weight[:, 80:]).item() > 0


def test_numpy_export_dimension_shape_validation_and_chunk_parity(tmp_path):
    _, model, _, _, _, _ = migrated(tmp_path)
    exported = tmp_path / "v2.npz"
    export_npz(model, str(exported))
    numpy_model = NumpyModel(exported)
    assert (numpy_model.feat_dim, numpy_model.feature_version) == (224, 2)
    toks = [1, 3, 15, 20, 36]
    feats = np.random.default_rng(8).normal(size=(29, 224)).astype(np.float32)
    with torch.no_grad():
        expected = model(torch.tensor([toks]), torch.tensor([len(toks)]),
                         torch.from_numpy(feats[None]))[0][0].numpy()
    np.testing.assert_allclose(numpy_model.q_values(toks, feats, candidate_chunk=7),
                               expected, atol=2e-5, rtol=2e-5)
    with pytest.raises(ValueError, match="feature shape"):
        numpy_model.q_values(toks, feats[:, :80])
    with np.load(exported, allow_pickle=False) as archive:
        arrays = dict(archive)
    bad_shape = dict(arrays)
    bad_shape["hand_mlp.0.weight"] = bad_shape["hand_mlp.0.weight"][:, :80]
    with pytest.raises(ValueError, match="shape mismatch"):
        NumpyModel(bad_shape)
    legacy_header = dict(arrays)
    legacy_header["__config__"] = np.array([item for item in arrays["__config__"]
                                          if not str(item).startswith("feature_version=")])
    with pytest.raises(ValueError, match="feature_version"):
        NumpyModel(legacy_header)


def test_legacy_numpy_header_without_version_remains_supported(tmp_path):
    source = tmp_path / "v1.npz"
    export_npz(FableDanNet(config()), str(source))
    with np.load(source, allow_pickle=False) as archive:
        arrays = dict(archive)
    arrays["__config__"] = np.array([item for item in arrays["__config__"]
                                     if not str(item).startswith("feature_version=")])
    assert NumpyModel(arrays).feature_version == 1


def test_resume_requires_matching_dimension_and_version(tmp_path):
    _, model, opt, _, _, _ = migrated(tmp_path)
    checkpoint = tmp_path / "dmc.pt"
    meta = {"training_kind": "dmc", "trainer": "train_fast", "cycle": 1,
            "total_samples": 16, "reward_schema": DMC_REWARD_SCHEMA,
            "feature_dim": 224, "feature_version": 2}
    save_ckpt(model, opt, meta, str(checkpoint))
    args = initialize_args(tmp_path, resume=str(checkpoint), warm_start_q_scale=None,
                           feature_dim=80)
    with pytest.raises(ValueError, match="differs from resume"):
        initialize_training(args)
    args.feature_dim = 224
    loaded, _, _, _ = initialize_training(args)
    assert loaded.cfg.feat_dim == 224
    meta["feature_version"] = 1
    save_ckpt(model, opt, meta, str(checkpoint))
    with pytest.raises(ValueError, match="metadata disagrees"):
        initialize_training(args)


def test_replay_v2_roundtrip_and_legacy_v1_restore_reject_cross_dimension():
    replay = Replay(3, feat_dim=224)
    replay.add([1, 3], np.arange(224, dtype=np.float32), 1 / 3)
    restored = Replay(3, feat_dim=224)
    restored.load_state_dict(replay.state_dict())
    np.testing.assert_array_equal(restored.feat[0], replay.feat[0])
    with pytest.raises(ValueError, match="feature dimension/version"):
        Replay(3).load_state_dict(replay.state_dict())
    with pytest.raises(ValueError, match="action features"):
        replay.add([1, 3], np.zeros(80), 0)
    legacy = Replay(3).state_dict()
    legacy["schema"] = "fabledan_replay_v1"
    legacy.pop("feat_dim")
    legacy.pop("feature_version")
    Replay(3).load_state_dict(legacy)
    with pytest.raises(ValueError, match="feature dimension/version"):
        Replay(3, feat_dim=224).load_state_dict(legacy)


def test_ring_and_model_agents_use_structure_features(tmp_path):
    _, model, _, _, _, _ = migrated(tmp_path)
    ring = RingRunner(ring=1, seed=21, ladder_frac=1, feature_dim=224)
    request = ring.collect_requests()[0]
    assert request[2].shape[1] == 224
    obs = ring.games[0].obs
    assert obs["feature_dim"] == 224
    npz = tmp_path / "model.npz"
    export_npz(model, str(npz))
    agents = [TorchAgent(model), NumpyAgent(NumpyModel(npz)),
              _ActorPolicy(model, eps=0, top_k=1, rng=random.Random(1))]
    for agent in agents:
        assert agent.feature_dim == 224
        assert 0 <= agent.act(obs) < len(obs["legal"])
    ring.step({0: np.zeros(len(obs["legal"]), dtype=np.float32)})
    assert ring.games[0].samples[-1][1].base is None
    assert agents[-1].samples[-1][1].base is None
    assert ring.collect_requests()[0][2].shape[1] == 224


def test_pack_uses_exported_dimension_and_bot_does_not_drop_candidates(tmp_path, monkeypatch):
    from botzone import bot_fabledan as bot
    _, model, _, _, _, _ = migrated(tmp_path)
    npz = tmp_path / "weights.npz"
    export_npz(model, str(npz))
    zip_path, _, _ = pack(npz, tmp_path / "bot.zip")
    with zipfile.ZipFile(zip_path) as archive:
        assert "fabledan/model_np.py" in archive.namelist()
    mirror = bot.Mirror()
    mirror.my_id, mirror.lv, mirror.hand = 0, 1, [8, 12]
    mirror.left = [2, 27, 27, 27]
    first = classify_claim([8], [8], 1)
    last = classify_claim([12], [12], 1)
    legal = [first] * 599 + [last]
    seen = {}

    def fake_generator(hand, level, lead, feature_dim):
        seen["generation_dim"] = feature_dim
        return legal

    class TailPreferringModel:
        cfg = {"feat_dim": 224}

        def q_values(self, tokens, features, candidate_chunk):
            seen["shape"] = features.shape
            seen["chunk"] = candidate_chunk
            return np.arange(len(features))

    monkeypatch.setattr(bot, "gen_moves", fake_generator)
    assert bot.choose_play(mirror, "transformer", TailPreferringModel()) == [[12], [12]]
    assert seen == {"generation_dim": 224, "shape": (600, 224), "chunk": 512}


def test_structure_fast_training_smoke_and_resume(tmp_path):
    torch.set_num_threads(1)
    source = tmp_path / "v1.pt"
    save_ckpt(FableDanNet(config()), None, {"training_kind": "real_bc"}, str(source))
    out = tmp_path / "structure-dmc"
    command = [sys.executable, "-m", "fabledan.train_fast", "--out", str(out),
               "--feature-dim", "224", "--device", "cpu", "--infer-device", "cpu",
               "--actors", "1", "--ring", "1", "--buffer", "16", "--diversity", "1",
               "--steps-per-cycle", "1", "--batch", "2", "--micro-batch", "2",
               "--eps", "0.15", "--top-k", "1", "--eval-cycles", "99", "--eval-games", "2",
               "--ckpt-cycles", "1", "--belief-weight", "0", "--export-cycles", "0",
               "--snapshot-cycles", "99", "--ladder-frac", "1", "--seed", "23",
               "--max-hours", "0.02"]
    first = subprocess.run(command + ["--cycles", "1", "--warm-start", str(source)],
                           cwd=ROOT / "competition", capture_output=True, text=True, timeout=90)
    assert first.returncode == 0, first.stdout + first.stderr
    checkpoint = torch.load(out / "latest.pt", map_location="cpu", weights_only=False)
    assert checkpoint["config"]["feat_dim"] == 224
    assert checkpoint["config"]["feature_version"] == 2
    assert checkpoint["meta"]["replay"]["feat_dim"] == 224
    assert checkpoint["meta"]["cycle"] == 1
    second = subprocess.run(command + ["--cycles", "2", "--resume", str(out / "latest.pt")],
                            cwd=ROOT / "competition", capture_output=True, text=True, timeout=90)
    assert second.returncode == 0, second.stdout + second.stderr
    assert torch.load(out / "latest.pt", map_location="cpu", weights_only=False)["meta"]["cycle"] == 2
