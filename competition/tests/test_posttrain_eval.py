"""Paired evaluation statistics, tactical diagnostics, and frozen inference."""
from collections import Counter
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fabledan import posttrain_eval as pe
from fabledan.combos import gen_moves


def pair_rows(rewards):
    return [{"pair_id": i, "candidate_parity": parity, "pair_seed": i,
             "level": 1, "tribute_mode": None, "deal_sha256": f"deal-{i}",
             "team_score_difference": reward, "win": int(reward > 0)}
            for i, pair in enumerate(rewards) for parity, reward in enumerate(pair)]


def obs(hand, lead=None, owner=None, left=None, done=None):
    return {"player": 0, "level": 1, "hand": hand,
            "legal": gen_moves(hand, 1, lead), "lead": lead, "lead_owner": owner,
            "left": left or [len(hand), 8, 8, 8], "done": done or [False] * 4,
            "events": [] if lead is None else [("play", owner, lead)]}


def test_pair_bootstrap_keeps_anticorrelated_games_together():
    summary = pe.paired_summary(pair_rows([(3, -3), (1, -1), (2, -2)]), bootstrap_samples=100)
    assert summary["win_rate"] == 0.5
    assert summary["win_rate_ci"] == [0.5, 0.5]
    assert summary["team_score_difference_ci"] == [0.0, 0.0]


def test_paired_summary_rejects_unpaired_or_mismatched_deals():
    rows = pair_rows([(3, -3)])
    with pytest.raises(ValueError, match="both seat"):
        pe.paired_summary(rows[:1])
    rows[1]["deal_sha256"] = "different"
    with pytest.raises(ValueError, match="identical"):
        pe.paired_summary(rows)


def test_small_sample_gate_cannot_be_bypassed_by_lowering_min_pairs():
    summary = pe.paired_summary(pair_rows([(3, 3)] * 2))
    comparisons = {key: {"summary": summary} for key in ("bc_anchor", "rule")}
    gate = pe.promotion_gate(comparisons, min_pairs=1)
    assert not gate["passed"]
    assert gate["required_pairs_per_opponent"] == 200
    assert "bc_anchor_insufficient_pairs" in gate["reasons"]
    assert not gate["release_eligible"]


def test_gate_requires_confident_improvement_and_rule_score_floor():
    good = pe.paired_summary(pair_rows([(1, 1)] * 200), bootstrap_samples=1000)
    tied = pe.paired_summary(pair_rows([(1, -1)] * 200), bootstrap_samples=1000)
    weak = pe.paired_summary(pair_rows([(-1, -1)] * 200), bootstrap_samples=1000)
    assert pe.promotion_gate({"bc_anchor": {"summary": good}, "rule": {"summary": tied}})["passed"]
    assert not pe.promotion_gate({"bc_anchor": {"summary": tied}, "rule": {"summary": good}})["passed"]
    assert not pe.promotion_gate({"bc_anchor": {"summary": good}, "rule": {"summary": weak}})["passed"]


def test_pass_diagnostics_distinguish_partner_and_enemy_endgame():
    lead = gen_moves([44], 1, None)[0]  # Q
    friendly = obs([48, 49, 4], lead, 2, [3, 1, 3, 5])
    hostile = obs([48, 49, 4], lead, 1, [3, 1, 3, 5])
    metrics = Counter()
    pe.record_decision(metrics, friendly, 0)
    assert metrics["partner_control_passes"] == 1
    assert metrics["enemy_endgame_pass_with_ordinary_reply"] == 0
    pe.record_decision(metrics, hostile, 0)
    assert metrics["enemy_endgame_pass_with_ordinary_reply"] == 1
    assert not any("mistake" in key or "error" in key for key in metrics)


def test_rule_selfplay_is_reproducible_symmetric_and_deals_match():
    a = pe.evaluate_posttrain("rule", "rule", pairs=2, seed=17, batch_size=4, bootstrap_samples=100)
    b = pe.evaluate_posttrain("rule", "rule", pairs=2, seed=17, batch_size=1, bootstrap_samples=100)
    assert a["comparisons"] == b["comparisons"]
    for comparison in a["comparisons"].values():
        assert comparison["summary"]["mean_team_score_difference"] == 0
        assert comparison["summary"]["win_rate"] == 0.5
        for first, second in zip(comparison["games"][::2], comparison["games"][1::2]):
            assert first["deal_sha256"] == second["deal_sha256"]
            assert first["ranking"] == second["ranking"]
            assert first["team_score_difference"] == first["own_official_points"] - first["opponent_official_points"]
    assert a["total_games"] == 8
    assert not a["promotion_gate"]["passed"]


def test_mixed_partner_diagnostic_covers_four_seats_and_stays_out_of_gate():
    report = pe.evaluate_posttrain("rule", "rule", pairs=1, mixed_pairs=2,
                                  seed=17, batch_size=4, bootstrap_samples=100)
    mixed = report["mixed_partner_diagnostic"]
    assert mixed["measured"] and not mixed["promotion_gate_input"]
    assert mixed["summary"]["deal_blocks"] == 2
    assert mixed["summary"]["games"] == 8
    assert mixed["summary"]["mean_team_score_difference"] == 0
    assert mixed["summary"]["win_rate"] == 0.5
    assert mixed["summary"]["bootstrap_unit"] == "same_deal_four_candidate_seats"
    assert {row["candidate_seat"] for row in mixed["games"]} == {0, 1, 2, 3}
    assert report["total_games"] == 12
    assert not report["promotion_gate"]["passed"]


def test_stop_callback_and_invalid_policy_index():
    class Invalid:
        def act(self, observation):
            return -1
    with pytest.raises(ValueError, match="invalid action index"):
        pe.evaluate_posttrain(Invalid(), "rule", pairs=1, bootstrap_samples=100)
    class BadBatch:
        def act_batch(self, observations):
            return []
    with pytest.raises(ValueError, match="wrong number"):
        pe.evaluate_posttrain(BadBatch(), "rule", pairs=1, bootstrap_samples=100)
    def stop():
        raise RuntimeError("cancelled")
    with pytest.raises(RuntimeError, match="cancelled"):
        pe.evaluate_posttrain("rule", "rule", pairs=1, check_stop=stop, bootstrap_samples=100)


def test_torch_batch_numpy_parity_and_snapshot_does_not_mutate_learner(tmp_path):
    torch = pytest.importorskip("torch")
    from fabledan.model_torch import FableDanNet, ModelConfig, export_npz, save_ckpt
    from fabledan.model_np import NumpyModel
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(21)
        model = FableDanNet(ModelConfig(d_model=8, n_blocks=1, n_heads=1,
                                       qk_dim=4, v_dim=4, ffn_hidden=16,
                                       hand_hidden=16, n_hand_layers=1, q_hidden=16, n_q_layers=1))
    model.train()
    source = tmp_path / "tiny.npz"
    export_npz(model, source)
    numpy_policy = pe.make_policy(NumpyModel(source))
    torch_policy = pe.make_policy(model)
    assert model.training and all(parameter.requires_grad for parameter in model.parameters())
    observations = [obs([8, 9, 10, 12, 13, 16, 20]), obs([48, 49, 4])]
    expected = numpy_policy.act_batch(observations)
    assert torch_policy.act_batch(observations) == expected
    assert [torch_policy.act(item) for item in observations] == expected
    assert pe.make_policy(source, backend="torch").act_batch(observations) == expected
    pt_source = tmp_path / "tiny.pt"
    save_ckpt(model, None, {}, pt_source)
    pt_policy = pe.make_policy(pt_source)
    assert pt_policy.act_batch(observations) == expected
    assert len(pt_policy.source["sha256"]) == 64
    assert pt_policy.source["model_sha256"] == torch_policy.source["model_sha256"]
    frozen = [parameter.detach().clone() for parameter in torch_policy.model.parameters()]
    with torch.no_grad():
        next(model.parameters()).add_(10)
    assert all(torch.equal(old, new) for old, new in zip(frozen, torch_policy.model.parameters()))
    assert torch_policy.act_batch(observations) == expected
