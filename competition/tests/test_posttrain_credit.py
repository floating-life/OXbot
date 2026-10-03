"""Counterfactual credit must preserve the live game and deployment contract."""
import copy
import hashlib
import json
from pathlib import Path
import random
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fabledan.agents import RuleAgent
from fabledan.combos import SFLUSH, gen_moves
from fabledan.encode import encode_decision
from fabledan.engine import GuandanRound
from fabledan.posttrain_credit import (SCHEMA, act_batch, collect_reviews, fit_reviews,
                                      preference_pairs, review_position,
                                      select_alternatives, validate_reviews)


def move_key(move):
    if move is None:
        return None
    return move.type, move.key, tuple(move.cards), tuple(move.claim_ranks)


def event_keys(events):
    return [(kind, player, move_key(value[0]) if kind == "play" else tuple(value))
            for kind, player, *value in events]


def round_state(rnd):
    return (copy.deepcopy(rnd.hands), event_keys(rnd.events), list(rnd.done_order),
            rnd.lead_player, rnd.resist, rnd.rng.getstate())


def finish_from(gen, observation, policy=None):
    policy = policy or RuleAgent()
    for _ in range(1024):
        try:
            observation = gen.send(policy.act(observation))
        except StopIteration as terminal:
            return terminal.value
    raise AssertionError("fixture continuation failed to terminate")


@pytest.mark.parametrize("tribute", [None, ("single", 1, 0), ("double", 3, 0)])
def test_fork_replays_identically_without_touching_original_or_repeating_tribute(tribute, monkeypatch):
    # Both big jokers belong to seat zero, so the paying team cannot resist.
    deck = [53, 107] + [card for card in range(108) if card not in (53, 107)]
    deal = [deck[index * 27:(index + 1) * 27] for index in range(4)]
    rnd = GuandanRound(1, random.Random(21), tribute, deal=deal)
    gen = rnd.play_steps()
    observation = next(gen)
    for _ in range(3):
        observation = gen.send(RuleAgent().act(observation))
    before = round_state(rnd)
    exchanges = [event for event in rnd.events if event[0] in ("tribute", "return")]
    assert len(exchanges) == (0 if tribute is None else 2 if tribute[0] == "single" else 4)

    def reject_repeated_exchange(self):
        raise AssertionError("fork must not execute tribute a second time")

    monkeypatch.setattr(GuandanRound, "_do_tribute", reject_repeated_exchange)
    clone, branch = rnd.fork_decision(observation)
    initial = next(branch)
    assert [move_key(move) for move in initial["legal"]] == [move_key(move) for move in observation["legal"]]
    assert initial["hand"] == observation["hand"]
    assert initial["done"] == observation["done"]
    assert clone.events is not rnd.events
    assert all(left is not right for left, right in zip(clone.hands, rnd.hands))
    branch_result = finish_from(branch, initial)
    assert round_state(rnd) == before
    assert [event for event in clone.events if event[0] in ("tribute", "return")] == exchanges

    original_result = finish_from(gen, observation)
    assert branch_result == original_result
    assert clone.hands == rnd.hands
    assert clone.done_order == rnd.done_order
    assert event_keys(clone.events) == event_keys(rnd.events)


def test_fork_preserves_finished_player_and_live_comparison_target():
    rnd = GuandanRound(12, deal=[[0, 4], [8], [12], [16]])
    gen = rnd.play_steps()
    opening = next(gen)
    ace = next(index for index, move in enumerate(opening["legal"]) if move.cards == [0])
    observation = gen.send(ace)
    assert rnd.done_order == [0]
    assert observation["done"][0] and observation["lead_owner"] == 0
    before = round_state(rnd)
    clone, branch = rnd.fork_decision(observation)
    initial = next(branch)
    assert clone.done_order == [0]
    assert initial["player"] == observation["player"]
    assert move_key(initial["lead"]) == move_key(observation["lead"])
    assert initial["done"] == observation["done"]
    branch_result = finish_from(branch, initial)
    assert round_state(rnd) == before
    assert branch_result == finish_from(gen, observation)
    assert event_keys(clone.events) == event_keys(rnd.events)


class PublicOnlyRule:
    """A policy assertion guards every continuation observation, not just output."""

    def __init__(self):
        self.calls = 0

    def act(self, observation):
        assert set(observation) == {"player", "level", "hand", "legal", "lead",
                                    "lead_owner", "events", "done", "left", "feature_dim"}
        assert all(set(move.cards).issubset(observation["hand"]) for move in observation["legal"])
        assert len(observation["left"]) == len(observation["done"]) == 4
        self.calls += 1
        return RuleAgent().act(observation)


def tiny_review(hidden=((8,), (12,), (16,))):
    # At level K, leading A retains control and scores +2; leading 2 lets the
    # opponents finish first and scores -2 under the deterministic rule policy.
    rnd = GuandanRound(12, deal=[[0, 4]] + [list(hand) for hand in hidden])
    gen = rnd.play_steps()
    observation = next(gen)
    before = round_state(rnd)
    chosen = next(index for index, move in enumerate(observation["legal"]) if move.cards == [4])
    policy = PublicOnlyRule()
    row = review_position(rnd, observation, chosen, [("rule", [policy] * 4)],
                          max_candidates=8, seed=3)
    assert round_state(rnd) == before
    assert policy.calls > 0
    gen.close()
    return row, observation


def test_counterfactual_uses_actual_terminal_team_margin_and_public_inputs_only():
    row, observation = tiny_review()
    values = {tuple(move["cards"]): result[0] for move, result in zip(row["actions"], row["returns"])}
    assert values == pytest.approx({(4,): -2 / 3, (0,): 2 / 3})
    assert row["chosen"] == 0 and row["actions"][0]["cards"] == [4]
    assert row["best_reviewed"] == 1
    assert row["estimated_regret"] == pytest.approx(4 / 3)
    assert all(set(move["cards"]).issubset(observation["hand"]) for move in row["actions"])
    assert not {"hands", "allocations", "allocation", "deal", "hidden_hands"}.intersection(row)

    # Two private worlds have identical acting-player information. Only their
    # oracle targets may differ; neither private allocation enters the input.
    other, _ = tiny_review(hidden=((16,), (12,), (8,)))
    assert other["tokens"] == row["tokens"]
    np.testing.assert_array_equal(other["features"], row["features"])
    assert other["actions"] == row["actions"]
    assert not np.array_equal(other["returns"], row["returns"])


def test_counterfactual_rejects_missing_scenarios():
    rnd = GuandanRound(12, deal=[[0, 4], [8], [12], [16]])
    gen = rnd.play_steps()
    observation = next(gen)
    with pytest.raises(ValueError, match="scenario"):
        review_position(rnd, observation, 0, [])
    gen.close()


def test_feature_equivalent_physical_moves_are_not_contradictory_alternatives():
    hand = [0, 4, 8, 12, 16, 1, 5, 9, 13, 17, 20, 24]
    observation = {"player": 0, "level": 8, "hand": hand,
                   "legal": gen_moves(hand, 8), "lead": None, "lead_owner": None,
                   "events": [], "done": [False] * 4, "left": [12, 27, 27, 27]}
    _, features = encode_decision(observation)
    equivalent = [index for index, move in enumerate(observation["legal"])
                  if move.type == SFLUSH and move.key == 1]
    assert len(equivalent) == 2
    np.testing.assert_array_equal(features[equivalent[0]], features[equivalent[1]])
    selected = select_alternatives(observation, equivalent[1], 512, random.Random(4))
    assert selected[0] == equivalent[1]
    assert sum(index in selected for index in equivalent) == 1
    assert len({features[index].tobytes() for index in selected}) == len(selected)


def test_preference_requires_robust_gap_without_scenario_reversal():
    # A wins on average against B, but B wins in scenario two: discard A>B.
    returns = [[1.0, -1 / 3], [0.0, 1 / 3], [-1.0, -1.0], [1.0, -1 / 3]]
    better, worse = preference_pairs(returns)
    pairs = set(zip(better.tolist(), worse.tolist()))
    assert (0, 1) not in pairs and (1, 0) not in pairs
    assert {(0, 2), (1, 2), (3, 2)}.issubset(pairs)
    assert (0, 3) not in pairs and (3, 0) not in pairs
    near_tie = preference_pairs([[0.1, 0.1], [0.0, 0.0]])
    assert len(near_tie[0]) == 0


@pytest.fixture(scope="module")
def collected_reviews():
    return collect_reviews(RuleAgent(), RuleAgent(), games=2, positions_per_game=2,
                           max_candidates=2, seed=51000, ladder_frac=1.0)


def test_collected_splits_keep_whole_source_games_together(collected_reviews):
    validate_reviews(collected_reviews)
    assert collected_reviews["release_eligible"] is False
    groups = {}
    for row in collected_reviews["rows"]:
        groups.setdefault(row["game_seed"], set()).add(row["split"])
        assert len({np.asarray(feat, dtype=np.float32).tobytes() for feat in row["features"]}) == len(row["features"])
    assert len(groups) == 2
    assert all(len(splits) == 1 for splits in groups.values())
    assert {next(iter(splits)) for splits in groups.values()} == {"train", "validation"}

    leaked = copy.deepcopy(collected_reviews)
    train = next(row for row in leaked["rows"] if row["split"] == "train")
    validation = next(row for row in leaked["rows"] if row["split"] == "validation")
    validation["game_seed"], validation["decision"] = train["game_seed"], 99999
    with pytest.raises(ValueError, match="disjoint.*source games"):
        validate_reviews(leaked)
    duplicate = copy.deepcopy(collected_reviews)
    duplicate["rows"].append(copy.deepcopy(duplicate["rows"][0]))
    with pytest.raises(ValueError, match="duplicate source decision"):
        validate_reviews(duplicate)


def test_import_rejects_conflicting_labels_for_identical_features(collected_reviews):
    corrupt = copy.deepcopy(collected_reviews)
    row = corrupt["rows"][0]
    assert len(row["features"]) >= 2
    row["features"][1] = list(row["features"][0])
    row["returns"][0] = [-1.0] * len(row["scenario_names"])
    row["returns"][1] = [1.0] * len(row["scenario_names"])
    with pytest.raises(ValueError, match="feature|indistinguishable|duplicate"):
        validate_reviews(corrupt)


@pytest.mark.parametrize("bad", [True, False, 0.0, 1.5, -1, 2])
def test_invalid_policy_action_indices_fail_closed(bad):
    class BadPolicy:
        def act(self, observation):
            return bad

    with pytest.raises(ValueError, match="action index"):
        act_batch(BadPolicy(), [{"legal": [None, None]}])


def test_tiny_cpu_fit_preserves_source_and_exports_numpy_model(tmp_path):
    torch = pytest.importorskip("torch")
    from fabledan.model_np import NumpyModel
    from fabledan.model_torch import FableDanNet, ModelConfig, load_ckpt, save_ckpt

    torch.manual_seed(31)
    cfg = ModelConfig(d_model=8, n_blocks=1, n_heads=1, qk_dim=4, v_dim=4,
                      ffn_hidden=16, hand_hidden=16, n_hand_layers=1,
                      q_hidden=16, n_q_layers=1)
    initial = tmp_path / "initial.pt"
    save_ckpt(FableDanNet(cfg), None, {"training_kind": "real_bc"}, str(initial))
    initial_sha = hashlib.sha256(initial.read_bytes()).hexdigest()
    row, _ = tiny_review()
    train, validation = copy.deepcopy(row), copy.deepcopy(row)
    train.update(game_seed=71, decision=1, split="train")
    validation.update(game_seed=72, decision=1, split="validation")
    reviews = tmp_path / "reviews.json"
    reviews.write_text(json.dumps({"schema": SCHEMA, "rows": [train, validation]}), encoding="utf-8")
    output = tmp_path / "credit"
    report = fit_reviews(initial, reviews, output, epochs=2, lr=1e-3,
                         device="cpu", q_scale=0.1, seed=32)
    assert hashlib.sha256(initial.read_bytes()).hexdigest() == initial_sha
    assert report["source_sha256"] == initial_sha
    assert report["release_eligible"] is False
    assert report["train_positions"] == report["validation_positions"] == 1
    assert report["preference_pairs"] == 1
    assert all(np.isfinite(item["validation_loss"]) for item in report["epochs"])
    assert report["candidate_sha256"] == hashlib.sha256((output / "best.npz").read_bytes()).hexdigest()
    for name in ("best.pt", "best.npz", "latest.pt", "latest.npz", "training.json"):
        assert (output / name).is_file()

    model, checkpoint = load_ckpt(str(output / "best.pt"), device="cpu")
    assert checkpoint["meta"]["training_kind"] == "posttrain_credit"
    assert model.cfg.to_dict() == cfg.to_dict()
    exported = NumpyModel(str(output / "best.npz"))
    features = np.asarray(row["features"], dtype=np.float32)
    scores = exported.q_values(row["tokens"], features)
    with torch.no_grad():
        expected, _ = model(torch.tensor([row["tokens"]]),
                            torch.tensor([len(row["tokens"])]),
                            torch.tensor(features[None]))
    assert scores.shape == (len(features),) and np.isfinite(scores).all()
    np.testing.assert_allclose(scores, expected[0].numpy(), atol=1e-5, rtol=1e-5)
