"""Bounded candidate diversity must preserve suit choices and legacy semantics."""
from pathlib import Path
import random
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "competition"))
sys.path.insert(0, str(ROOT / "competition/tools"))

from fabledan.combos import PASS, SFLUSH, SINGLE, STRUCTURE_CANDIDATE_BUDGET, beats, claim_ids, classify_claim, gen_moves
from fabledan.encode import encode_decision
from check_structure_cpp import build_cases, candidate_key, compare_case, fixture_hands, semantic_key


def observation(hand, moves, level=1):
    return {"player": 0, "level": level, "hand": hand, "legal": moves,
            "lead": None, "lead_owner": None, "left": [len(hand), 27, 27, 27],
            "done": [False] * 4, "events": []}


def test_structure_preserves_double_straight_flush_features_that_legacy_merges():
    _, level, hand = fixture_hands()[0]
    moves = gen_moves(hand, level, feature_dim=224)
    selected = [move for move in moves if move.type == SFLUSH and move.key == 3]
    assert len(selected) == 2
    obs = observation(hand, selected)
    _, legacy = encode_decision(obs)
    _, structure = encode_decision(obs, feat_dim=224)
    np.testing.assert_array_equal(legacy[0], legacy[1])
    assert len({row.tobytes() for row in legacy}) == 1
    assert len({row.tobytes() for row in structure}) == 2
    assert sorted(structure[:, 215]) == pytest.approx([1 / 40, 2 / 40])


def test_structure_preserves_distinct_suit_singles():
    _, level, hand = fixture_hands()[1]
    old = gen_moves(hand, level, feature_dim=80)
    new = gen_moves(hand, level, feature_dim=224)
    old_threes = [move for move in old if move.type == SINGLE and move.claim_ranks == [2]]
    new_threes = [move for move in new if move.type == SINGLE and move.claim_ranks == [2]]
    assert len(old_threes) == 1 and len(new_threes) == 2
    assert {tuple(move.cards) for move in new_threes} == {(8,), (9,)}


def test_duplicate_deck_faces_do_not_create_duplicate_candidates():
    _, level, hand = fixture_hands()[2]
    moves = gen_moves(hand, level, feature_dim=224)
    keys = list(map(candidate_key, moves))
    assert len(keys) == len(set(keys))
    singles = [key[-1] for move, key in zip(moves, keys) if move.type == SINGLE]
    assert len(singles) == len(set(card % 54 for card in hand))
    swapped = gen_moves([(card + 54) % 108 for card in hand], level, feature_dim=224)
    assert keys == list(map(candidate_key, swapped))


def test_candidate_budget_never_discards_an_entire_rank_semantic():
    _, level, hand = fixture_hands()[-1]
    legacy = gen_moves(hand, level, feature_dim=80)
    moves = gen_moves(hand, level, feature_dim=224)
    expected = set(map(semantic_key, legacy))
    assert STRUCTURE_CANDIDATE_BUDGET == 256
    assert len(moves) == 256
    assert expected <= set(map(semantic_key, moves))
    assert len(moves) <= max(256, len(set(map(semantic_key, moves))))


def test_all_levels_legal_roundtrip_and_order_invariance():
    levels, modes = set(), set()
    for case in build_cases(seed=20261007, deals_per_level=1):
        obs = case["observation"]
        levels.add(obs["level"])
        modes.add(obs["lead"] is None)
        assert case["python_invariance_passed"]
        keys = list(map(candidate_key, case["moves"]))
        assert keys == sorted(keys) and len(keys) == len(set(keys))
        for move in case["moves"]:
            replay = classify_claim(move.cards, claim_ids(move), obs["level"])
            assert candidate_key(move) == candidate_key(replay)
            assert len(move.cards) == len(set(move.cards))
            assert set(move.cards) <= set(obs["hand"])
            assert obs["lead"] is not None if move.type == PASS else beats(move, obs["lead"], obs["level"])
    assert levels == set(range(13)) and modes == {False, True}


def test_checker_detects_candidate_order_and_feature_misalignment():
    case = next(build_cases(seed=11, deals_per_level=1))
    actual = {"moves": [[list(move.cards), claim_ids(move)] for move in case["moves"]],
              "actions": case["features"].tolist()}
    assert compare_case(case, actual)["passed"]
    actual["actions"][0][216] += 0.25
    mismatch = compare_case(case, actual)
    assert not mismatch["passed"]
    assert mismatch["worst_feature_misalignment"]["feature_column"] == 216
    assert mismatch["worst_feature_misalignment"]["abs_error"] == pytest.approx(0.25)
    actual["actions"] = case["features"].tolist()
    actual["moves"][0], actual["moves"][1] = actual["moves"][1], actual["moves"][0]
    mismatch = compare_case(case, actual)
    assert not mismatch["passed"] and not mismatch["ordered_candidates_match"]
    assert mismatch["candidate_mismatch_samples"][0]["index"] == 0


def test_checker_rejects_legacy_feature_shape_and_nonwild_claim_changes():
    case = next(build_cases(seed=11, deals_per_level=1))
    actual = {"moves": [[list(move.cards), claim_ids(move)] for move in case["moves"]],
              "actions": case["features"][:, :80].tolist()}
    mismatch = compare_case(case, actual)
    assert not mismatch["passed"] and "224-dimensional" in mismatch["error"]
    actual["actions"] = case["features"].tolist()
    actual["moves"][0][1] = [52] * len(actual["moves"][0][0])
    mismatch = compare_case(case, actual)
    assert not mismatch["passed"] and "claim_" in mismatch["error"]
