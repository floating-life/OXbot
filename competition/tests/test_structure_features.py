"""Physical-card structure features preserve the legacy contract and privacy."""
from collections import Counter
import copy
import hashlib
from pathlib import Path
import random
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fabledan.cards import SEQV_TO_RANK, is_wildcard, rank_of
from fabledan.combos import Move, PASS_MOVE, PAIR, SINGLE, SFLUSH, gen_moves
from fabledan.encode import FEAT_DIM, STRUCTURE_FEAT_DIM, encode_decision, hand_action_features


def observation(hand, moves=None, level=1):
    return {"hand": hand, "legal": moves if moves is not None else gen_moves(hand, level, None),
            "level": level, "player": 0, "left": [len(hand), 17, 12, 4],
            "done": [False] * 4, "lead": None, "lead_owner": None, "events": []}


def independent_structure(hand, action, level):
    """Small explicit oracle; computes windows without vectorization/caching."""
    remaining = list(hand)
    for card in action:
        remaining.remove(card)
    faces = Counter(card % 54 for card in hand)
    residual_faces = Counter(card % 54 for card in remaining)
    natural = Counter(rank_of(card) for card in remaining if not is_wildcard(card, level))
    wild = sum(is_wildcard(card, level) for card in remaining)
    result = [faces[face] / 2 for face in range(54)]
    result += [residual_faces[face] / 2 for face in range(54)]
    result += [natural[rank] / 4 for rank in range(15)]
    result += [wild / 2, len(remaining) / 27]
    result += [sum(natural[rank] >= k for rank in range(15)) / 15 for k in range(1, 9)]
    def count_windows(length, multiplicity, allowance, suit=None):
        total = 0
        for low in range(1, 16 - length):
            ranks = SEQV_TO_RANK[low:low + length]
            if suit is None:
                cost = sum(max(0, multiplicity - natural[rank]) for rank in ranks)
            else:
                suited = Counter(rank_of(card) for card in remaining
                                 if card % 54 < 52 and card % 54 % 4 == suit and not is_wildcard(card, level))
                cost = sum(max(0, multiplicity - suited[rank]) for rank in ranks)
            total += cost <= allowance
        return total
    result += [count_windows(5, 1, 0) / 10, count_windows(5, 1, wild) / 10]
    result += [sum(count_windows(5, 1, 0, suit) for suit in range(4)) / 40,
               sum(count_windows(5, 1, wild, suit) for suit in range(4)) / 40]
    result += [count_windows(3, 2, 0) / 12, count_windows(3, 2, wild) / 12,
               count_windows(2, 3, 0) / 13, count_windows(2, 3, wild) / 13]
    full = 0
    for triple in range(13):
        for pair in range(15):
            if triple == pair or not natural[triple] or not natural[pair]:
                continue
            if pair >= 13 and natural[pair] < 2:
                continue
            full += max(0, 3 - natural[triple]) + max(0, 2 - natural[pair]) <= wild
    result.append(full / 182)
    bombs = [min(natural[rank] + wild, 10) for rank in range(13)
             if natural[rank] and natural[rank] + wild >= 4]
    result += [max(bombs, default=0) / 10, float(natural[13] >= 2 and natural[14] >= 2)]
    return np.asarray(result, dtype=np.float32)


def test_same_legacy_move_suits_leave_different_straight_flush_options():
    # Two interchangeable rank-3 singles: only removing the diamond preserves
    # the heart 3..7 straight flush. Legacy rank features cannot distinguish it.
    hand = [8, 9, 12, 16, 20, 24]
    a, b = Move(SINGLE, 0, [8], [2]), Move(SINGLE, 0, [9], [2])
    obs = observation(hand, [a, b])
    _, old = encode_decision(obs)
    _, new = encode_decision(obs, feat_dim=STRUCTURE_FEAT_DIM)
    np.testing.assert_array_equal(old[0], old[1])
    assert not np.array_equal(new[0], new[1])
    assert new[0, 215] == 0
    assert new[1, 215] == pytest.approx(1 / 40)
    np.testing.assert_array_equal(new[:, :80], old)


def test_same_legacy_straight_flushes_preserve_different_suit_extensions():
    hearts = [8, 12, 16, 20, 24]
    diamonds = [9, 13, 17, 21, 25]
    moves = [Move(SFLUSH, 3, cards, [2, 3, 4, 5, 6]) for cards in (hearts, diamonds)]
    obs = observation(hearts + diamonds + [28], moves)
    _, old = encode_decision(obs)
    _, new = encode_decision(obs, feat_dim=224)
    np.testing.assert_array_equal(old[0], old[1])
    assert new[0, 215] == pytest.approx(1 / 40)
    assert new[1, 215] == pytest.approx(2 / 40)


def test_high_ace_level_wildcards_and_independent_window_semantics():
    hand = [36, 40, 44, 48, 0, 4]  # Heart 10 J Q K A, plus heart-2 wildcard.
    row = hand_action_features(observation(hand, [PASS_MOVE]), PASS_MOVE, feat_dim=224)
    assert row[213] == pytest.approx(1 / 10)  # 10..A
    assert row[214] == pytest.approx(2 / 10)  # 9..K with the wildcard too
    assert row[215] == pytest.approx(1 / 40)
    assert row[216] == pytest.approx(2 / 40)
    assert row[188 + 1] == 0  # the wildcard is excluded from natural level rank
    assert row[203] == 0.5
    # Both heart level cards are wild; a missing sequence position needs just
    # one each, but a single combo must never spend one wildcard twice.
    short = [8, 12, 4]
    row = hand_action_features(observation(short, [PASS_MOVE]), PASS_MOVE, 224)
    assert row[214] == 0 and row[216] == 0
    assert row[218] == 0 and row[220] == 0


def test_full_house_joker_attachment_wildcard_budget_and_rocket():
    for hand in ([8, 9, 12, 4], [8, 9, 12, 4, 58], [8, 9, 4, 52],
                 [8, 9, 4, 52, 106], [8, 9, 10, 4, 52, 106, 53, 107]):
        row = hand_action_features(observation(hand, [PASS_MOVE]), PASS_MOVE, 224)
        np.testing.assert_array_equal(row[80:], independent_structure(hand, [], 1))
    single_joker = hand_action_features(observation([8, 9, 4, 52], [PASS_MOVE]), PASS_MOVE, 224)
    assert single_joker[221] == 0  # a wild card cannot turn a singleton joker into a pair
    complete = hand_action_features(observation([8, 9, 10, 4, 52, 106, 53, 107], [PASS_MOVE]), PASS_MOVE, 224)
    assert complete[222] == pytest.approx(0.4)
    assert complete[223] == 1


def test_physical_removal_pass_and_rejection_of_unheld_or_duplicate_ids():
    hand = [8, 62, 12]
    obs = observation(hand, [PASS_MOVE, Move(SINGLE, 0, [62], [2])])
    _, rows = encode_decision(obs, feat_dim=224)
    np.testing.assert_array_equal(rows[0, 80:134], rows[0, 134:188])
    assert rows[1, 134 + 8] == 0.5
    assert rows[1, 188 + 2] == 0.25
    assert rows[1, 204] == pytest.approx(2 / 27)
    for cards in ([66], [8, 8]):
        with pytest.raises(ValueError, match="physical cards"):
            hand_action_features(obs, Move(PAIR, 0, cards, [2] * len(cards)), 224)
    # The face exists but the other deck's physical ID is not held.
    with pytest.raises(ValueError, match="physical cards"):
        hand_action_features(observation([8]), Move(SINGLE, 0, [62], [2]), 224)


def test_order_and_global_deck_swap_invariance_and_no_hidden_information():
    hand = [8, 9, 12, 16, 20, 24, 52, 106, 4]
    moves = [PASS_MOVE, Move(SINGLE, 0, [9], [2]), Move(SINGLE, 0, [4], [1])]
    obs = observation(hand, moves)
    original = encode_decision(obs, feat_dim=224)
    transformed = copy.deepcopy(obs)
    transformed["hand"] = [(card + 54) % 108 for card in reversed(hand)]
    transformed["legal"] = [Move(move.type, move.key, [(card + 54) % 108 for card in move.cards], move.claim_ranks)
                            for move in moves]
    transformed["opponent_hands"] = {1: list(range(108))}
    transformed["future"] = {"winner": 1, "hidden_cards": list(range(108))}
    assert original[0] == encode_decision(transformed, feat_dim=224)[0]
    np.testing.assert_array_equal(original[1], encode_decision(transformed, feat_dim=224)[1])


def test_fast_slow_and_independent_reference_across_levels_and_actions():
    rng = random.Random(3021)
    for level in range(13):
        hand = rng.sample(range(108), 20)
        legal = gen_moves(hand, level, None)
        selected = [PASS_MOVE] + rng.sample(legal, min(14, len(legal)))
        obs = observation(hand, selected, level)
        tokens, fast = encode_decision(obs, {}, feat_dim=224)
        legacy_tokens, legacy = encode_decision(obs)
        slow = np.stack([hand_action_features(obs, move, 224) for move in selected])
        np.testing.assert_array_equal(fast, slow)
        np.testing.assert_array_equal(fast[:, :80], legacy)
        assert tokens == legacy_tokens
        for move, row in zip(selected, fast):
            np.testing.assert_array_equal(row[80:], independent_structure(hand, move.cards, level))


def test_legacy_reference_digest_and_default_dimensions():
    # Golden legacy-80 fixture, independent of the slow path so changing both
    # public encoders cannot silently change this compatibility contract.
    rng = random.Random(5813)
    digest = hashlib.sha256()
    for level in (0, 1, 9, 12):
        hand = rng.sample(range(108), 27)
        obs = observation(hand, level=level)
        _, rows = encode_decision(obs)
        assert rows.shape[1] == FEAT_DIM == 80 and rows.dtype == np.float32
        digest.update(rows.tobytes())
    assert digest.hexdigest() == "52982e915e71ba70ccac9f82b27f1fb8fdcb081764aadfbd62a30aa6772ff840"
    assert STRUCTURE_FEAT_DIM == 224
    with pytest.raises(ValueError, match="feature dimension"):
        encode_decision(obs, feat_dim=81)
