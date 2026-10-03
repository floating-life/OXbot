from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from generate_selfplay import canonical_move, game_spec, make_init, move_key, response_for_move  # noqa: E402


def test_move_normalization_and_matching_key_are_protocol_safe():
    assert canonical_move([]) == [[], []]
    assert response_for_move([[4, 1], [4, 1]], "play") == [[4, 1], [4, 1]]
    assert response_for_move([[4], [4]], "tribute") == [4]
    assert move_key([[4, 1], [4, 1]]) == move_key([[1, 4], [1, 4]])


def test_game_spec_matches_seeded_first_last_contract():
    first = game_spec(0, 20261020)
    second = game_spec(0, 20261020)
    assert first == second
    init = make_init(first)
    assert init["first"] != init["last"]
    assert init["seed"] == str(first["seed"])

