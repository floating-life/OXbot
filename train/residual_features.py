"""Candidate-local residual features (development audit only).

The feature in this module is deliberately information-set local.  Given an
actor hand ``H`` and one already-legal candidate ``m``, it removes only the
physical cards in ``m.action`` and asks the C++ rule core for *lead* moves
from the residual hand.  It therefore measures the actor's future lead
flexibility, not an opponent response and not a counterfactual Q value.

This module does not alter the release feature contract (v1) or the anchor
model.  A caller must supply an offline :class:`tools.probe.Probe` instance;
the probe is never part of the BotZone package.
"""
from __future__ import annotations

from collections import Counter
import math
from typing import Any, Iterable, Mapping, Sequence


RESIDUAL_FEATURE_VERSION = "oxbot-candidate-residual-v1"
KIND_ORDER = (
    "single", "pair", "three", "straight", "set", "three_straight",
    "triple_pairs", "bomb", "straight_flush", "rocket",
)
POWER_KINDS = frozenset(("bomb", "straight_flush", "rocket"))


def _cards(value: Any, name: str) -> list[int]:
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{name} must be a card sequence")
    cards = list(value)
    if any(type(card) is not int or not 0 <= card < 108 for card in cards):
        raise ValueError(f"{name} contains an invalid card id")
    return cards


def residual_hand(hand: Sequence[int], action: Sequence[int]) -> list[int]:
    """Remove an action's physical IDs exactly once, preserving no hidden data."""
    remaining = _cards(hand, "hand")
    for card in _cards(action, "action"):
        try:
            remaining.remove(card)
        except ValueError as exc:
            raise ValueError("candidate action is not a sub-multiset of hand") from exc
    return sorted(remaining)


def _move_parts(move: Any) -> tuple[list[int], list[int]]:
    if not isinstance(move, (list, tuple)) or len(move) != 2:
        raise ValueError("probe move must be [action, claim]")
    return _cards(move[0], "move.action"), _cards(move[1], "move.claim")


def _kind(meta: Any, move: Any) -> str:
    if isinstance(meta, Mapping) and isinstance(meta.get("kind"), str):
        return meta["kind"]
    # This fallback is intentionally conservative.  The C++ probe should
    # always return metadata; an unknown kind is not silently counted as a
    # power move.
    del move
    return "unknown"


def summarize_lead_moves(moves: Sequence[Any], types: Sequence[Any]) -> dict[str, Any]:
    """Summarize a C++ ``generate(..., leading=true, metadata=true)`` result."""
    if len(moves) != len(types):
        raise ValueError("probe moves/types length mismatch")
    counts: Counter[str] = Counter()
    bomb_lengths: list[int] = []
    for move, meta in zip(moves, types):
        action, claim = _move_parts(move)
        # A lead generation must not contain pass.  Rejecting it here keeps a
        # future probe change from silently changing the feature definition.
        if not action:
            raise ValueError("lead move list unexpectedly contains pass")
        kind = _kind(meta, move)
        if kind == "pass":
            raise ValueError("lead move metadata unexpectedly says pass")
        counts[kind] += 1
        if kind == "bomb":
            bomb_lengths.append(len(claim))
    total = int(sum(counts.values()))
    power = int(sum(counts[kind] for kind in POWER_KINDS))
    return {
        "total": total,
        "power": power,
        "nonpower": total - power,
        "by_kind": {kind: int(counts[kind]) for kind in KIND_ORDER},
        "unknown": int(counts["unknown"]),
        "bomb_max_length": max(bomb_lengths, default=0),
        "has_rocket": bool(counts["rocket"]),
    }


def _wildcards(cards: Iterable[int], level: str) -> int:
    # Card IDs use deck-index rank/suit; heart is suit zero.  The level value
    # follows the established v1 contract (10 is represented as 0).
    normalized = "0" if level == "10" else level
    ranks = "A234567890JQK"
    if normalized not in ranks:
        raise ValueError("unknown level")
    rank = ranks.index(normalized)
    return sum(1 for card in cards if card % 54 < 52 and card % 4 == 0 and card // 4 == rank)


def _rank_multiplicity(cards: Iterable[int]) -> tuple[int, int, int]:
    counts: Counter[int] = Counter()
    for card in cards:
        face = card % 54
        if face < 52:
            counts[face // 4] += 1
    return (sum(value >= 2 for value in counts.values()),
            sum(value >= 3 for value in counts.values()),
            sum(value >= 4 for value in counts.values()))


def residual_features_from_summaries(hand: Sequence[int], action: Sequence[int], level: str,
                                     base: Mapping[str, Any], residual: Mapping[str, Any]) -> dict[str, Any]:
    """Build stable raw/derived fields after summaries were obtained by Probe.

    Keeping this part probe-independent makes it cheap to unit test and makes
    the audit artifact reproducible if the C++ probe is rebuilt.
    """
    left = residual_hand(hand, action)
    total0 = int(base["total"])
    total1 = int(residual["total"])
    if total0 < 0 or total1 < 0:
        raise ValueError("negative lead-move count")
    ge2, ge3, ge4 = _rank_multiplicity(left)
    result: dict[str, Any] = {
        "schema": RESIDUAL_FEATURE_VERSION,
        "residual_hand_size": len(left),
        "candidate_action_size": len(action),
        "terminal_after_action": not left,
        "residual_wildcards": _wildcards(left, level),
        "residual_ranks_ge2": ge2,
        "residual_ranks_ge3": ge3,
        "residual_ranks_ge4": ge4,
        "base_lead_total": total0,
        "residual_lead_total": total1,
        "residual_lead_power": int(residual["power"]),
        "residual_lead_nonpower": int(residual["nonpower"]),
        "residual_bomb_max_length": int(residual["bomb_max_length"]),
        "residual_has_rocket": bool(residual["has_rocket"]),
        "lead_total_delta": total1 - total0,
        "lead_log_total_delta": math.log1p(total1) - math.log1p(total0),
        "residual_by_kind": dict(residual["by_kind"]),
    }
    # No arbitrary clipping is applied in the audit sidecar.  Training can
    # later choose an explicit normalization after checking validation ranges.
    return result


def residual_features(probe: Any, hand: Sequence[int], level: str,
                      action: Sequence[int], base_summary: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Query the offline C++ probe and return one candidate-local feature row."""
    hand = _cards(hand, "hand")
    action = _cards(action, "action")
    base = base_summary
    if base is None:
        generated = probe.call(command="generate", hand=sorted(hand), level=level,
                               leading=True, metadata=True)
        if "error" in generated:
            raise ValueError(f"base lead generation failed: {generated['error']}")
        base = summarize_lead_moves(generated.get("moves", []), generated.get("types", []))
    left = residual_hand(hand, action)
    generated = probe.call(command="generate", hand=left, level=level,
                           leading=True, metadata=True)
    if "error" in generated:
        raise ValueError(f"residual lead generation failed: {generated['error']}")
    after = summarize_lead_moves(generated.get("moves", []), generated.get("types", []))
    return residual_features_from_summaries(hand, action, level, base, after)
