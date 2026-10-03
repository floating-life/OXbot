"""Python reference for the frozen candidate-selection experiments.

The BotZone submission implements the same descriptor and stable tie rules in
``core/src/policy.cpp``.  This module is intentionally dependency-free so a
parity check can run beside the C++ probe without opening the held-out split.
"""
from __future__ import annotations

import math
from collections import OrderedDict


SELECTION_SCHEMA = "oxbot-selection-v1"
RAW = "raw"
RAW_PASS_BIAS = "raw-pass-bias"
GROUP_LOGMEANEXP = "group-logmeanexp"
SUPPORTED = (RAW, RAW_PASS_BIAS, GROUP_LOGMEANEXP)


def group_key(move, metadata, level):
    """Build the canonical strategic group key used by C++.

    ``move`` is ``[action, claim]`` and ``metadata`` has C++ classify fields
    ``kind``, ``key`` and ``secondary``.  Physical cards are reduced to their
    two-deck face IDs; non-straight-flush suits are intentionally marginalized.
    """
    action = move[0]
    faces = [0] * 54
    for card in action:
        if type(card) is not int or not 0 <= card < 108:
            raise ValueError("invalid BotZone card ID")
        face = int(card) % 54
        faces[face] += 1
    rank_counts = [sum(faces[4 * rank:4 * rank + 4]) for rank in range(13)]
    rank_counts.extend((faces[52], faces[53]))
    normalized = "0" if level == "10" else level
    ranks = "A234567890JQK"
    rank = ranks.index(normalized)
    wild_count = faces[4 * rank]
    kind = metadata["kind"]
    signature = tuple(faces) if kind == "straight_flush" else None
    return (kind, len(action), int(metadata["key"]), int(metadata["secondary"]),
            tuple(rank_counts), wild_count, signature)


def _logmeanexp(values):
    values = [float(value) for value in values]
    if not values:
        raise ValueError("empty group")
    largest = max(values)
    return largest + math.log(sum(math.exp(value - largest) for value in values) / len(values))


def choose_group_logmeanexp(moves, scores, metadata, level):
    """Return (chosen candidate index, groups) with C++-matching tie rules."""
    if len(moves) != len(scores) or len(moves) != len(metadata) or not moves:
        raise ValueError("selection shape mismatch")
    groups = OrderedDict()
    for index, (move, meta) in enumerate(zip(moves, metadata, strict=True)):
        groups.setdefault(group_key(move, meta, level), []).append(index)
    best_group = None
    best_score = -math.inf
    for key, members in groups.items():
        value = _logmeanexp([scores[index] for index in members])
        if value > best_score:
            best_score, best_group = value, key
    members = groups[best_group]
    chosen = members[0]
    for index in members:
        if float(scores[index]) > float(scores[chosen]):
            chosen = index
    return chosen, groups


def choose(moves, scores, metadata, level, strategy=RAW):
    if strategy == RAW or strategy == RAW_PASS_BIAS:
        bias = -0.5 if strategy == RAW_PASS_BIAS else 0.0
        return max(range(len(scores)), key=lambda index: float(scores[index]) + (bias if not moves[index][0] else 0.0))
    if strategy == GROUP_LOGMEANEXP:
        return choose_group_logmeanexp(moves, scores, metadata, level)[0]
    raise ValueError(f"unknown selection strategy: {strategy}")
