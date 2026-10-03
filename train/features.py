"""Feature contract v1. Inputs are an acting player's information set only."""
from __future__ import annotations
import collections
import numpy as np

FEATURE_VERSION = "oxbot-observation-v1"
RANKS = "A234567890JQK"
KINDS = ("pass", "invalid", "single", "pair", "three", "straight", "set", "three_straight",
         "triple_pairs", "bomb", "straight_flush", "rocket")


def face_counts(cards):
    out = np.zeros(54, dtype=np.float32)
    for card in cards:
        if type(card) is not int or not 0 <= card < 108:
            raise ValueError("invalid BotZone card ID")
        out[card % 54] += .5
    return out


def state_features(observation):
    """54 hand + 54 public played + 4 remaining + 13 level + 3 context."""
    result = np.zeros(128, dtype=np.float32)
    result[:54] = face_counts(observation["hand"])
    history = observation.get("history", [])
    if "played_face_counts" in observation:
        counts = observation["played_face_counts"]
        if len(counts) != 54 or any(type(x) is not int or not 0 <= x <= 2 for x in counts):
            raise ValueError("invalid cumulative public played-face counts")
        result[54:108] = np.asarray(counts, dtype=np.float32) / 2
    else:
        result[54:108] = face_counts([c for event in history for c in event["action"]])
    player = observation["player"]
    counts = observation.get("remaining_counts", [27] * 4)
    for relative in range(4):
        count = counts[(player + relative) % 4]
        if not 0 <= count <= 27:
            raise ValueError("unknown or inconsistent remaining count")
        result[108 + relative] = count / 27
    level = "0" if observation["level"] == "10" else observation["level"]
    result[112 + RANKS.index(level)] = 1
    result[125] = float(observation["leading"])
    result[126] = observation.get("tribute", 0) / 2
    result[127] = float(observation.get("resist", False))
    return result


def history_tokens(observation, limit=256):
    """Raw public action/claim faces, relative seats, explicit pass/unknown."""
    raw = []
    player = observation["player"]
    for event in observation.get("history", []):
        raw.append(3 + (event["player"] - player) % 4)
        action = sorted(c % 54 for c in event["action"])
        if not action:
            raw.append(7)
        else:
            raw.extend(8 + c for c in action)
            raw.append(62)
            claim = event.get("claim")
            if claim is None:
                raw.append(117)
            else:
                raw.extend(63 + c for c in sorted(c % 54 for c in claim))
        raw.append(2)
    tokens = [1] + raw[-(limit - 2):] + [118]
    return np.asarray(tokens, dtype=np.int64)


def action_features(move, kind, key, secondary, level, hand_size):
    result = np.zeros(128, dtype=np.float32)
    action, claim = move
    result[:54] = face_counts(action)
    result[54:108] = face_counts(claim)
    result[108 + KINDS.index(kind)] = 1
    result[120] = len(action) / 10
    result[121] = max(0, key) / 14
    result[122] = max(0, secondary) / 14
    rank = RANKS.index("0" if level == "10" else level)
    result[123] = sum(c % 54 == rank * 4 for c in action) / 2
    result[124] = sum(c % 54 < 52 and (c % 54) // 4 == rank for c in action) / 8
    result[125] = float(kind in ("bomb", "straight_flush", "rocket"))
    result[126] = float(not action)
    result[127] = float(len(action) == hand_size and hand_size > 0)
    return result


def matching_actions(moves, demonstrated):
    """All valid claims for an observed action are positive BC labels.

    Duplicate deck IDs carry no information after their face is fixed. Claim
    ambiguity is retained explicitly; no guessed claim is treated as truth.
    """
    target = collections.Counter(c % 54 for c in demonstrated)
    return [i for i, move in enumerate(moves) if collections.Counter(c % 54 for c in move[0]) == target]
