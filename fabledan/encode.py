# -*- coding: utf-8 -*-
"""State/action encoding shared by training (torch) and inference (numpy).

Token vocabulary (size 48):
  0  PAD
  1  BOS
  2..14   level token (level rank 0..12)
  15..18  player token (relative seat: 0=self, 1=next, 2=partner, 3=prev)
  19..29  move-type tokens (PASS..ROCKET, see combos)
  30  TRIBUTE   31  RETURN
  32..46  rank tokens (A..K, sj, BJ)
  47  (reserved)

A play event emits  [P, TYPE, rank...claim ranks sorted].
A pass emits        [P, PASS].
Tribute/return emit [P, TRIBUTE/RETURN, rank].
Sequence starts with [BOS, LEVEL].
"""

import numpy as np

from .cards import NUM_RANKS, is_wildcard, order_of, rank_of
from .combos import PASS, TYPE_NAMES

VOCAB = 48
PAD_TOK, BOS_TOK = 0, 1
LEVEL_BASE = 2
PLAYER_BASE = 15
TYPE_BASE = 19          # + move type (0..10)
TRIBUTE_TOK, RETURN_TOK = 30, 31
RANK_BASE = 32
MAX_SEQ = 512

N_TYPES = 11
# hand/action feature layout
FEAT_DIM = (
    15      # hand rank counts /4
    + 1     # wildcards in hand /2
    + 1     # hand size /27
    + 4     # cards left per relative player /27
    + 4     # done flags per relative player
    + 13    # level one-hot
    + N_TYPES  # action type one-hot
    + 15    # action claim rank counts /4
    + 1     # action size /27
    + 1     # action wildcards used /2
    + 1     # action comparison key /15
    + N_TYPES  # current lead type one-hot (all 0 if leading)
    + 1     # lead key /15
    + 1     # leading flag
)  # = 80


def tokenize(events, viewer, level):
    """events: engine event list; viewer: absolute player id. -> list[int]"""
    toks = [BOS_TOK, LEVEL_BASE + level]
    for ev in events:
        kind = ev[0]
        p = (ev[1] - viewer) % 4
        if kind == 'pass':
            toks.append(PLAYER_BASE + p)
            toks.append(TYPE_BASE + PASS)
        elif kind == 'play':
            mv = ev[2]
            toks.append(PLAYER_BASE + p)
            toks.append(TYPE_BASE + mv.type)
            for r in sorted(mv.claim_ranks):
                toks.append(RANK_BASE + r)
        elif kind == 'tribute':
            toks += [PLAYER_BASE + p, TRIBUTE_TOK, RANK_BASE + ev[2]]
        elif kind == 'return':
            toks += [PLAYER_BASE + p, RETURN_TOK, RANK_BASE + ev[2]]
    if len(toks) > MAX_SEQ:
        toks = toks[:2] + toks[-(MAX_SEQ - 2):]
    return toks


def tokenize_cached(events, viewer, level, cache):
    """Same output as tokenize(), but only tokenizes events appended since
    the previous call for this viewer.  `cache` is a per-game dict that must
    be reset whenever a new round starts (events only ever grow)."""
    ent = cache.get(viewer)
    if ent is None or ent[0] > len(events) or ent[2] != level:
        ent = [0, [BOS_TOK, LEVEL_BASE + level], level]
        cache[viewer] = ent
    toks = ent[1]
    for ev in events[ent[0]:]:
        kind = ev[0]
        p = (ev[1] - viewer) % 4
        if kind == 'pass':
            toks.append(PLAYER_BASE + p)
            toks.append(TYPE_BASE + PASS)
        elif kind == 'play':
            mv = ev[2]
            toks.append(PLAYER_BASE + p)
            toks.append(TYPE_BASE + mv.type)
            for r in sorted(mv.claim_ranks):
                toks.append(RANK_BASE + r)
        elif kind == 'tribute':
            toks += [PLAYER_BASE + p, TRIBUTE_TOK, RANK_BASE + ev[2]]
        elif kind == 'return':
            toks += [PLAYER_BASE + p, RETURN_TOK, RANK_BASE + ev[2]]
    ent[0] = len(events)
    if len(toks) > MAX_SEQ:
        return toks[:2] + toks[-(MAX_SEQ - 2):]
    return list(toks)


def hand_action_features(obs, move):
    """69-dim float32 features for (state-side hand info, candidate move)."""
    lv = obs["level"]
    me = obs["player"]
    f = np.zeros(FEAT_DIM, dtype=np.float32)
    i = 0
    for c in obs["hand"]:
        f[rank_of(c)] += 0.25
    i += 15
    f[i] = sum(1 for c in obs["hand"] if is_wildcard(c, lv)) / 2.0; i += 1
    f[i] = len(obs["hand"]) / 27.0; i += 1
    for rel in range(4):
        f[i + rel] = obs["left"][(me + rel) % 4] / 27.0
    i += 4
    for rel in range(4):
        f[i + rel] = 1.0 if obs["done"][(me + rel) % 4] else 0.0
    i += 4
    f[i + lv] = 1.0; i += 13
    f[i + move.type] = 1.0; i += N_TYPES
    for r in move.claim_ranks:
        f[i + r] += 0.25
    i += 15
    f[i] = move.size / 27.0; i += 1
    f[i] = sum(1 for c in move.cards if is_wildcard(c, lv)) / 2.0; i += 1
    f[i] = (move.key / 15.0) if move.type != PASS else 0.0; i += 1
    lead = obs["lead"]
    if lead is not None and lead.type != PASS:
        f[i + lead.type] = 1.0
        f[i + N_TYPES] = lead.key / 15.0
        f[i + N_TYPES + 1] = 0.0
    else:
        f[i + N_TYPES + 1] = 1.0  # leading
    i += N_TYPES + 2
    assert i == FEAT_DIM
    return f


# offsets of the per-move block inside the feature vector
_OFF_MTYPE = 15 + 1 + 1 + 4 + 4 + 13          # 38: action type one-hot
_OFF_MCLAIM = _OFF_MTYPE + N_TYPES             # 49: claim rank counts
_OFF_MSIZE = _OFF_MCLAIM + 15                  # 64
_OFF_MWILD = _OFF_MSIZE + 1                    # 65
_OFF_MKEY = _OFF_MWILD + 1                     # 66
_OFF_LEAD = _OFF_MKEY + 1                      # 67: lead block (state side)


def _state_features(obs):
    """The state-side part of hand_action_features (identical values)."""
    lv = obs["level"]
    me = obs["player"]
    f = np.zeros(FEAT_DIM, dtype=np.float32)
    hand = obs["hand"]
    for c in hand:
        f[rank_of(c)] += 0.25
    i = 15
    f[i] = sum(1 for c in hand if is_wildcard(c, lv)) / 2.0; i += 1
    f[i] = len(hand) / 27.0; i += 1
    left = obs["left"]
    for rel in range(4):
        f[i + rel] = left[(me + rel) % 4] / 27.0
    i += 4
    done = obs["done"]
    for rel in range(4):
        f[i + rel] = 1.0 if done[(me + rel) % 4] else 0.0
    i += 4
    f[i + lv] = 1.0
    lead = obs["lead"]
    if lead is not None and lead.type != PASS:
        f[_OFF_LEAD + lead.type] = 1.0
        f[_OFF_LEAD + N_TYPES] = lead.key / 15.0
        f[_OFF_LEAD + N_TYPES + 1] = 0.0
    else:
        f[_OFF_LEAD + N_TYPES + 1] = 1.0  # leading
    return f


def encode_decision(obs, tok_cache=None):
    """-> (tokens list[int], feats ndarray [n_legal, FEAT_DIM])

    Fast path: the state-side features are computed once per decision and
    only the per-move block is filled per legal move.  Output is bit-for-bit
    identical to stacking hand_action_features(obs, m) (see tests).
    `tok_cache` (optional, per game): incremental tokenization."""
    if tok_cache is None:
        toks = tokenize(obs["events"], obs["player"], obs["level"])
    else:
        toks = tokenize_cached(obs["events"], obs["player"], obs["level"],
                               tok_cache)
    legal = obs["legal"]
    lv = obs["level"]
    base = _state_features(obs)
    feats = np.empty((len(legal), FEAT_DIM), dtype=np.float32)
    feats[:] = base
    for j, move in enumerate(legal):
        row = feats[j]
        row[_OFF_MTYPE + move.type] = 1.0
        for r in move.claim_ranks:
            row[_OFF_MCLAIM + r] += 0.25
        row[_OFF_MSIZE] = move.size / 27.0
        nw = 0
        for c in move.cards:
            if c % 54 < 52 and (c % 54) // 4 == lv and (c % 54) % 4 == 0:
                nw += 1
        row[_OFF_MWILD] = nw / 2.0
        row[_OFF_MKEY] = (move.key / 15.0) if move.type != PASS else 0.0
    return toks, feats


def pad_tokens(toks, length=None):
    length = length or MAX_SEQ
    arr = np.zeros(length, dtype=np.int64)
    arr[:len(toks)] = toks[:length]
    return arr, min(len(toks), length)


# ---------------------------------------------------------------------------
# flat encoding (for the numpy MLP demo model / fallback)
# ---------------------------------------------------------------------------

FLAT_DIM = FEAT_DIM + 4 * 15 + N_TYPES + 15 + 1   # 80+60+11+15+1 = 167


def encode_flat(obs, move):
    """Flat features = hand/action features + aggregated history."""
    from .combos import PASS as _PASS
    me = obs["player"]
    base = hand_action_features(obs, move)
    f = np.zeros(FLAT_DIM, dtype=np.float32)
    f[:FEAT_DIM] = base
    i = FEAT_DIM
    # cards played per relative player per rank
    for ev in obs["events"]:
        if ev[0] == 'play':
            rel = (ev[1] - me) % 4
            for r in ev[2].claim_ranks:
                f[i + rel * 15 + r] += 0.125
    i += 60
    # last non-pass move in history
    last = None
    for ev in reversed(obs["events"]):
        if ev[0] == 'play':
            last = ev[2]
            break
        if ev[0] in ('tribute', 'return'):
            break
    if last is not None:
        f[i + last.type] = 1.0
        f[i + N_TYPES + (last.claim_ranks[0] if last.claim_ranks else 0)] = 1.0
        f[i + N_TYPES + 15] = last.size / 27.0
    return f
