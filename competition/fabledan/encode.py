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

from .cards import NUM_RANKS, SEQV_TO_RANK, is_wildcard, order_of, rank_of
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

# v2 preserves the complete legacy prefix and appends physical-card structure.
STRUCTURE_FEAT_DIM = 224
_STRUCTURE_WINDOWS = {
    length: np.asarray([SEQV_TO_RANK[low:low + length]
                        for low in range(1, 16 - length)], dtype=np.int64)
    for length in (2, 3, 5)
}
_FULL_DISTINCT_RANKS = ~np.eye(13, 15, dtype=bool)


def _check_feat_dim(feat_dim):
    if feat_dim not in (FEAT_DIM, STRUCTURE_FEAT_DIM):
        raise ValueError("feature dimension must be 80 (legacy) or 224 (structure v2)")


def _structure_hand(hand, level):
    """One validated physical-card inventory shared by all candidate actions."""
    if not isinstance(level, (int, np.integer)) or not 0 <= level < 13:
        raise ValueError("structure features require a level rank in 0..12")
    faces = np.zeros(54, dtype=np.int16)
    physical = set()
    for card in hand:
        if isinstance(card, (bool, np.bool_)) or not isinstance(card, (int, np.integer)) or not 0 <= card < 108:
            raise ValueError("hand contains an invalid physical card id")
        if card in physical:
            raise ValueError("hand contains a duplicate physical card id")
        physical.add(card)
        faces[card % 54] += 1
    return faces, physical, int(level) * 4


def _remaining_structure(remaining, wildcard_face):
    """90 residual-hand features, vectorized over distinct face inventories.

    Each window is an independent potential combination, not a partition or
    optimal decomposition of the hand. Wildcards can support several potential
    windows, but the deficit sum within any one combination must fit the pool.
    """
    count = len(remaining)
    out = np.zeros((count, 90), dtype=np.float32)
    out[:, :54] = remaining / 2.0
    natural_faces = remaining.copy()
    wild = natural_faces[:, wildcard_face].copy()
    natural_faces[:, wildcard_face] = 0
    natural = np.empty((count, NUM_RANKS), dtype=np.int16)
    natural[:, :13] = natural_faces[:, :52].reshape(count, 13, 4).sum(axis=2)
    natural[:, 13:] = natural_faces[:, 52:]
    out[:, 54:69] = natural / 4.0
    out[:, 69] = wild / 2.0
    out[:, 70] = remaining.sum(axis=1) / 27.0
    out[:, 71:79] = (natural[:, :, None] >= np.arange(1, 9)[None, None, :]).sum(axis=1) / 15.0

    # A-low and A-high sequence windows exactly match combos.gen_moves.
    five = natural[:, _STRUCTURE_WINDOWS[5]]
    straight_cost = (five == 0).sum(axis=2)
    out[:, 79] = (straight_cost == 0).sum(axis=1) / 10.0
    out[:, 80] = (straight_cost <= wild[:, None]).sum(axis=1) / 10.0
    suits = natural_faces[:, :52].reshape(count, 13, 4).transpose(0, 2, 1)
    flush_cost = (suits[:, :, _STRUCTURE_WINDOWS[5]] == 0).sum(axis=3)
    out[:, 81] = (flush_cost == 0).sum(axis=(1, 2)) / 40.0
    out[:, 82] = (flush_cost <= wild[:, None, None]).sum(axis=(1, 2)) / 40.0
    plate_cost = np.maximum(0, 2 - natural[:, _STRUCTURE_WINDOWS[3]]).sum(axis=2)
    out[:, 83] = (plate_cost == 0).sum(axis=1) / 12.0
    out[:, 84] = (plate_cost <= wild[:, None]).sum(axis=1) / 12.0
    tube_cost = np.maximum(0, 3 - natural[:, _STRUCTURE_WINDOWS[2]]).sum(axis=2)
    out[:, 85] = (tube_cost == 0).sum(axis=1) / 13.0
    out[:, 86] = (tube_cost <= wild[:, None]).sum(axis=1) / 13.0

    triple_cost = np.maximum(0, 3 - natural[:, :13])
    pair_cost = np.maximum(0, 2 - natural)
    possible = ((natural[:, :13, None] > 0) & (natural[:, None, :] > 0)
                & _FULL_DISTINCT_RANKS[None]
                & (triple_cost[:, :, None] + pair_cost[:, None, :] <= wild[:, None, None]))
    # Jokers can supply only natural pair attachments; never substitute them.
    possible[:, :, 13:] &= natural[:, None, 13:] >= 2
    out[:, 87] = possible.sum(axis=(1, 2)) / 182.0
    bomb_sizes = np.minimum(natural[:, :13] + wild[:, None], 10)
    bomb_sizes = np.where((natural[:, :13] > 0) & (bomb_sizes >= 4), bomb_sizes, 0)
    out[:, 88] = bomb_sizes.max(axis=1) / 10.0
    out[:, 89] = (natural[:, 13] >= 2) & (natural[:, 14] >= 2)
    return out


def _structure_features(hand, level, moves):
    """Build 144 appended features without enumerating residual-hand moves."""
    faces, physical, wildcard_face = _structure_hand(hand, level)
    rows = np.empty((len(moves), STRUCTURE_FEAT_DIM - FEAT_DIM), dtype=np.float32)
    rows[:, :54] = faces / 2.0
    if not moves:
        return rows
    # Claims may give the same physical action several meanings. The residual
    # face inventory needs computing only once for each distinct inventory.
    inventory_index = {}
    inventories, row_indices = [], []
    for move in moves:
        cards = move.cards
        if len(cards) != move.size or (move.type == PASS and cards):
            raise ValueError("candidate physical action has an invalid size")
        removed = set()
        remaining = faces.copy()
        for card in cards:
            if isinstance(card, (bool, np.bool_)) or not isinstance(card, (int, np.integer)) or card not in physical or card in removed:
                raise ValueError("candidate must remove distinct physical cards actually held")
            removed.add(card)
            remaining[card % 54] -= 1
        key = remaining.tobytes()
        if key not in inventory_index:
            inventory_index[key] = len(inventories)
            inventories.append(remaining)
        row_indices.append(inventory_index[key])
    residual = _remaining_structure(np.stack(inventories), wildcard_face)
    rows[:, 54:] = residual[row_indices]
    return rows


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


def hand_action_features(obs, move, feat_dim=FEAT_DIM):
    """Legacy 80 or structure-v2 224 float32 state/action features."""
    _check_feat_dim(feat_dim)
    lv = obs["level"]
    me = obs["player"]
    f = np.zeros(feat_dim, dtype=np.float32)
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
    if feat_dim == STRUCTURE_FEAT_DIM:
        f[FEAT_DIM:] = _structure_features(obs["hand"], lv, [move])[0]
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


def encode_decision(obs, tok_cache=None, feat_dim=FEAT_DIM):
    """-> (tokens list[int], feats ndarray [n_legal, feat_dim])

    Fast path: the state-side features are computed once per decision and
    only the per-move block is filled per legal move.  Output is bit-for-bit
    identical to stacking hand_action_features(obs, m) (see tests).
    `tok_cache` (optional, per game): incremental tokenization."""
    _check_feat_dim(feat_dim)
    if tok_cache is None:
        toks = tokenize(obs["events"], obs["player"], obs["level"])
    else:
        toks = tokenize_cached(obs["events"], obs["player"], obs["level"],
                               tok_cache)
    legal = obs["legal"]
    lv = obs["level"]
    base = _state_features(obs)
    feats = np.empty((len(legal), feat_dim), dtype=np.float32)
    feats[:, :FEAT_DIM] = base
    if feat_dim == STRUCTURE_FEAT_DIM:
        feats[:, FEAT_DIM:] = _structure_features(obs["hand"], lv, legal)
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
