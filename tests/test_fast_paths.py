# -*- coding: utf-8 -*-
"""The actor fast paths must be bit-for-bit identical to upstream:
  - gen_moves() with type pruning when following
  - encode_decision() with state features computed once per decision
Run: python tests/test_fast_paths.py
"""
import os
import random
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

import _combos_ref as ref                      # noqa: E402
from fabledan import combos                    # noqa: E402
from fabledan.encode import (encode_decision, hand_action_features,  # noqa: E402
                             tokenize)
from fabledan.engine import GuandanRound, random_tribute_mode       # noqa: E402


def sig(moves):
    return [(m.type, m.key, tuple(m.cards), tuple(m.claim_ranks)) for m in moves]


def to_ref_move(m):
    return ref.Move(m.type, m.key, list(m.cards), list(m.claim_ranks))


def check_state(hand, lv, lead):
    new = combos.gen_moves(hand, lv, lead)
    old = ref.gen_moves(hand, lv, None if lead is None else to_ref_move(lead))
    assert sig(new) == sig(old), "gen_moves mismatch"


class _Spy:
    """Random agent that checks both fast paths at every decision."""

    def __init__(self, rng):
        self.rng = rng
        self.n = 0
        self.caches = {}      # id(events) -> (events ref, per-round cache)

    def act(self, obs):
        hand, lv, lead = obs["hand"], obs["level"], obs["lead"]
        check_state(hand, lv, lead)
        ev = obs["events"]
        ent = self.caches.get(id(ev))
        if ent is None or ent[0] is not ev:
            ent = (ev, {})
            self.caches = {id(ev): ent}
        toks, feats = encode_decision(obs, ent[1])
        assert toks == tokenize(ev, obs["player"], obs["level"]), \
            "tokenize_cached mismatch"
        slow = np.stack([hand_action_features(obs, m) for m in obs["legal"]])
        assert feats.dtype == slow.dtype and feats.shape == slow.shape
        assert np.array_equal(feats, slow), "encode_decision mismatch"
        self.n += 1
        return self.rng.randrange(len(obs["legal"]))


def run(rounds=400, seed=0):
    rng = random.Random(seed)
    spies = [_Spy(random.Random(seed + i)) for i in range(4)]
    for _ in range(rounds):
        r = random.Random(rng.getrandbits(32))
        rnd = GuandanRound(r.randrange(13), r, random_tribute_mode(r))
        rnd.play(spies)
    n = sum(s.n for s in spies)
    # extra: random hands vs random leads (incl. bomb-class leads)
    for _ in range(3000):
        lv = rng.randrange(13)
        hand = rng.sample(range(108), rng.randint(1, 27))
        other = rng.sample([c for c in range(108) if c not in hand], 27)
        leads = combos.gen_moves(other, lv, None)
        lead = rng.choice(leads) if leads and rng.random() < 0.85 else None
        check_state(hand, lv, lead)
        n += 1
    return n


if __name__ == "__main__":
    n = run()
    print("fast paths identical on %d states OK" % n)
