# -*- coding: utf-8 -*-
"""RingRunner: run many rounds concurrently (generator engine) and batch
their decision points for external (GPU) Q evaluation.

The runner is transport-agnostic: collect_requests() returns encoded
decisions, step(results) advances games. train_fast wires it to the GPU
inference server; tests wire it to a stub.
"""

import random

import numpy as np

from .cards import rank_of
from .encode import encode_decision
from .engine import GuandanRound, random_tribute_mode


def _belief_label(rnd, me):
    """Oracle label: rank-count vectors (15 each) of the 3 other players'
    hidden hands, in relative seat order (next, partner, prev). /4 normed."""
    lab = np.zeros(45, dtype=np.float32)
    for rel in range(1, 4):
        p = (me + rel) % 4
        base = (rel - 1) * 15
        for c in rnd.hands[p]:
            lab[base + rank_of(c)] += 0.25
    return lab


class _Game:
    __slots__ = ("gen", "obs", "samples", "round", "enc", "tok_cache")

    def __init__(self, gen, rnd):
        self.gen = gen
        self.round = rnd
        self.obs = None
        self.enc = None
        self.samples = []   # (toks, feat_chosen, player)
        self.tok_cache = {}  # incremental tokenizer state (per round)


LADDER_LEVEL = 1        # rank index of '2' (Botzone default initdata)


def sample_setting(rng, ladder_frac=0.0):
    """(level, tribute_mode) for a new training round.

    With probability `ladder_frac` use the Botzone default setting (level 2,
    no tribute, seat 0 leads) -- what ladder games use when the platform
    passes empty initdata.  Otherwise random level and tribute mode."""
    if ladder_frac > 0 and rng.random() < ladder_frac:
        return LADDER_LEVEL, None
    level = rng.randrange(13)
    return level, random_tribute_mode(rng)


class RingRunner:
    def __init__(self, ring=16, seed=0, eps=0.02, top_k=10, ladder_frac=0.0):
        self.rng = random.Random(seed)
        self.eps = eps
        self.top_k = top_k
        self.ladder_frac = ladder_frac
        self.games = []
        self.finished = []  # list of episodes: (samples, rewards)
        self.episodes_done = 0
        for _ in range(ring):
            self.games.append(self._new_game())

    # ------------------------------------------------------------------
    def _new_game(self):
        rng = random.Random(self.rng.getrandbits(48))
        level, tmode = sample_setting(rng, self.ladder_frac)
        rnd = GuandanRound(level, rng, tmode)
        gen = rnd.play_steps()
        g = _Game(gen, rnd)
        self._advance(g, None, first=True)
        return g

    def _advance(self, g, idx, first=False):
        """Advance one game; on completion, bank episode and restart."""
        while True:
            try:
                g.obs = next(g.gen) if first else g.gen.send(idx)
                return
            except StopIteration as e:
                rewards, _ranking = e.value
                if g.samples:
                    self.finished.append((g.samples, rewards))
                self.episodes_done += 1
                # restart in place
                rng = random.Random(self.rng.getrandbits(48))
                level, tmode = sample_setting(rng, self.ladder_frac)
                g.round = GuandanRound(level, rng, tmode)
                g.gen = g.round.play_steps()
                g.samples = []
                g.tok_cache = {}
                first, idx = True, None
                continue

    # ------------------------------------------------------------------
    def collect_requests(self):
        """-> list of (slot, toks, feats[A,F]) for all pending decisions."""
        out = []
        for slot, g in enumerate(self.games):
            g.enc = encode_decision(g.obs, g.tok_cache)
            out.append((slot, g.enc[0], g.enc[1]))
        return out

    def step(self, results):
        """results: dict slot -> q ndarray [A]. Choose actions, advance."""
        for slot, q in results.items():
            g = self.games[slot]
            n = len(q)
            if self.eps > 0 and self.rng.random() < self.eps:
                if self.top_k > 1:
                    k = min(self.top_k, n)
                    idx = int(self.rng.choice(list(np.argsort(q)[-k:])))
                else:
                    idx = self.rng.randrange(n)
            else:
                idx = int(np.argmax(q))
            toks, feats = g.enc
            g.samples.append((np.asarray(toks, dtype=np.int16),
                              feats[idx], g.obs["player"],
                              _belief_label(g.round, g.obs["player"])))
            self._advance(g, idx)

    def pop_episodes(self):
        """-> list of (toks, feat, target, belief_label) flattened samples."""
        out = []
        for samples, rewards in self.finished:
            for toks, feat, player, belief in samples:
                out.append((toks, feat, np.float32(rewards[player] / 3.0),
                            belief))
        self.finished = []
        return out
