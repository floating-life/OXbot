# -*- coding: utf-8 -*-
"""GuanDan single-round engine (botzone-compatible rules incl. tribute/接风).

Players 0..3, teams (0,2) and (1,3).
Events recorded for tokenization:
  ('tribute', player, rank)   ('return', player, rank)
  ('play', player, Move)      ('pass', player)
Rewards: winning team gets +3 (双下) / +2 (1st+3rd) / +1 (1st+4th), losers
the negative. Per-player reward = team reward.
"""

import random

from .cards import BJ, NUM_CARDS, is_wildcard, order_of, rank_of
from .combos import PASS_MOVE, Move, beats, gen_moves


def partner(p):
    return (p + 2) % 4


def forced_tribute_card(cards, lv):
    """Largest card excluding the wildcard (heart level)."""
    cand = [c for c in cards if not is_wildcard(c, lv)]
    if not cand:
        cand = list(cards)
    return max(cand, key=lambda c: order_of(rank_of(c), lv))


def return_ok_judge(card, lv):
    """Official Botzone return rule, after set_level reorders pointorder.

    The official source uses a ceiling of natural 9, or natural 8 when
    playing 9.  Natural 10 is rejected; the level card is above the ceiling.
    This differs from the separately supplied corrected referee."""
    r = rank_of(card)
    ceiling = 7 if lv == 8 else 8       # rank indices for natural 8 / 9
    return r < 13 and order_of(r, lv) <= order_of(ceiling, lv)


def return_ok_strict(card, lv):
    """Backward-compatible name for the official return predicate."""
    return return_ok_judge(card, lv)


def default_return_card(cards, lv, exclude=()):
    """Heuristic return (还贡): smallest card the official judge accepts.

    `exclude`: cards that must not be returned (e.g. the tribute card just
    received -- the judge checks the return against the ORIGINAL deal)."""
    pool = [c for c in cards if c not in exclude]
    cand = [c for c in pool if return_ok_judge(c, lv)]
    if not cand:
        raise ValueError("no official-legal return card in the original deal")
    return min(cand, key=lambda c: (order_of(rank_of(c), lv), c))


class GuandanRound:
    """One round. `agents` is a list of 4 objects with .act(obs) -> move index.

    tribute_mode: None | ('single', last, first) | ('double', last, first)
    """

    def __init__(self, level, rng=None, tribute_mode=None, deal=None, feature_dim=80):
        self.lv = level
        self.feature_dims = ([feature_dim] * 4 if isinstance(feature_dim, int)
                             else list(feature_dim))
        if len(self.feature_dims) != 4 or any(v not in (80, 224) for v in self.feature_dims):
            raise ValueError("each player needs feature dimension 80 or 224")
        self.rng = rng or random.Random()
        if deal is None:
            deck = list(range(NUM_CARDS))
            self.rng.shuffle(deck)
            self.hands = [deck[i * 27:(i + 1) * 27] for i in range(4)]
        else:
            self.hands = [list(h) for h in deal]
        self.events = []
        self.tribute_mode = tribute_mode
        self.done_order = []          # players in finish order
        self.lead_player = 0
        self.resist = False

    # ------------------------------------------------------------------
    def _do_tribute(self):
        """Tribute / return / resist exactly as the official Botzone judge
        (verified against the judge source, see tests/test_judge_compat.py):

        - single: last pays first, first returns to last; (last+2)%4 leads.
        - double, different ranks: bigger card -> first, smaller -> first+2,
          each receiver returns to its own payer; the bigger payer leads.
        - double, SAME rank: clockwise -- first takes (first+1)'s card,
          first+2 takes (first+3)'s card; (first+1) leads.
        - resist (payers hold both big jokers): no exchange; first leads.
        """
        mode = self.tribute_mode
        if not mode:
            self.lead_player = 0
            return
        kind, last, first = mode
        lv = self.lv
        if kind == 'single':
            payers = [last]
        else:
            payers = [last, partner(last)]
        n_bj = sum(1 for p in payers for c in self.hands[p]
                   if rank_of(c) == BJ)
        if n_bj >= 2:
            self.resist = True
            self.lead_player = first
            return
        if kind == 'single':
            c = forced_tribute_card(self.hands[last], lv)
            self.hands[last].remove(c)
            self.hands[first].append(c)
            self.events.append(('tribute', last, rank_of(c)))
            r = default_return_card(self.hands[first], lv, exclude=(c,))
            self.hands[first].remove(r)
            self.hands[last].append(r)
            self.events.append(('return', first, rank_of(r)))
            self.lead_player = (last + 2) % 4
            return
        cards = {p: forced_tribute_card(self.hands[p], lv) for p in payers}
        p_a, p_b = payers
        if rank_of(cards[p_a]) == rank_of(cards[p_b]):
            recv_of = {(first + 1) % 4: first, (first + 3) % 4: partner(first)}
            self.lead_player = (first + 1) % 4
        else:
            big = max(payers, key=lambda p: order_of(rank_of(cards[p]), lv))
            small = p_b if big == p_a else p_a
            recv_of = {big: first, small: partner(first)}
            self.lead_player = big
        # the judge asks `last` first, then its partner
        for payer in (last, partner(last)):
            card = cards[payer]
            self.hands[payer].remove(card)
            self.hands[recv_of[payer]].append(card)
            self.events.append(('tribute', payer, rank_of(card)))
        payer_of = dict((v, k) for k, v in recv_of.items())
        # returns: first, then first+2 (each to the player who paid them)
        for recv in (first, partner(first)):
            payer = payer_of[recv]
            r = default_return_card(self.hands[recv], lv,
                                    exclude=(cards[payer],))
            self.hands[recv].remove(r)
            self.hands[payer].append(r)
            self.events.append(('return', recv, rank_of(r)))

    # ------------------------------------------------------------------
    def play(self, agents, sample_cb=None):
        """Run the round with callback-style agents. Returns rewards list[4].

        sample_cb(player, obs, legal, chosen_idx) is called at each decision
        point with >=2 legal moves (for training data collection).
        """
        gen = self.play_steps()
        try:
            obs = next(gen)
            while True:
                idx = agents[obs["player"]].act(obs)
                if sample_cb is not None:
                    sample_cb(obs["player"], obs, obs["legal"], idx)
                obs = gen.send(idx)
        except StopIteration as e:
            return e.value

    # ------------------------------------------------------------------
    def play_steps(self):
        """Generator interface: yields obs at each decision point (>=2 legal
        moves); caller .send(chosen_idx). Returns (rewards, ranking)."""
        self._do_tribute()
        return (yield from self._play_from(self.lead_player, None, None,
                                          [False] * 4))

    def fork_decision(self, obs):
        """Clone a live decision for offline counterfactual rollouts.

        Private allocations belong only to the simulator. Policies still
        receive _make_obs(), never this round object or the other hands.
        The prefix and tribute exchange are retained, not replayed twice.
        """
        if (obs["level"] != self.lv or obs["events"] != self.events
                or list(obs["hand"]) != self.hands[obs["player"]]
                or obs["left"] != list(map(len, self.hands))):
            raise ValueError("fork requires the round's current observation")
        clone = GuandanRound(self.lv, random.Random(0), deal=self.hands,
                             feature_dim=self.feature_dims)
        clone.events = list(self.events)
        clone.done_order = list(self.done_order)
        clone.lead_player = self.lead_player
        clone.resist = self.resist
        return clone, clone._play_from(obs["player"], obs["lead"],
                                       obs["lead_owner"], list(obs["done"]))

    def _play_from(self, cur, lead_move, lead_owner, done):
        """Shared rules for a new game and a fork of an existing decision."""
        lv = self.lv

        def next_active(p):
            q = (p + 1) % 4
            while done[q]:
                q = (q + 1) % 4
            return q

        while len(self.done_order) < 3:
            if lead_move is not None and cur == lead_owner:
                # trick won by lead_owner
                if done[cur]:
                    # 接风: partner leads (or next active if partner done)
                    nxt = partner(cur)
                    if done[nxt]:
                        nxt = next_active(cur)
                    cur = nxt
                lead_move = None
                lead_owner = None
                continue

            legal = gen_moves(self.hands[cur], lv, lead_move,
                              feature_dim=self.feature_dims[cur])
            if len(legal) == 1:
                idx = 0
            else:
                obs = self._make_obs(cur, legal, lead_move, lead_owner, done)
                idx = yield obs
            move = legal[idx]

            if move.type == PASS_MOVE.type:
                self.events.append(('pass', cur))
            else:
                for c in move.cards:
                    self.hands[cur].remove(c)
                self.events.append(('play', cur, move))
                lead_move = move
                lead_owner = cur
                if not self.hands[cur]:
                    done[cur] = True
                    self.done_order.append(cur)
                    if len(self.done_order) == 2:
                        a, b = self.done_order
                        if b == partner(a):     # 双下, round over
                            break
            # advance to next player still holding cards, but keep
            # lead_owner reachable so trick-completion check fires
            cur = (cur + 1) % 4
            while done[cur] and cur != lead_owner:
                cur = (cur + 1) % 4

        # final ranking
        rest = [p for p in range(4) if p not in self.done_order]
        # order remaining by nothing meaningful; double-down case rest=2
        ranking = self.done_order + rest
        return self._rewards(ranking), ranking

    # play_steps ends here; `return` inside a generator sets StopIteration.value

    # ------------------------------------------------------------------
    def _rewards(self, ranking):
        first = ranking[0]
        winners = (first, partner(first))
        pos_partner = ranking.index(partner(first))
        score = {1: 3, 2: 2, 3: 1}[pos_partner]
        return [score if p in winners else -score for p in range(4)]

    # ------------------------------------------------------------------
    def _make_obs(self, p, legal, lead_move, lead_owner, done):
        return {
            "player": p,
            "feature_dim": self.feature_dims[p],
            "level": self.lv,
            "hand": list(self.hands[p]),
            "legal": legal,
            "lead": lead_move,
            "lead_owner": lead_owner,
            "events": self.events,
            "done": list(done),
            "left": [len(self.hands[i]) for i in range(4)],
        }


def random_tribute_mode(rng):
    """Random previous-round outcome for training diversity."""
    x = rng.random()
    if x < 1 / 3:
        return None
    first = rng.randrange(4)
    last = rng.choice([p for p in range(4) if p != first and p != partner(first)])
    if x < 2 / 3:
        return ('single', last, first)
    return ('double', last, first)


def play_round(agents, rng=None, level=None, tribute_mode='random',
               sample_cb=None, feature_dim=None):
    rng = rng or random.Random()
    if level is None:
        level = rng.randrange(13)
    if tribute_mode == 'random':
        tribute_mode = random_tribute_mode(rng)
    if feature_dim is None:
        feature_dim = [getattr(agent, "feature_dim", 80) for agent in agents]
    rnd = GuandanRound(level, rng, tribute_mode, feature_dim=feature_dim)
    rewards, ranking = rnd.play(agents, sample_cb)
    return rewards, ranking, rnd
