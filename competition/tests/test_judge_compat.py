# -*- coding: utf-8 -*-
"""The training engine's tribute / return / resist / lead-player logic must
match the official Botzone judge exactly (hands after the exchange and the
player who leads), including the clockwise rule for equal double tributes.

    python tests/test_judge_compat.py [path/to/judge.py]

Default judge: judge/judge_official.py.  A missing official source is an error.
"""
import copy
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))
sys.path.insert(0, os.path.join(ROOT, "botzone"))

import judge_runner as JR                       # noqa: E402
from fabledan.cards import level_rank, rank_of  # noqa: E402
from fabledan.engine import (GuandanRound, default_return_card,
                            return_ok_judge, return_ok_strict)  # noqa: E402
from bot_fabledan import Mirror, exchange_response  # noqa: E402

OFFICIAL_JUDGE = os.path.join(ROOT, "judge", "judge_official.py")


class OfficialRejected(Exception):
    pass


def return_boundaries(path=OFFICIAL_JUDGE):
    """Golden cases query the actual referee, independent of our helpers.

    Every physical card is tested at every level, including both jokers,
    natural 9/10, level cards and the original-deal membership requirement.
    A manually specified natural-rank set also protects against accidentally
    switching these tests back to the corrected attachment."""
    judge = JR.JudgeHost(path)
    ns = {"__name__": "botzone_judge"}
    exec(judge.code, ns)

    def reject(player, reason):
        raise OfficialRejected(reason)

    ns["setError"] = reject
    hand = list(range(108))
    for level in JR.LEVELS:
        lv = level_rank(level)
        ns["pointorder"] = list("234567890JQKA")
        ns["set_level"](level)
        expected_ranks = set(range(1, 9)) - {lv}  # natural 2..9, no level
        for card in hand:
            try:
                accepted = ns["isValidReturn"](hand, card, level, 0)
            except OfficialRejected:
                accepted = False
            expected = rank_of(card) in expected_ranks
            assert accepted == expected, (level, card, accepted, expected)
            assert return_ok_judge(card, lv) == accepted, (level, card)
            assert return_ok_strict(card, lv) == accepted, (level, card)

        # A received tribute may be small enough but is absent from the
        # original deal.  Check the selector and Bot mirror both exclude it.
        legal = [card for card in hand if rank_of(card) in expected_ranks]
        received = min(legal)
        original = [card for card in legal if card != received]
        selected = default_return_card(original + [received], lv, (received,))
        assert selected in original and selected != received
        assert ns["isValidReturn"](original, selected, level, 0)
        try:
            ns["isValidReturn"](original, received, level, 0)
        except OfficialRejected as error:
            assert str(error) == "NOT_YOUR_POKER"
        else:
            raise AssertionError("official judge accepted a received tribute")
        mirror = Mirror()
        mirror.lv, mirror.hand = lv, original + [received]
        mirror.received_tribute = {received}
        assert exchange_response(mirror, "return") == [selected]
        mirror.resist = True
        assert exchange_response(mirror, "return") == []
        assert exchange_response(mirror, "tribute") == []

    # Old fallbacks returned 10 or an excluded tribute when no legal card
    # remained.  Surface that state rather than knowingly emitting it.
    for cards, lv, excluded in [([36, 0], 0, ()), ([4], 0, (4,))]:
        try:
            default_return_card(cards, lv, excluded)
        except ValueError:
            pass
        else:
            raise AssertionError("selector returned an official-illegal card")
    return len(JR.LEVELS) * len(hand)


def judge_after_exchange(judge, initdata):
    """Drive the judge until the first play request.
    -> (leader, hands after exchange, effective level string)"""
    bots = [JR.InProcBot(("rule", None)) for _ in range(4)]
    log = []
    init_obj = copy.deepcopy(initdata)
    out = judge.call({"log": log, "initdata": init_obj})
    init_obj = out.get("initdata", init_obj)
    while True:
        assert out["command"] == "request", out
        (pid_s, req), = out["content"].items()
        if req["stage"] == "play":
            break
        resp = bots[int(pid_s)].turn(req)
        log.append({"output": out})
        log.append({pid_s: {"response": resp, "verdict": "OK"}})
        out = judge.call({"log": log, "initdata": init_obj})
    leader = int(pid_s)
    ns = {"__name__": "botzone_judge"}
    exec(judge.code, ns)
    full = {"log": log + [{"output": out}, {pid_s: {"response": []}}],
            "initdata": copy.deepcopy(init_obj)}
    hands = ns["recover_state"](full)
    return leader, [sorted(h) for h in hands], req["global"]["level"]


def engine_after_exchange(initdata, level_str):
    kind = "single" if initdata["tribute"] == 1 else "double"
    rnd = GuandanRound(level_rank(level_str), random.Random(0),
                       (kind, initdata["last"], initdata["first"]),
                       deal=initdata["allocation"])
    rnd._do_tribute()
    return rnd.lead_player, [sorted(h) for h in rnd.hands]


def exchanges(path=OFFICIAL_JUDGE):
    judge = JR.JudgeHost(path)
    rng = random.Random(1)
    n = 0
    for level in JR.LEVELS:
        for scn in ["single", "double", "resist1", "resist2", "tie_cw", "tie_ccw"] * 2:
            initdata, _ = JR.make_initdata(rng, scn)
            initdata["level"] = [level, level]
            lj, hj, lvs = judge_after_exchange(judge, initdata)
            le, he = engine_after_exchange(initdata, lvs)
            assert lj == le, "%s level %s: leader judge=%d engine=%d" % (scn, level, lj, le)
            assert hj == he, "%s level %s: hands differ after exchange" % (scn, level)
            n += 1
    return n


def test_official_return_boundaries():
    return_boundaries()


def test_official_exchange_compatibility():
    exchanges()


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else OFFICIAL_JUDGE
    boundaries = return_boundaries(path)
    n = exchanges(path)
    print("official return boundary golden: %d card/level cases OK" % boundaries)
    print("engine tribute/return/resist/lead == judge (%s) on %d deals OK"
          % (os.path.basename(path), n))


if __name__ == "__main__":
    main()
