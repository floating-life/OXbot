# -*- coding: utf-8 -*-
"""The training engine's tribute / return / resist / lead-player logic must
match the official Botzone judge exactly (hands after the exchange and the
player who leads), including the clockwise rule for equal double tributes.

    python tests/test_judge_compat.py [path/to/judge.py]

Default judge: judge/judge_official.py if present, else judge/judge_fixed.py.
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
from fabledan.cards import level_rank           # noqa: E402
from fabledan.engine import GuandanRound        # noqa: E402


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


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        ROOT, "judge", "judge_official.py")
    if not os.path.exists(path):
        path = os.path.join(ROOT, "judge", "judge_fixed.py")
    judge = JR.JudgeHost(path)
    rng = random.Random(1)
    n = 0
    for scn in ["single", "double", "resist1", "resist2", "tie_cw", "tie_ccw"] * 25:
        initdata, _ = JR.make_initdata(rng, scn)
        lj, hj, lvs = judge_after_exchange(judge, initdata)
        le, he = engine_after_exchange(initdata, lvs)
        assert lj == le, "%s: leader judge=%d engine=%d" % (scn, lj, le)
        assert hj == he, "%s: hands differ after exchange" % scn
        n += 1
    print("engine tribute/return/resist/lead == judge (%s) on %d deals OK"
          % (os.path.basename(path), n))


if __name__ == "__main__":
    main()
