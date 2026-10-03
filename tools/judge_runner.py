# -*- coding: utf-8 -*-
"""Run GuanDan games through the OFFICIAL Botzone judge program.

The judge source (a .py that reads {"log": [...], "initdata": ...} and prints
{"command": "request"|"finish", ...}) is executed in-process exactly the way
Botzone drives it: one fresh judge run per turn, full log each time.  Bots
are driven with the Botzone bot protocol (requests/responses).

Any illegal response is detected by the judge itself (INVALID_*,
ILLEGAL_CLAIM, NOT_YOUR_POKER, UNKNOWN_ERROR ... -> score -2).

Bot drivers
  inproc     botzone/bot_fabledan.py logic in-process (fast; thousands of games)
  trad:CMD   fresh process per turn, full JSON history (= Botzone default)
  keep:CMD   long-running process (first turn full JSON, then raw requests)

Examples
  # legality sweep, every scenario, rule fallback policy
  python tools/judge_runner.py --judge judge/judge_official.py --games 2000
  # same with a (random or trained) transformer through the real numpy path
  python tools/judge_runner.py --judge judge/judge_official.py --games 300 \
      --weights ckpts/run1/latest.npz
  # protocol check of the real bot process (slow, a few games)
  python tools/judge_runner.py --judge judge/judge_official.py --games 3 \
      --driver "trad:python botzone/bot_fabledan.py"
  # duplicate head-to-head through the judge: A (seats 0,2 then 1,3) vs B
  python tools/judge_runner.py --judge judge/judge_official.py --games 400 \
      --weights a.npz --weights-b b.npz --scenario ladder
"""

import argparse
import contextlib
import copy
import io
import json
import os
import random
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "botzone"))

LEVELS = ['A', '2', '3', '4', '5', '6', '7', '8', '9', '0', 'J', 'Q', 'K']
SMALL_JOKERS, BIG_JOKERS = (52, 106), (53, 107)


# ---------------------------------------------------------------------------
# judge host
# ---------------------------------------------------------------------------

class JudgeHost:
    """Executes the judge source once per turn in a fresh namespace."""

    def __init__(self, path):
        with open(path, encoding="utf-8") as f:
            src = f.read()
        self.code = compile(src, path, "exec")

    def call(self, full_input):
        ns = {"__name__": "botzone_judge", "__file__": "judge.py"}
        exec(self.code, ns)
        buf = io.StringIO()
        try:
            with contextlib.redirect_stdout(buf):
                ns["main"](copy.deepcopy(full_input))
        except SystemExit:
            pass
        lines = [ln for ln in buf.getvalue().splitlines() if ln.strip()]
        if not lines:
            raise RuntimeError("judge printed nothing")
        return json.loads(lines[-1])


# ---------------------------------------------------------------------------
# bots
# ---------------------------------------------------------------------------

def load_model(weights):
    """None -> rule; 'random' -> uniform random; path -> transformer/mlp."""
    import numpy as np
    import bot_fabledan as B
    if weights in (None, "", "rule"):
        return ("rule", None)
    if weights == "random":
        return ("random", random.Random(12345))
    return B._classify_weights(np.load(weights, allow_pickle=False))


class InProcBot:
    """bot_fabledan's Mirror/respond in-process (= long-running behaviour)."""

    def __init__(self, model):
        import bot_fabledan as B
        self.B = B
        self.mirror = B.Mirror()
        self.kind, self.model = model
        self.times = []

    def turn(self, request):
        B = self.B
        t0 = time.time()
        req = json.loads(json.dumps(request))
        self.mirror.feed_request(req)
        action = B.respond(self.mirror, req, self.kind, self.model)
        stage = req.get("stage")
        self.mirror._apply_my_response(stage, action if stage != "deal" else None)
        self.times.append(time.time() - t0)
        return json.loads(json.dumps(action))     # must be JSON-serialisable

    def close(self):
        pass


class TradProcBot:
    """Fresh process per turn with the full requests/responses history."""

    def __init__(self, cmd):
        self.cmd = cmd
        self.requests, self.responses, self.times = [], [], []

    def turn(self, request):
        self.requests.append(request)
        payload = json.dumps({"requests": self.requests, "responses": self.responses})
        t0 = time.time()
        p = subprocess.run(self.cmd, shell=True, input=payload + "\n",
                           stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                           universal_newlines=True, timeout=60)
        self.times.append(time.time() - t0)
        lines = [ln for ln in p.stdout.splitlines() if ln.strip()]
        if len(lines) != 1:
            raise RuntimeError("traditional mode must print exactly one JSON "
                               "line, got %r (stderr %r)" % (p.stdout, p.stderr[-500:]))
        resp = json.loads(lines[0])["response"]
        self.responses.append(resp)
        return resp

    def close(self):
        pass


class KeepProcBot:
    """Long-running process: first turn full JSON, then one raw request/line."""

    MARK = ">>>BOTZONE_REQUEST_KEEP_RUNNING<<<"

    def __init__(self, cmd):
        self.p = subprocess.Popen(cmd, shell=True, stdin=subprocess.PIPE,
                                  stdout=subprocess.PIPE, universal_newlines=True,
                                  bufsize=1)
        self.started = False
        self.times = []

    def turn(self, request):
        if not self.started:
            payload = json.dumps({"requests": [request], "responses": []})
            self.started = True
        else:
            payload = json.dumps(request)
        t0 = time.time()
        self.p.stdin.write(payload + "\n")
        self.p.stdin.flush()
        line = self.p.stdout.readline()
        if not line:
            raise RuntimeError("bot process died")
        resp = json.loads(line)["response"]
        mark = self.p.stdout.readline()
        self.times.append(time.time() - t0)
        if self.MARK not in mark:
            raise RuntimeError("missing keep-running marker: %r" % mark)
        return resp

    def close(self):
        try:
            self.p.kill()
        except Exception:
            pass


def make_bot(driver, model):
    if driver == "inproc":
        return InProcBot(model)
    kind, _, cmd = driver.partition(":")
    if kind == "trad":
        return TradProcBot(cmd)
    if kind == "keep":
        return KeepProcBot(cmd + " --keep-running")
    raise ValueError("unknown driver %r" % driver)


# ---------------------------------------------------------------------------
# history format variants
# ---------------------------------------------------------------------------

def to_positional(req, pid):
    """FableDan notes that live Botzone logs carry a positional 4-slot history
    (slot i = latest move of seat (pid+i)%4 in the window); convert the judge's
    dict-list history to that form to exercise both parser paths."""
    hist = req.get("history")
    if req.get("stage") != "play" or not hist or \
            not any(isinstance(h, dict) and "player" in h for h in hist):
        return req
    moves = [h for h in hist if isinstance(h, dict) and "player" in h]
    slots = [[], [], [], []]
    last_self = -1
    for i in range(len(moves) - 1, -1, -1):
        if moves[i]["player"] == pid:
            last_self = i
            break
    if last_self >= 0:
        slots[0] = moves[last_self]["response"]
    for h in (moves[last_self + 1:] if last_self >= 0 else moves):
        slots[(h["player"] - pid) % 4] = h["response"]
    out = dict(req)
    out["history"] = slots
    return out


# ---------------------------------------------------------------------------
# scenarios
# ---------------------------------------------------------------------------

def _deal(rng):
    deck = list(range(108))
    rng.shuffle(deck)
    return [deck[i * 27:(i + 1) * 27] for i in range(4)]


def _move_cards_to(alloc, cards, target, rng, forbid=()):
    """Put `cards` into hand `target`, swapping out random cards not in forbid."""
    for c in cards:
        owner = [p for p in range(4) if c in alloc[p]][0]
        if owner == target:
            continue
        cand = [x for x in alloc[target] if x not in cards and x not in forbid]
        x = rng.choice(cand)
        alloc[target].remove(x)
        alloc[owner].remove(c)
        alloc[target].append(c)
        alloc[owner].append(x)


def _keep_out(alloc, cards, hands, rng):
    """Make sure none of `cards` is in any of `hands`."""
    others = [p for p in range(4) if p not in hands]
    for c in cards:
        for h in hands:
            if c in alloc[h]:
                o = rng.choice(others)
                x = rng.choice([y for y in alloc[o] if y not in cards])
                alloc[h].remove(c)
                alloc[o].remove(x)
                alloc[h].append(x)
                alloc[o].append(c)


def make_initdata(rng, scenario):
    """initdata dict for one game.  scenario in
    ladder | random | single | double | resist1 | resist2 | tie_cw | tie_ccw | mix"""
    if scenario == "mix":
        scenario = rng.choice(["ladder", "random", "random", "single", "double",
                               "resist1", "resist2", "tie_cw", "tie_ccw"])
    if scenario == "ladder":
        return {"seed": str(rng.getrandbits(31))}, scenario   # judge defaults
    alloc = _deal(rng)
    lvA, lvB = rng.choice(LEVELS), rng.choice(LEVELS)
    if scenario == "random":
        t = rng.choice([0, 1, 2])
        if t == 0:
            return {"allocation": alloc, "level": [lvA, lvB]}, scenario
        scenario = "single" if t == 1 else "double"
    first = rng.randrange(4)
    if scenario in ("single", "resist1"):
        last = rng.choice([(first + 1) % 4, (first + 3) % 4])
        if scenario == "resist1":
            _move_cards_to(alloc, BIG_JOKERS, last, rng)
        return {"allocation": alloc, "level": [lvA, lvB], "tribute": 1,
                "first": first, "last": last}, scenario
    if scenario in ("double", "resist2"):
        last = rng.choice([(first + 1) % 4, (first + 3) % 4])
        if scenario == "resist2":
            payers = [last, (last + 2) % 4]
            if rng.random() < 0.5:
                _move_cards_to(alloc, BIG_JOKERS, rng.choice(payers), rng)
            else:
                _move_cards_to(alloc, BIG_JOKERS[:1], payers[0], rng)
                _move_cards_to(alloc, BIG_JOKERS[1:], payers[1], rng)
        return {"allocation": alloc, "level": [lvA, lvB], "tribute": 2,
                "first": first, "last": last}, scenario
    if scenario in ("tie_cw", "tie_ccw"):
        # double tribute where both payers' biggest card is a small joker:
        # equal ranks -> the judge assigns clockwise.  tie_cw: last=first+3
        # (wiki rule and judge DISAGREE), tie_ccw: last=first+1 (agree).
        last = (first + 3) % 4 if scenario == "tie_cw" else (first + 1) % 4
        payers = [last, (last + 2) % 4]
        _keep_out(alloc, BIG_JOKERS, payers, rng)
        _move_cards_to(alloc, SMALL_JOKERS[:1], payers[0], rng, forbid=SMALL_JOKERS)
        _move_cards_to(alloc, SMALL_JOKERS[1:], payers[1], rng, forbid=SMALL_JOKERS)
        return {"allocation": alloc, "level": [lvA, lvB], "tribute": 2,
                "first": first, "last": last}, scenario
    raise ValueError(scenario)


# ---------------------------------------------------------------------------
# game loop
# ---------------------------------------------------------------------------

def run_game(judge, bots, initdata, positional=False):
    log = []
    init_obj = copy.deepcopy(initdata)
    out = judge.call({"log": log, "initdata": init_obj})
    if "initdata" in out:
        init_obj = out["initdata"]
    turns = 0
    while out.get("command") == "request":
        (pid_s, req), = out["content"].items()
        pid = int(pid_s)
        send = to_positional(req, pid) if positional else req
        try:
            resp = bots[pid].turn(send)
        except Exception as e:                # bot crashed -> record as error
            return {"error": {pid: "BOT_CRASH: %r" % e}, "turns": turns}
        log.append({"output": out})
        log.append({pid_s: {"response": resp, "verdict": "OK"}})
        out = judge.call({"log": log, "initdata": init_obj})
        turns += 1
        if turns > 2000:
            return {"error": {-1: "TOO_MANY_TURNS"}, "turns": turns}
    disp = out.get("display") or {}
    errs = disp.get("error")
    res = {"scores": {int(k): v for k, v in out["content"].items()},
           "turns": turns}
    if errs and any(e for e in errs):
        res["error"] = {i: e for i, e in enumerate(errs) if e}
    return res


def percentile(xs, q):
    if not xs:
        return 0.0
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(q / 100.0 * len(xs)))]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--judge", default=os.path.join(ROOT, "judge", "judge_official.py"))
    ap.add_argument("--games", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--scenario", default="mix")
    ap.add_argument("--weights", default=None,
                    help="team A model: .npz path | rule | random (default rule)")
    ap.add_argument("--weights-b", default=None,
                    help="team B model (default = same as A); enables duplicate "
                         "head-to-head scoring")
    ap.add_argument("--driver", default="inproc")
    ap.add_argument("--positional", action="store_true",
                    help="send FableDan's positional history format")
    ap.add_argument("--fail-fast", action="store_true")
    args = ap.parse_args()

    if not os.path.exists(args.judge):
        alt = os.path.join(ROOT, "judge", "judge_fixed.py")
        print("[warn] %s not found -- using %s. Put the official judge source "
              "there for real verification." % (args.judge, alt))
        args.judge = alt
    judge = JudgeHost(args.judge)
    rng = random.Random(args.seed)
    model_a = load_model(args.weights)
    model_b = load_model(args.weights_b) if args.weights_b else model_a
    h2h = args.weights_b is not None
    n_err, by_scn, times = 0, {}, []
    a_score = 0.0
    t_start = time.time()
    pair = None
    for g in range(args.games):
        if not h2h or g % 2 == 0:
            pair = make_initdata(rng, args.scenario)
        initdata, scn = pair
        a_seats = (0, 2) if g % 2 == 0 else (1, 3)
        bots = [make_bot(args.driver, model_a if p in a_seats else model_b)
                for p in range(4)]
        res = run_game(judge, bots, initdata, positional=args.positional)
        for b in bots:
            times.extend(b.times)
            b.close()
        st = by_scn.setdefault(scn, [0, 0])
        st[0] += 1
        if "error" in res:
            n_err += 1
            st[1] += 1
            print("game %d [%s] ERROR %s  initdata=%s" % (
                g, scn, res["error"], json.dumps(initdata)[:300]), flush=True)
            if args.fail_fast:
                break
        elif h2h:
            sc = res["scores"]
            a_score += sc[a_seats[0]] - sc[(a_seats[0] + 1) % 4]
    el = time.time() - t_start
    print("\n%d games through %s in %.1fs, errors: %d"
          % (g + 1, os.path.basename(args.judge), el, n_err))
    for scn, (n, e) in sorted(by_scn.items()):
        print("  %-8s games %5d  errors %d" % (scn, n, e))
    if times:
        print("bot time per turn: p50 %.1f ms  p99 %.1f ms  max %.1f ms" % (
            1000 * percentile(times, 50), 1000 * percentile(times, 99),
            1000 * max(times)))
    if h2h:
        print("A vs B (duplicate, through the judge): avg score diff per game "
              "%+.3f" % (a_score / max(1, g + 1)))
    sys.exit(1 if n_err else 0)


if __name__ == "__main__":
    main()
