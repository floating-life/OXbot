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
  cpp:CMD    C++ long-running process; no Python-only --keep-running argument

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
  # two different C++ bots (e.g. new FableDan vs online cf8), one shard of 4
  python tools/judge_runner.py --games 1000 --require-model --shard 0/4 \
      --driver "cpp:../bin/oxbot --model new.fbd" \
      --driver-b "cpp:../bin/oxbot --model cf8.bin" --report shard0.json
"""

import argparse
import contextlib
import copy
import io
import json
import hashlib
import os
import queue
import random
import re
import subprocess
import sys
import threading
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
    with np.load(weights, allow_pickle=False) as arrays:
        kind, model = B._classify_weights(arrays)
    if kind == "rule":
        raise ValueError("weights are not a recognized model: %s" % weights)
    return kind, model


class InProcBot:
    """bot_fabledan's Mirror/respond in-process (= long-running behaviour)."""

    def __init__(self, model, require_model=False):
        import bot_fabledan as B
        self.B = B
        self.mirror = B.Mirror()
        self.kind, self.model = model
        self.times = []
        self.require_model = require_model
        self.play_turns = 0

    def turn(self, request):
        B = self.B
        B.DIAG[:] = []
        t0 = time.time()
        req = json.loads(json.dumps(request))
        self.mirror.feed_request(req)
        action = B.respond(self.mirror, req, self.kind, self.model)
        if req.get("stage") == "play":
            self.play_turns += 1
        if self.require_model and (self.kind not in ("transformer", "mlp") or
                any("inference failed" in s or "unhandled:" in s for s in B.DIAG)):
            raise RuntimeError("model validation failed: %s" % B.DIAG)
        stage = req.get("stage")
        self.mirror._apply_my_response(stage, action if stage != "deal" else None)
        self.times.append(time.time() - t0)
        return json.loads(json.dumps(action))     # must be JSON-serialisable

    def close(self):
        pass


def check_response_model(out, require_model, stage=None, model_protocol="python"):
    if not require_model:
        return
    debug = out.get("debug", "")
    if model_protocol == "cpp":
        fields = dict(item.split("=", 1) for item in debug.split(";") if "=" in item)
        if out.get("error") or fields.get("stage") != stage or \
                fields.get("legality_fallback") != "0":
            raise RuntimeError("C++ bot failed its stage/legality contract: %s" % debug)
        if stage == "play":
            # BotAdapter currently emits a 12-hex payload SHA prefix.  Accept
            # the full 64-hex digest as well if diagnostics become richer.
            sha = fields.get("model_sha", "")
            if fields.get("policy") != "model" or \
                    fields.get("model_status") != "model_selected" or \
                    not re.fullmatch(r"(?:[0-9a-f]{12}|[0-9a-f]{64})", sha):
                raise RuntimeError("C++ bot did not use the requested model: %s" % debug)
        elif fields.get("policy") != "stage_rules":
            raise RuntimeError("C++ non-play stage did not use stage rules: %s" % debug)
        return fields
    if model_protocol != "python":
        raise ValueError("unknown model protocol: %s" % model_protocol)
    if not any("model=" + kind in debug for kind in ("transformer", "mlp")) or \
            "inference failed" in debug or "unhandled:" in debug:
        raise RuntimeError("bot did not use the requested model: %s" % debug)


def bot_environment():
    env = os.environ.copy()
    for name in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
        env[name] = "1"
    return env


class TradProcBot:
    """Fresh process per turn with the full requests/responses history."""

    def __init__(self, cmd, cwd=None, timeout=60, require_model=False):
        self.cmd = cmd
        self.cwd, self.timeout, self.require_model = cwd, timeout, require_model
        self.requests, self.responses, self.times = [], [], []

    def turn(self, request):
        self.requests.append(request)
        payload = json.dumps({"requests": self.requests, "responses": self.responses})
        t0 = time.time()
        p = subprocess.run(self.cmd, shell=isinstance(self.cmd, str), cwd=self.cwd,
                           env=bot_environment(),
                           input=payload + "\n",
                           stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                           universal_newlines=True, timeout=self.timeout)
        self.times.append(time.time() - t0)
        lines = [ln for ln in p.stdout.splitlines() if ln.strip()]
        if p.returncode or len(lines) != 1:
            raise RuntimeError("traditional mode must print exactly one JSON "
                               "line, got %r (stderr %r)" % (p.stdout, p.stderr[-500:]))
        out = json.loads(lines[0])
        check_response_model(out, self.require_model)
        resp = out["response"]
        self.responses.append(resp)
        return resp

    def close(self):
        pass


class KeepProcBot:
    """Long-running process: first turn full JSON, then one raw request/line."""

    MARK = ">>>BOTZONE_REQUEST_KEEP_RUNNING<<<"

    def __init__(self, cmd, cwd=None, timeout=60, require_model=False,
                 model_protocol="python"):
        self.p = subprocess.Popen(cmd, shell=isinstance(cmd, str), cwd=cwd,
                                  env=bot_environment(),
                                  stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                  stderr=subprocess.DEVNULL,
                                  universal_newlines=True, bufsize=1)
        self.timeout, self.require_model = timeout, require_model
        self.model_protocol = model_protocol
        self.play_turns = 0
        self.model_play_turns = 0
        self.model_sha = None
        self.lines = queue.Queue()
        def read_lines():
            for line in self.p.stdout:
                self.lines.put(line)
            self.lines.put(None)
        self.reader = threading.Thread(target=read_lines, daemon=True)
        self.reader.start()
        self.started = False
        self.times = []
        self.times_by_stage = {}

    def turn(self, request):
        if not self.started:
            payload = json.dumps({"requests": [request], "responses": []})
            self.started = True
        else:
            payload = json.dumps(request)
        t0 = time.time()
        self.p.stdin.write(payload + "\n")
        self.p.stdin.flush()
        line = self.lines.get(timeout=self.timeout)
        if not line:
            raise RuntimeError("bot process died")
        out = json.loads(line)
        fields = check_response_model(out, self.require_model, request.get("stage"),
                                      self.model_protocol)
        if request.get("stage") == "play":
            self.play_turns += 1
            if self.model_protocol == "cpp" and self.require_model:
                sha = fields["model_sha"]
                if self.model_sha is not None and self.model_sha != sha:
                    raise RuntimeError("C++ bot changed model SHA within a game")
                self.model_sha = sha
                self.model_play_turns += 1
        resp = out["response"]
        mark = self.lines.get(timeout=self.timeout)
        elapsed = time.time() - t0
        self.times.append(elapsed)
        self.times_by_stage.setdefault(request.get("stage", "unknown"), []).append(elapsed)
        if not mark or self.MARK not in mark:
            raise RuntimeError("missing keep-running marker: %r" % mark)
        return resp

    def close(self):
        try:
            self.p.kill()
            self.p.wait(timeout=5)
            self.p.stdin.close()
            self.p.stdout.close()
        except Exception:
            pass


def make_bot(driver, model, cwd=None, timeout=60, require_model=False):
    if driver == "inproc":
        return InProcBot(model, require_model=require_model)
    kind, _, cmd = driver.partition(":")
    if kind == "trad":
        return TradProcBot(cmd, cwd, timeout, require_model)
    if kind == "keep":
        return KeepProcBot(cmd + " --keep-running", cwd, timeout, require_model)
    if kind == "cpp":
        return KeepProcBot(cmd, cwd, timeout, require_model, model_protocol="cpp")
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
    setting = {}
    while out.get("command") == "request":
        (pid_s, req), = out["content"].items()
        pid = int(pid_s)
        if req.get("stage") == "deal":
            setting = {k: req.get("global", {}).get(k) for k in ("level", "tribute", "resist")}
        send = to_positional(req, pid) if positional else req
        try:
            resp = bots[pid].turn(send)
        except Exception as e:                # bot crashed -> record as error
            return {"error": {pid: "BOT_CRASH: %r" % e}, "turns": turns,
                    "setting": setting, "failed_request": send}
        log.append({"output": out})
        log.append({pid_s: {"response": resp, "verdict": "OK"}})
        out = judge.call({"log": log, "initdata": init_obj})
        turns += 1
        if turns > 2000:
            return {"error": {-1: "TOO_MANY_TURNS"}, "turns": turns}
    disp = out.get("display") or {}
    errs = disp.get("error")
    res = {"scores": {int(k): v for k, v in out["content"].items()},
           "turns": turns, "setting": setting}
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
    ap.add_argument("--driver-b", default=None,
                    help="team B driver (default = --driver); enables duplicate "
                         "head-to-head between two different bot programs")
    ap.add_argument("--shard", default="0/1",
                    help="K/N: play only deal pairs with index %% N == K (same deal "
                         "schedule in every shard, for parallel runs)")
    ap.add_argument("--bot-cwd", default=None, help="bot working directory (user data/ storage)")
    ap.add_argument("--bot-artifact", default=None,
                    help="local path of the exact executable used by a process driver; record its SHA-256")
    ap.add_argument("--timeout", type=float, default=60, help="local per-turn timeout in seconds")
    ap.add_argument("--require-model", action="store_true", help="fail on missing weights or model fallback")
    ap.add_argument("--report", default="", help="write JSON evidence to this file")
    ap.add_argument("--positional", action="store_true",
                    help="send FableDan's positional history format")
    ap.add_argument("--fail-fast", action="store_true")
    args = ap.parse_args()

    if args.games < 1 or args.timeout <= 0:
        ap.error("games and timeout must be positive")
    h2h = args.weights_b is not None or args.driver_b is not None
    if h2h and args.games % 2:
        ap.error("duplicate head-to-head requires an even number of games")
    try:
        shard_k, shard_n = (int(x) for x in args.shard.split("/"))
        if not 0 <= shard_k < shard_n:
            raise ValueError
    except ValueError:
        ap.error("--shard must be K/N with 0 <= K < N")
    driver_b = args.driver_b or args.driver
    if not os.path.isfile(args.judge):
        ap.error("judge not found: %s (no automatic fallback; specify --judge explicitly)" % args.judge)
    bot_artifact = None
    if args.bot_artifact:
        if not os.path.isfile(args.bot_artifact):
            ap.error("bot artifact not found: %s" % args.bot_artifact)
        with open(args.bot_artifact, "rb") as executable:
            artifact_bytes = executable.read()
        bot_artifact = {"path": os.path.abspath(args.bot_artifact),
                        "sha256": hashlib.sha256(artifact_bytes).hexdigest(),
                        "bytes": len(artifact_bytes)}
    judge = JudgeHost(args.judge)
    rng = random.Random(args.seed)
    cpp_driver = args.driver.startswith("cpp:")
    cpp_driver_b = driver_b.startswith("cpp:")
    if (cpp_driver and args.weights is not None) or \
            (cpp_driver_b and args.weights_b is not None):
        ap.error("cpp: loads its compiled/default or --model CMD weight path; --weights is for Python NPZ drivers")
    model_a = ("external_cpp", None) if cpp_driver else load_model(args.weights)
    if cpp_driver_b:
        model_b = ("external_cpp", None)
    elif args.weights_b:
        model_b = load_model(args.weights_b)
    elif args.driver_b is not None and cpp_driver:
        model_b = load_model(None)
    else:
        model_b = model_a
    times_by_team = {"a": [], "b": []}
    times_by_team_and_stage = {"a": {}, "b": {}}
    game_records = []
    played = 0
    n_err, by_scn, times, settings, failures = 0, {}, [], [], []
    cpp_play_turns, cpp_model_turns, cpp_model_shas = 0, 0, set()
    a_score = 0.0
    t_start = time.time()
    pair = None
    g = -1
    for g in range(args.games):
        if not h2h or g % 2 == 0:
            pair = make_initdata(rng, args.scenario)
        # Every shard draws the full schedule so deal i is identical everywhere.
        pair_index = g // 2 if h2h else g
        if pair_index % shard_n != shard_k:
            continue
        played += 1
        initdata, scn = pair
        a_seats = (0, 2) if g % 2 == 0 else (1, 3)
        bots = [make_bot(args.driver if p in a_seats else driver_b,
                         model_a if p in a_seats else model_b,
                         args.bot_cwd, args.timeout, args.require_model)
                for p in range(4)]
        try:
            res = run_game(judge, bots, initdata, positional=args.positional)
        finally:
            for p, b in enumerate(bots):
                times.extend(b.times)
                times_by_team["a" if p in a_seats else "b"].extend(b.times)
                for stage, stage_times in getattr(b, "times_by_stage", {}).items():
                    times_by_team_and_stage["a" if p in a_seats else "b"].setdefault(stage, []).extend(stage_times)
                if isinstance(b, KeepProcBot) and b.model_protocol == "cpp":
                    cpp_play_turns += b.play_turns
                    cpp_model_turns += b.model_play_turns
                    if b.model_sha:
                        cpp_model_shas.add(b.model_sha)
                b.close()
        settings.append(res.get("setting", {}))
        st = by_scn.setdefault(scn, [0, 0])
        st[0] += 1
        if "error" in res:
            n_err += 1
            failure = {"game": g, "scenario": scn, "error": res["error"], "initdata": initdata}
            if "failed_request" in res:
                failure["request"] = res["failed_request"]
            failures.append(failure)
            st[1] += 1
            print("game %d [%s] ERROR %s  initdata=%s" % (
                g, scn, res["error"], json.dumps(initdata)[:300]), flush=True)
            if args.fail_fast:
                break
        elif h2h:
            sc = res["scores"]
            diff = sc[a_seats[0]] - sc[(a_seats[0] + 1) % 4]
            a_score += diff
            game_records.append({"game": g, "pair": pair_index, "scenario": scn,
                                 "a_seats": list(a_seats), "a_diff": diff})
    el = time.time() - t_start
    g = played - 1     # report the games this shard actually played
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
    if args.report:
        with open(args.judge, "rb") as source:
            judge_sha = hashlib.sha256(source.read()).hexdigest()
        report = {"games": g + 1, "errors": n_err, "scenarios": by_scn,
                  "settings": settings, "failures": failures,
                  "judge": os.path.abspath(args.judge), "judge_sha256": judge_sha,
                  "driver": args.driver, "require_model": args.require_model,
                  "positional": args.positional, "elapsed_seconds": el,
                  "turn_seconds": {"p50": percentile(times, 50), "p99": percentile(times, 99),
                                   "max": max(times) if times else 0},
                  "avg_score_diff": a_score / (g + 1) if h2h else None,
                  "driver_b": driver_b, "shard": args.shard, "seed": args.seed,
                  "scenario": args.scenario, "game_records": game_records,
                  "turn_seconds_by_team": {
                      team: {"n": len(ts), "p50": percentile(ts, 50),
                             "p99": percentile(ts, 99), "max": max(ts) if ts else 0}
                      for team, ts in times_by_team.items()},
                  "turn_seconds_by_team_and_stage": {
                      team: {stage: {"n": len(ts), "p50": percentile(ts, 50),
                                     "p99": percentile(ts, 99), "max": max(ts)}
                             for stage, ts in stages.items() if ts}
                      for team, stages in times_by_team_and_stage.items()},
                  "timing_scope": "host wall clock including IPC and CPU contention; not BotZone CPU accounting"}
        if bot_artifact is not None:
            report["bot_artifact"] = bot_artifact
        if cpp_driver or cpp_driver_b:
            report["cpp_model"] = {"play_turns": cpp_play_turns,
                                   "model_selected_turns": cpp_model_turns,
                                   "model_sha_prefixes": sorted(cpp_model_shas)}
        os.makedirs(os.path.dirname(os.path.abspath(args.report)), exist_ok=True)
        with open(args.report, "w", encoding="utf-8") as output:
            json.dump(report, output, ensure_ascii=False, indent=2)
    sys.exit(1 if n_err else 0)


if __name__ == "__main__":
    main()
