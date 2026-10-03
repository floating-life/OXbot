# -*- coding: utf-8 -*-
"""Parallel duplicate (seat-swapped) evaluation with a paired bootstrap CI.

Each deal is played twice with the teams swapped; the unit of observation is
the deal (pair), not the single game, so the confidence interval is honest.

    python tools/duplicate_eval.py --a ckpts/dmc1/latest.npz \
        --b ckpts/real-v2/best.npz --deals 1000 --workers 14 \
        --report reports/dmc1_vs_realv2.json

Spec for --a/--b: rule | random | <weights.npz> | <checkpoint.pt>.
Exit code 0 always on a finished run; read "verdict" in the JSON report.
"""

import argparse
import hashlib
import json
import math
import multiprocessing as mp
import os
import random
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

_AGENTS = {}


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def describe(spec):
    if os.path.isfile(spec):
        return {"spec": spec, "sha256": _sha256(spec)}
    return {"spec": spec}


def _init_worker(spec_a, spec_b, seed):
    # One model copy per worker; NumPy/Torch must stay single-threaded so
    # that N workers really use N cores.
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[var] = "1"
    from fabledan.evaluate import make_agent
    _AGENTS["a"] = make_agent(spec_a, seed)
    _AGENTS["b"] = make_agent(spec_b, seed + 1)


def play_deal(task):
    """Play one deal twice with the teams swapped; return A's rewards."""
    from fabledan.engine import play_round
    from fabledan.ring import sample_setting
    deal_index, deal_seed, ladder_frac = task
    out = []
    for a_seats in ((0, 2), (1, 3)):
        agents = [_AGENTS["a"]() if p in a_seats else _AGENTS["b"]() for p in range(4)]
        grng = random.Random(deal_seed)
        level, tmode = sample_setting(grng, ladder_frac)
        rewards, _, _ = play_round(agents, rng=grng, level=level, tribute_mode=tmode)
        out.append(float(rewards[a_seats[0]]))
    return deal_index, out


def paired_bootstrap(pair_scores, iters=10000, seed=0):
    """95% CI of the mean per-game score difference, resampling deals."""
    n = len(pair_scores)
    per_game = [s / 2.0 for s in pair_scores]
    mean = sum(per_game) / n
    rng = random.Random(seed)
    means = []
    for _ in range(iters):
        means.append(sum(per_game[rng.randrange(n)] for _ in range(n)) / n)
    means.sort()
    return mean, means[int(0.025 * iters)], means[int(0.975 * iters) - 1]


def wilson(wins, n, z=1.96):
    if n == 0:
        return 0.0, 0.0
    p = wins / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def summarize(pairs, iters=10000, seed=0):
    pair_scores = [a + b for a, b in pairs]
    mean, lo, hi = paired_bootstrap(pair_scores, iters, seed)
    deal_wins = sum(1 for s in pair_scores if s > 0)
    deal_ties = sum(1 for s in pair_scores if s == 0)
    games = [x for p in pairs for x in p]
    if lo > 0:
        verdict = "A_STRONGER"
    elif hi < 0:
        verdict = "A_WEAKER"
    else:
        verdict = "INCONCLUSIVE"
    return {
        "deals": len(pairs),
        "games": len(games),
        "avg_score_diff_per_game": mean,
        "ci95_per_game": [lo, hi],
        "game_win_rate": sum(1 for x in games if x > 0) / len(games),
        "deal_wins": deal_wins,
        "deal_ties": deal_ties,
        "deal_losses": len(pairs) - deal_wins - deal_ties,
        # Wilson over deals (independent unit), ties excluded.
        "deal_win_rate_wilson95": list(wilson(deal_wins, len(pairs) - deal_ties)),
        "verdict": verdict,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--a", required=True, help="candidate: rule|random|<npz>|<pt>")
    ap.add_argument("--b", required=True, help="opponent: rule|random|<npz>|<pt>")
    ap.add_argument("--deals", type=int, default=1000,
                    help="seat-swapped deal pairs (games = 2 x deals)")
    ap.add_argument("--seed", type=int, default=20261101,
                    help="fixed schedule seed; keep it constant across candidates")
    ap.add_argument("--ladder-frac", type=float, default=0.5,
                    help="share of deals at Botzone default (level 2, no tribute)")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("--bootstrap", type=int, default=10000)
    ap.add_argument("--report", default="", help="write JSON summary here")
    args = ap.parse_args()
    if args.deals < 2:
        ap.error("--deals must be >= 2")
    if not 0 <= args.ladder_frac <= 1:
        ap.error("--ladder-frac must be in [0, 1]")
    for spec in (args.a, args.b):
        if spec not in ("rule", "random") and not os.path.isfile(spec):
            ap.error("not found: %s" % spec)

    rng = random.Random(args.seed)
    tasks = [(i, rng.getrandbits(48), args.ladder_frac) for i in range(args.deals)]
    pairs = [None] * args.deals
    t0 = time.time()
    ctx = mp.get_context("spawn")
    with ctx.Pool(args.workers, initializer=_init_worker,
                  initargs=(args.a, args.b, args.seed)) as pool:
        done = 0
        for idx, res in pool.imap_unordered(play_deal, tasks, chunksize=4):
            pairs[idx] = res
            done += 1
            if done % max(1, args.deals // 10) == 0 or done == args.deals:
                partial = [p for p in pairs if p is not None]
                avg = sum(a + b for a, b in partial) / (2 * len(partial))
                print("  %d/%d deals  avg diff/game %+.3f  %.0fs"
                      % (done, args.deals, avg, time.time() - t0), flush=True)

    summary = summarize(pairs, args.bootstrap, args.seed)
    report = {"a": describe(args.a), "b": describe(args.b), "seed": args.seed,
              "ladder_frac": args.ladder_frac, "elapsed_seconds": time.time() - t0,
              **summary}
    print("A=%s vs B=%s: %+.3f/game, 95%% CI [%+.3f, %+.3f], %d deals -> %s"
          % (args.a, args.b, summary["avg_score_diff_per_game"],
             summary["ci95_per_game"][0], summary["ci95_per_game"][1],
             summary["deals"], summary["verdict"]))
    if args.report:
        os.makedirs(os.path.dirname(os.path.abspath(args.report)), exist_ok=True)
        with open(args.report, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
