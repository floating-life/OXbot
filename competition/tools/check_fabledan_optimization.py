#!/usr/bin/env python3
"""Compare optimized inference with a preserved pre-change C++ probe.

Checks exact scores on growing, sliding, repeated and replaced histories,
reloads, and checksum rejection. Timings are sequential host measurements.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import statistics
import sys
import tempfile
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
from probe import Probe  # noqa: E402


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--before-probe", type=Path, required=True)
    ap.add_argument("--probe", type=Path, required=True)
    ap.add_argument("--weights", type=Path, required=True)
    ap.add_argument("--feature-dim", type=int, choices=(80, 224), default=80)
    ap.add_argument("--report", type=Path, required=True)
    args = ap.parse_args()
    args.report.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(20261008)
    history = rng.integers(0, 48, 540).tolist()
    cases = [("grow_%d" % n, history[:n], k) for n, k in
             [(2, 1), (17, 7), (64, 16), (127, 32), (256, 33), (511, 64), (512, 256)]]
    cases += [("repeat", history[:512], 17), ("slide", history[7:519], 31),
              ("shorten", history[7:135], 8),
              ("replace_prefix", [47 - history[7]] + history[8:135], 9),
              ("reload", history[:256], 5)]
    rows = []
    with Probe(args.before_probe.resolve(), timeout=30) as before, \
            Probe(args.probe.resolve(), timeout=30) as after:
        for name, tokens, count in cases:
            req = dict(path=str(args.weights.resolve()), tokens=tokens,
                       features=rng.normal(size=(count, args.feature_dim)).astype(np.float32).tolist(),
                       reuse=True, reload=name == "reload")
            ref = before.call(**req)
            actual = after.call(**req)
            if not ref.get("ok") or not actual.get("ok") or ref.get("error") or actual.get("error"):
                raise RuntimeError(f"{name}: probe failed: {ref}, {actual}")
            if ref["scores"] != actual["scores"] or ref["sha"] != actual["sha"]:
                raise RuntimeError(f"{name}: scores or payload SHA changed")
            rows.append(dict(name=name, tokens=len(tokens), candidates=count, exact_scores=True,
                             before_ms=ref["milliseconds"], after_ms=actual["milliseconds"]))
        loads = []
        # Force actual reloads, keeping IPC timing separate from load timing.
        for _ in range(7):
            req = dict(path=str(args.weights.resolve()), tokens=[0, 1],
                       features=[[0.0] * args.feature_dim], reload=True)
            start = time.perf_counter()
            before.call(**req)
            before_ms = 1000 * (time.perf_counter() - start)
            start = time.perf_counter()
            result = after.call(**req)
            loads.append(dict(before_roundtrip_ms=before_ms,
                              after_roundtrip_ms=1000 * (time.perf_counter() - start),
                              after_load_ms=result["load_milliseconds"]))
        with tempfile.TemporaryDirectory(dir=args.report.resolve().parent) as folder:
            damaged = Path(folder) / "corrupt.fbd"
            blob = bytearray(args.weights.read_bytes())
            blob[-1] ^= 1
            damaged.write_bytes(blob)
            bad = after.call(path=str(damaged), reload=True)
            if bad.get("ok") or bad.get("status") != "payload_sha_mismatch":
                raise RuntimeError(f"corrupt weights accepted: {bad}")
            recovered = after.call(path=str(args.weights.resolve()), reload=True)
            if not recovered.get("ok"):
                raise RuntimeError("reload after rejection failed")
    report = dict(status="passed", exact_cases=rows, checksum_rejection=True,
                  reload_after_failure=True, cold_loads=loads,
                  before_roundtrip_median_ms=statistics.median(r["before_roundtrip_ms"] for r in loads),
                  after_roundtrip_median_ms=statistics.median(r["after_roundtrip_ms"] for r in loads),
                  weights_sha256=digest(args.weights), before_probe_sha256=digest(args.before_probe),
                  probe_sha256=digest(args.probe), feature_dim=args.feature_dim,
                  timing_scope="sequential local wall clock; not BotZone CPU accounting")
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
