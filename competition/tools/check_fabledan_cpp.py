#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run a small NumPy-vs-C++ parity check for a versioned FableDan weight file.

The probe is intentionally a line-oriented process so the same executable can
also be used while integrating the backend into a long-running bot.  Build it
from the repository root with::

    g++ -std=c++17 -O2 -Icore/include \
        core/src/fabledan_network.cpp tools/fabledan_probe.cpp \
        -o bin/fabledan_probe

Then run this checker with an NPZ source and the matching exported FBDN file.
The NumPy side is the existing ``fabledan.model_np.NumpyModel``; only the
weight container differs, so a successful check exercises the entire
token/context/hand/Q path.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Dict, List

import numpy as np


ROOT = Path(__file__).resolve().parents[2]


def _cases(feature_dim=80) -> List[Dict[str, object]]:
    rng = np.random.default_rng(20261003)
    out = []
    for length, actions in ((2, 1), (17, 7), (64, 16), (256, 32), (512, 256)):
        tokens = rng.integers(0, 48, size=length, dtype=np.int64).tolist()
        features = rng.normal(size=(actions, feature_dim)).astype(np.float32)
        out.append({"tokens": tokens, "features": features.tolist()})
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--npz", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--report", type=Path, help="optional JSON parity report")
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument(
        "--tolerance",
        type=float,
        default=5e-3,
        help="maximum absolute Q error (float16 default is 5e-3)",
    )
    args = parser.parse_args()

    import sys

    sys.path.insert(0, str(ROOT / "competition"))
    sys.path.insert(0, str(ROOT / "tools"))
    from fabledan.model_np import NumpyModel
    from probe import Probe

    model = NumpyModel(str(args.npz))
    requests = []
    expected = []
    cases = _cases(model.cfg["feat_dim"])
    for case in cases:
        tokens = [int(value) for value in case["tokens"]]
        features = np.asarray(case["features"], dtype=np.float32)
        expected.append(model.q_values(tokens, features).astype(np.float64))
        request = {
            "path": str(args.weights.resolve()),
            "tokens": tokens,
            "features": features.tolist(),
        }
        requests.append(request)

    # Reuse OXbot's bounded bridge so Windows Python can test the WSL ELF
    # executable produced by tools/build.ps1, as well as native probes.
    with Probe(args.probe.resolve(), timeout=args.timeout) as probe:
        results = [probe.call(**request) for request in requests]

    checks = []
    maximum = 0.0
    for index, (actual, target) in enumerate(zip(results, expected)):
        if not actual.get("ok") or "scores" not in actual:
            raise SystemExit(f"probe case {index} failed: {actual}")
        scores = np.asarray(actual["scores"], dtype=np.float64)
        if scores.shape != target.shape:
            raise SystemExit(
                f"probe case {index} shape {scores.shape} != {target.shape}"
            )
        error = float(np.max(np.abs(scores - target))) if scores.size else 0.0
        maximum = max(maximum, error)
        checks.append(
            {
                "length": len(cases[index]["tokens"]),
                "candidates": int(scores.size),
                "max_absolute_error": error,
                "argmax_matches": bool(np.argmax(scores) == np.argmax(target)),
                "milliseconds": actual.get("milliseconds"),
                "sha": actual.get("sha"),
            }
        )
        if error > args.tolerance:
            raise SystemExit(
                f"probe case {index} max error {error:.8g} > {args.tolerance:.8g}"
            )

    report = {
        "status": "passed",
        "feature_dim": model.cfg["feat_dim"],
        "feature_version": model.cfg.get("feature_version", 1),
        "max_absolute_error": maximum,
        "tolerance": args.tolerance,
        "npz_sha256": hashlib.sha256(args.npz.read_bytes()).hexdigest(),
        "weights_sha256": hashlib.sha256(args.weights.read_bytes()).hexdigest(),
        "probe_sha256": hashlib.sha256(args.probe.read_bytes()).hexdigest(),
        "cases": checks,
        "timing_scope": "local inference only; excludes model load and IPC",
    }
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
