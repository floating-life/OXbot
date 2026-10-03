"""Compare deployed C++ encoders with Python for public information sets."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import random
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "tools"), str(ROOT / "train")]
from features import state_features, history_tokens, action_features
from probe import Probe


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe", type=Path, default=ROOT / "bin/core_probe")
    parser.add_argument("--report", type=Path, default=ROOT / "reports/feature_parity.json")
    args = parser.parse_args()
    rng = random.Random(20261001)
    maximum, cases, candidates = 0., 0, 0
    with Probe(args.probe) as probe:
        for i in range(52):
            deck = list(range(108))
            rng.shuffle(deck)
            size = 1 + i % 27
            hand = deck[:size]
            history = []
            for j, card in enumerate(deck[size:size + i % 65]):
                history.append({"player": j % 4, "action": [card], "claim": [card] if j % 3 else None})
                if j % 2 == 0:
                    history.append({"player": (j+1) % 4, "action": [], "claim": []})
            # Enough pass events to cover history truncation independently of
            # the finite deck. No extra face is counted in the public tally.
            if i == 51:
                history.extend({"player": j % 4, "action": [], "claim": []} for j in range(100))
            observation = {"hand": hand, "player": i % 4, "level": "A234567890JQK"[i % 13],
                           "leading": i % 2 == 0, "tribute": i % 3, "resist": i % 7 == 0,
                           "remaining_counts": [rng.randrange(28) for _ in range(4)], "history": history}
            generated = probe.call(command="generate", hand=hand, level=observation["level"], leading=True)
            assert "error" not in generated, generated
            moves = generated["moves"] + [[[], []]]
            actual = probe.call(command="features", observation=observation, moves=moves)
            assert "error" not in actual, actual
            assert history_tokens(observation).tolist() == actual["tokens"]
            errors = [float(np.max(np.abs(state_features(observation) - actual["state"])))]
            for move, meta, vector in zip(moves, actual["types"], actual["actions"], strict=True):
                expected = action_features(move, meta["kind"], meta["key"], meta["secondary"], observation["level"], size)
                errors.append(float(np.max(np.abs(expected - vector))))
            maximum = max(maximum, *errors)
            assert max(errors) < 1e-7, errors
            cases += 1
            candidates += len(moves)
    report = {"status": "passed", "observations": cases, "actions": candidates,
              "max_absolute_error": maximum, "scope": "C++/Python feature and history-token equality"}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
