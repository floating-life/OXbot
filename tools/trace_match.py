"""Trace one model-vs-rule full deal for BC policy diagnosis.

This is deliberately descriptive: it never turns action counts into a
strength claim.  Private hands and requests are written to the trace artifact
because the artifact is local evaluation data, not a BotZone upload.
"""
from __future__ import annotations

import argparse
import collections
import copy
import json
from pathlib import Path
import random
import time

from local_judge import (ROOT, expected_allocation, model_identity, require, score_teams,
                         sha256_file)
from oracle import DEFAULT_JUDGE, Oracle
from probe import Probe


def classify(probe, claim):
    return probe.call(command="classify", claim=claim)


def card_names(cards):
    # Keep the trace portable without relying on a second card encoder.  IDs
    # remain authoritative; this short string is only a quick visual hint.
    return [int(card) for card in cards]


def run(args):
    identity = model_identity(args.model)
    oracle = Oracle(args.judge)
    trace_path = args.trace.resolve()
    probe = Probe(args.probe.resolve(), timeout=args.probe_timeout)
    init = {"seed": str(args.seed), "level": [args.level, args.level], "tribute": args.tribute,
            "first": args.first, "last": args.last}
    initdata = init
    log, requests, responses = [], [[] for _ in range(4)], [[] for _ in range(4)]
    data = [None] * 4
    events = []
    policy_counts = collections.Counter()
    type_counts = collections.Counter()
    seat_counts = collections.defaultdict(collections.Counter)
    anomalies = []
    started = time.monotonic()
    player = -1
    try:
        for turn in range(1000):
            output = oracle.step({"initdata": initdata, "log": log})
            if "initdata" in output:
                initdata = output["initdata"]
            if output["command"] == "finish":
                scores = score_teams(output["content"])
                result = {"schema": "oxbot-model-vs-rule-trace-v1", "status": "passed",
                    "model": identity, "oracle_sha256": oracle.sha256, "probe_sha256": sha256_file(args.probe),
                    "seed": args.seed, "level": args.level, "tribute": args.tribute,
                    "first": args.first, "last": args.last, "model_team": args.model_team,
                    "scores": output["content"], "team_scores": scores,
                    "model_team_won": scores[int(args.model_team)] > 0,
                    "turns": turn, "seconds": time.monotonic() - started,
                    "policy_counts": dict(policy_counts), "move_type_counts": dict(type_counts),
                    "seat_move_type_counts": {str(k): dict(v) for k, v in seat_counts.items()},
                    "anomalies": anomalies, "events": events}
                trace_path.parent.mkdir(parents=True, exist_ok=True)
                trace_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
                return result
            require(output.get("command") == "request", "unexpected_referee_command", output=output)
            log.append({"output": output})
            player_text, request = next(iter(output["content"].items()))
            player = int(player_text)
            requests[player].append(copy.deepcopy(request))
            full_input = {"requests": requests[player], "responses": responses[player], "data": data[player]}
            state = probe.call(command="state", input=full_input)
            require(state.get("ok") is True, "state_rebuild_failed", state=state)
            allocation = expected_allocation(oracle, log, initdata, player, request["stage"])
            require(sorted(state["hand"]) == sorted(allocation[player]), "private_hand_mismatch",
                    expected=allocation[player], actual=state)
            selected_model = str(args.model.resolve()) if player % 2 == int(args.model_team) else ""
            response = probe.call(command="bot", input=full_input, model=selected_model, strategy=args.strategy)
            require(not response.get("error"), "adapter_failed", response=response)
            debug = response.get("debug", "")
            fields = {part.split("=", 1)[0]: part.split("=", 1)[1]
                      for part in debug.split(";") if "=" in part}
            policy = fields.get("policy", "missing")
            policy_counts[policy] += 1
            event = {"turn": turn, "player": player, "team": player % 2, "stage": request["stage"],
                     "policy": policy, "debug": debug, "hand_before": len(state["hand"]),
                     "hand_cards": card_names(state["hand"]), "response": response.get("response"),
                     "remaining_counts": state.get("remaining_counts"), "leading": state.get("leading"),
                     "candidate_count": None}
            if request["stage"] == "play":
                require("previous" in state, "probe_state_missing_previous",
                        hint="rebuild the core probe before tracing")
                previous = state["previous"]
                candidates = probe.call(command="generate", hand=state["hand"],
                                        level=request["global"]["level"], previous=previous,
                                        leading=state["leading"])
                # Candidate generation must use the same replayed comparison
                # target as the policy.  Previously this diagnostic passed
                # previous=None, overstating legal options on follow turns.
                event["candidate_count"] = len(candidates.get("moves", []))
                event["previous"] = previous
                action = response["response"][0] if response["response"] else []
                claim = response["response"][1] if response["response"] else []
                kind = classify(probe, claim).get("kind", "unknown")
                event["move_type"] = kind
                event["action_count"] = len(action)
                event["claim"] = claim
                type_counts[kind] += 1
                seat_counts[player][kind] += 1
                if policy == "model":
                    if not action and state["leading"]:
                        anomalies.append({"turn": turn, "player": player, "kind": "leading_pass"})
                    if not action and len(state["hand"]) <= 1:
                        anomalies.append({"turn": turn, "player": player, "kind": "single_card_pass"})
                    if len(state["hand"]) > 1 and len(action) == len(state["hand"]):
                        event["finishing_move"] = True
                    if len(state["hand"]) == 1 and len(action) == 0:
                        anomalies.append({"turn": turn, "player": player, "kind": "one_card_not_finished"})
            events.append(event)
            data[player] = response.get("data")
            responses[player].append(response["response"])
            log.append({player_text: {"response": response["response"]}})
        raise RuntimeError("decision_limit_exceeded")
    finally:
        probe.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--probe", type=Path, default=Path("bin/core_probe"))
    parser.add_argument("--judge", type=Path, default=DEFAULT_JUDGE)
    parser.add_argument("--seed", type=int, default=20261001)
    parser.add_argument("--level", default="2", choices=list("234567890JQKA"))
    parser.add_argument("--tribute", type=int, default=0, choices=(0, 1, 2))
    parser.add_argument("--first", type=int, default=0, choices=(0, 1, 2, 3))
    parser.add_argument("--last", type=int, default=1, choices=(0, 1, 2, 3))
    parser.add_argument("--model-team", default="0", choices=("0", "1"))
    parser.add_argument("--strategy", choices=("raw", "raw-pass-bias", "group-logmeanexp"), default="raw")
    parser.add_argument("--probe-timeout", type=float, default=5)
    parser.add_argument("--trace", type=Path, default=Path("reports/bc_v1_model_vs_rule_trace.json"))
    args = parser.parse_args()
    if args.first % 2 == args.last % 2:
        parser.error("first and last must be opposite teams")
    result = run(args)
    print(json.dumps({key: value for key, value in result.items() if key != "events"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
