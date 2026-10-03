"""Local CPU decision benchmarks over valid ordinary-JSON fixture histories.

Use --process to include a fresh executable and model load on every sample.
The measurements are host wall-clock milliseconds, not BotZone CPU accounting.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import platform
import random
import subprocess
import threading
import time
import traceback

from local_judge import (ROOT, GameFailure, atomic_json, debug_fields, latency_summary,
                         model_identity, parse_process_response, probe_peak_rss_kib, require, sha256_file)
from probe import Probe


def globals_for(level="2"):
    return {"level": level, "tribute": 0, "first": None, "last": None, "resist": False,
            "tribute_cards": {}, "return_cards": {}}


def event(player, cards):
    return {"player": player, "response": [list(cards), list(cards)]}


def play_request(events, level="2"):
    require(len(events) <= 4, "fixture_history_too_long")
    return {"stage": "play", "global": globals_for(level), "history": [{}] * (4 - len(events)) + list(events),
            "done": [], "pass_on": -1}


def initial_input(player, hand, events=(), level="2"):
    require(len(hand) == 27 and len(set(hand)) == 27, "fixture_hand_invalid")
    return {"requests": [{"stage": "deal", "your_id": player, "deliver": list(hand), "global": globals_for(level)},
                         play_request(events, level)], "responses": [[]]}


def fixtures():
    dense = list(range(16)) + list(range(54, 65))
    shuffled = list(range(108))
    random.Random(20261001).shuffle(shuffled)
    cases = [
        {"name": "dense_two_wildcards_lead", "input": initial_input(0, dense)},
        {"name": "random_lead", "input": initial_input(0, shuffled[:27])},
        {"name": "follow_four_card_bomb", "input": initial_input(3, dense,
            [event(0, [48, 49, 50, 51]), event(1, []), event(2, [])])},
        {"name": "follow_rocket", "input": initial_input(3, dense,
            [event(0, [52, 53, 106, 107]), event(1, []), event(2, [])])},
    ]
    long_follow = initial_input(3, dense, [event(0, [27]), event(1, []), event(2, [])])
    for card in range(28, 51):
        long_follow["responses"].append([[], []])
        long_follow["requests"].append(play_request([event(3, []), event(0, [card]), event(1, []), event(2, [])]))
    cases.append({"name": "long_history_dense_follow", "input": long_follow})
    long_lead = initial_input(0, list(range(27)))
    for card in range(25):
        long_lead["responses"].append([[card], [card]])
        long_lead["requests"].append(play_request([event(0, [card]), event(1, []), event(2, []), event(3, [])]))
    cases.append({"name": "long_history_two_card_lead", "input": long_lead})
    return cases


def prior_move(payload):
    for item in reversed(payload["requests"][-1]["history"]):
        if isinstance(item, dict) and item.get("response", [[], []])[0]:
            return item["response"]
    return None


def process_rss_kib(pid):
    """Read one child process' VmHWM, never the parent's cumulative children usage."""
    if platform.system() != "Linux":
        return None
    try:
        for line in Path(f"/proc/{pid}/status").read_text().splitlines():
            if line.startswith("VmHWM:"):
                return int(line.split()[1])
    except (OSError, ValueError):
        return None
    return None


def run_process(executable, model_path, payload, timeout, cwd):
    process = subprocess.Popen([str(executable), "--model", model_path], stdin=subprocess.PIPE,
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, cwd=cwd)
    peak = [None]
    stop = threading.Event()

    def sample_memory():
        while not stop.is_set():
            value = process_rss_kib(process.pid)
            if value is not None:
                peak[0] = max(peak[0] or 0, value)
            time.sleep(0.001)

    sampler = threading.Thread(target=sample_memory, daemon=True)
    sampler.start()
    try:
        stdout, stderr = process.communicate(input=json.dumps(payload) + "\n", timeout=timeout)
    except subprocess.TimeoutExpired:
        process.kill()
        stdout, stderr = process.communicate()
        raise GameFailure("benchmark_process_timeout", {"stdout": stdout[:2000], "stderr": stderr[:2000]})
    finally:
        stop.set()
        sampler.join(timeout=1)
    return process.returncode, stdout, stderr, peak[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    choice = parser.add_mutually_exclusive_group(required=True)
    choice.add_argument("--model", type=Path)
    choice.add_argument("--rule-baseline", action="store_true")
    parser.add_argument("--probe", type=Path, default=Path("bin/core_probe"))
    parser.add_argument("--process", type=Path)
    parser.add_argument("--embedded-process", action="store_true",
                        help="with --process/--model, pass :embedded: to the single-file binary")
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=2)
    parser.add_argument("--report", type=Path, default=Path("reports/bot_benchmark.json"))
    args = parser.parse_args()
    if args.repeats < 1 or args.warmup < 0 or args.timeout <= 0:
        parser.error("repeats and timeout must be positive; warmup must be nonnegative")
    if args.embedded_process and (not args.process or not args.model):
        parser.error("--embedded-process requires both --process and --model")
    report_path = args.report.resolve()
    failure_path = report_path.with_name(report_path.stem + ".failure.json")
    started = time.monotonic()
    report = {"schema": "oxbot-cpu-benchmark-v1", "status": "running", "case_results": [],
              "mode": "embedded_short_process" if args.embedded_process else ("short_process" if args.process else "offline_probe"),
              "embedded_process": args.embedded_process, "model": None,
              "repeats_per_case": args.repeats, "warmups_per_case": args.warmup,
              "latency_measurement": {"unit": "ms", "clock": "local host wall clock",
                  "includes": "startup/model load/decision/IPC" if args.process else "decision/IPC; process/model reused",
                  "excludes": "separate fixture state checks, candidate enumeration and legality checks",
                  "platform_equivalence": "not BotZone CPU-time accounting; no platform deadline claim"},
              "acceptance": "model provenance and legal outputs on targeted CPU fixtures only; no win-rate claim",
              "host": {"system": platform.system(), "release": platform.release(), "machine": platform.machine()},
              "failure": None}
    all_ms = []
    process_rss_samples = []
    current_case = None
    identity = None
    exit_code = 0
    try:
        identity = model_identity(args.model)
        model_path = identity["path"] if identity else ""
        report.update({"model": identity, "probe": str(args.probe.resolve()), "probe_sha256": sha256_file(args.probe),
                       "process": str(args.process.resolve()) if args.process else None,
                       "process_sha256": sha256_file(args.process) if args.process else None})
        atomic_json(report_path, report)
        with Probe(args.probe.resolve(), timeout=max(args.timeout, 5)) as probe:
            for current_case in fixtures():
                payload = current_case["input"]
                state = probe.call(command="state", input=payload)
                require(state.get("ok") is True, "benchmark_fixture_state_invalid", case=current_case["name"], state=state)
                level = payload["requests"][-1]["global"]["level"]
                previous = prior_move(payload)
                candidates = probe.call(command="generate", hand=state["hand"], level=level,
                                        previous=previous, leading=state["leading"])
                require(isinstance(candidates.get("moves"), list) and candidates["moves"], "benchmark_candidates_missing")
                latencies = []
                for index in range(args.warmup + args.repeats):
                    start = time.perf_counter()
                    if args.process:
                        process_model = ":embedded:" if args.embedded_process else model_path
                        returncode, stdout, stderr, process_rss = run_process(
                            args.process.resolve(), process_model, payload, args.timeout, ROOT)
                        if process_rss is not None:
                            process_rss_samples.append(process_rss)
                        result = parse_process_response(returncode, stdout, stderr,
                                                        failure_reason="benchmark_process_contract_failed")
                    else:
                        result = probe.call(command="bot", input=payload, model=model_path)
                    elapsed = (time.perf_counter() - start) * 1000
                    require(not result.get("error"), "benchmark_adapter_failed", response=result)
                    debug = debug_fields(result.get("debug"))
                    expected_policy = "model" if identity else "rule_fallback"
                    require(debug.get("policy") == expected_policy and debug.get("legality_fallback") == "0",
                            "benchmark_policy_mismatch_or_fallback", expected=expected_policy, debug=debug)
                    if identity:
                        require(debug.get("model_sha") == identity["payload_sha256"][:12], "benchmark_model_sha_mismatch", debug=debug)
                    legality = probe.call(command="validate", move=result.get("response"), hand=state["hand"],
                                          level=level, previous=previous, leading=state["leading"])
                    require(legality.get("ok") is True, "benchmark_action_illegal", response=result, check=legality)
                    if index >= args.warmup:
                        latencies.append(elapsed)
                all_ms.extend(latencies)
                report["case_results"].append({"name": current_case["name"], "request_count": len(payload["requests"]),
                    "input_utf8_bytes": len(json.dumps(payload).encode()), "hand_cards": len(state["hand"]),
                    "candidate_count": len(candidates["moves"]), "latency_ms": latency_summary(latencies),
                    "policy": expected_policy, "legality_fallbacks": 0,
                    "model_sha_prefix": identity["payload_sha256"][:12] if identity else None})
                print(f"{current_case['name']}: {len(candidates['moves'])} candidates, p99={latency_summary(latencies)['p99']:.3f} local wall-ms", flush=True)
            report["offline_probe_peak_rss_kib"] = probe_peak_rss_kib(probe)
            report["short_process_vm_hwm_kib"] = {"samples": len(process_rss_samples),
                "max": max(process_rss_samples) if process_rss_samples else None,
                "method": "/proc/<pid>/status VmHWM sampler per child" if args.process else None,
                "scope": "short process only; no cumulative RUSAGE_CHILDREN" if args.process else None}
        report["status"] = "passed"
    except Exception as exc:
        exit_code = 1
        failure = {"reason": exc.reason if isinstance(exc, GameFailure) else type(exc).__name__,
                   "details": exc.details if isinstance(exc, GameFailure) else {"exception": str(exc)[:2000]},
                   "case": copy.deepcopy(current_case), "traceback": traceback.format_exc(limit=8)}
        atomic_json(failure_path, failure)
        report["status"] = "failed"
        report["failure"] = {"path": str(failure_path), "reason": failure["reason"]}
    report.update({"seconds": time.monotonic() - started, "latency_ms": latency_summary(all_ms), "report": str(report_path)})
    atomic_json(report_path, report)
    print(json.dumps(report, ensure_ascii=False, allow_nan=False))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
