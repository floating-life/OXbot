"""Replay complete deals through the unmodified supplied referee.

The offline probe reuses a C++ process/model for test throughput.  --process
starts the real Bot executable for every decision and includes model loading
in its local wall-clock timing.  Neither timing mode measures BotZone CPU time.
"""
from __future__ import annotations

import argparse
import collections
import concurrent.futures
import copy
import hashlib
import json
import math
import multiprocessing
import os
from pathlib import Path
import platform
import random
import struct
import subprocess
import tempfile
import time
import traceback

from oracle import Oracle, DEFAULT_JUDGE
from probe import Probe

ROOT = Path(__file__).resolve().parents[1]


class GameFailure(RuntimeError):
    def __init__(self, reason, details=None):
        super().__init__(reason)
        self.reason = reason
        self.details = details or {}


def require(condition, reason, **details):
    if not condition:
        raise GameFailure(reason, details)


def atomic_json(path, value):
    """Publish one complete UTF-8 report; an interrupted write leaves no torn JSON."""
    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent,
                                         prefix=path.name + ".", suffix=".tmp", delete=False) as handle:
            temporary = Path(handle.name)
            json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fabledan_model_identity(path, content):
    """Verify versioned FableDan container; C++ checks tensor names/shapes."""
    require(len(content) >= 116, "model_header_length_invalid")
    version, dtype, header_bytes, tensor_count = struct.unpack_from("<IIII", content, 8)
    expected_version = {b"FBDN001\0": 1, b"FBDN002\0": 2}.get(content[:8])
    require(expected_version is not None and version == expected_version, "model_version_invalid")
    require(dtype in (1, 2), "model_dtype_invalid")
    require(116 <= header_bytes <= len(content), "model_header_length_invalid")
    require(tensor_count == 64, "model_tensor_count_invalid")
    config_names = ("d_model", "n_blocks", "n_heads", "qk_dim", "v_dim", "ffn_hidden",
                    "hand_hidden", "n_hand_layers", "q_hidden", "n_q_layers", "max_seq", "vocab", "feat_dim")
    config_values = struct.unpack_from("<13I", content, 24)
    feature_dim = 80 if version == 1 else 224
    require(config_values == (128, 4, 4, 64, 64, 512, 512, 3, 1024, 3, 512, 48, feature_dim),
            "model_config_invalid")
    rms_epsilon, payload_bytes = struct.unpack_from("<fI", content, 76)
    require(math.isfinite(rms_epsilon) and 0 < rms_epsilon < 1e-3, "model_config_invalid")
    require(header_bytes + payload_bytes == len(content), "model_payload_length_invalid")
    payload_sha = hashlib.sha256(content[header_bytes:]).hexdigest()
    require(payload_sha == content[84:116].hex(), "model_payload_sha_invalid")

    # The payload checksum excludes the tensor table. Check its lengths and
    # ranges too, so damaged metadata cannot pass the artifact identity gate.
    cursor, names, spans = 116, set(), []
    for _ in range(tensor_count):
        require(cursor + 4 <= header_bytes, "model_tensor_table_invalid")
        name_bytes, rank, _reserved = struct.unpack_from("<HBB", content, cursor)
        cursor += 4
        require(1 <= name_bytes <= 512 and 1 <= rank <= 4, "model_tensor_table_invalid")
        require(cursor + rank * 4 + 16 + name_bytes <= header_bytes, "model_tensor_table_invalid")
        shape = struct.unpack_from("<" + "I" * rank, content, cursor)
        require(all(1 <= dimension <= 8192 for dimension in shape), "model_tensor_shape_invalid")
        count = math.prod(shape)
        require(count <= 16 * 1024 * 1024, "model_tensor_shape_invalid")
        cursor += rank * 4
        offset, byte_count = struct.unpack_from("<QQ", content, cursor)
        cursor += 16
        name = content[cursor:cursor + name_bytes]
        cursor += name_bytes
        require(name not in names, "model_tensor_name_invalid")
        names.add(name)
        require(byte_count == count * (2 if dtype == 1 else 4), "model_tensor_length_invalid")
        require(offset <= payload_bytes and byte_count <= payload_bytes - offset, "model_tensor_bounds_invalid")
        spans.append((offset, byte_count))
    require(cursor == header_bytes, "model_tensor_table_invalid")
    expected_offset = 0
    for offset, byte_count in sorted(spans):
        require(offset == expected_offset, "model_tensor_payload_gap")
        expected_offset += byte_count
    require(expected_offset == payload_bytes, "model_tensor_payload_length_invalid")
    return {"path": str(path), "bytes": len(content), "file_sha256": hashlib.sha256(content).hexdigest(),
            "payload_sha256": payload_sha, "architecture": "FableDan", "feature_version": version,
            "rules_contract": None, "manifest_selection_default": "raw", "format": content[:7].decode("ascii"),
            "format_version": version, "dtype": "fp16" if dtype == 1 else "fp32",
            "config": dict(zip(config_names, config_values)), "tensor_count": tensor_count}


def model_identity(path):
    if not path:
        return None
    path = Path(path).resolve()
    content = path.read_bytes()
    if content[:8] in (b"FBDN001\0", b"FBDN002\0"):
        return fabledan_model_identity(path, content)
    require(len(content) >= 12 and content[:8] == b"OXGDQ001", "model_header_magic_invalid", model=str(path))
    length = int.from_bytes(content[8:12], "little")
    require(0 < length <= 65536 and 12 + length <= len(content), "model_header_length_invalid")
    header = json.loads(content[12:12 + length])
    payload_sha = hashlib.sha256(content[12 + length:]).hexdigest()
    require(payload_sha == header.get("payload_sha256"), "model_payload_sha_invalid")
    selection = header.get("selection")
    if selection is None:
        selection_default = "raw"
    else:
        require(isinstance(selection, dict), "model_selection_manifest_invalid")
        selection_default = selection.get("default", "raw")
        require(selection_default in ("raw", "raw-pass-bias", "group-logmeanexp"),
                "model_selection_strategy_invalid")
    return {"path": str(path), "bytes": len(content), "file_sha256": hashlib.sha256(content).hexdigest(),
            "payload_sha256": payload_sha, "architecture": header.get("architecture"),
            "feature_version": header.get("feature_version"), "rules_contract": header.get("rules_contract"),
            "manifest_selection_default": selection_default}


def parse_process_response(returncode, stdout, stderr, *, failure_reason="bot_process_contract_failed"):
    """Accept one JSON response, optionally followed by BotZone's exact marker."""
    lines = stdout.splitlines()
    output_shape_ok = len(lines) == 1 or (
        len(lines) == 2 and lines[1] == ">>>BOTZONE_REQUEST_KEEP_RUNNING<<<")
    require(returncode == 0 and output_shape_ok and not stderr, failure_reason,
            returncode=returncode, stdout=stdout[:2000], stderr=stderr[:2000])
    return json.loads(lines[0])


def debug_fields(value):
    require(isinstance(value, str), "debug_missing_or_not_string")
    result = {}
    for field in value.split(";"):
        if "=" in field:
            key, content = field.split("=", 1)
            require(key not in result, "duplicate_debug_field", field=key)
            result[key] = content
    return result


def model_for_seat(model, model_team, player):
    return str(model) if model and (str(model_team) == "all" or player % 2 == int(model_team)) else ""


def score_teams(scores):
    require(isinstance(scores, dict) and set(scores) == {"0", "1", "2", "3"}, "invalid_score_shape", scores=scores)
    require(all(type(value) is int and 0 <= value <= 3 for value in scores.values()), "invalid_score_value", scores=scores)
    require(scores["0"] == scores["2"] and scores["1"] == scores["3"], "partner_scores_disagree", scores=scores)
    # The referee awards the same points to each partner.  Count a team's
    # score once, never add its two players and accidentally double the result.
    teams = [scores["0"], scores["1"]]
    require((teams[0] == 0) != (teams[1] == 0), "invalid_winning_team_scores", scores=scores)
    return teams


def paired_scores(first_records, second_records):
    """Compare complementary model seats only after the exact deal keys match.

    This helper is the contract for a later duplicate evaluator.  Margins are
    paired in model perspective; individual correlated legs are not treated
    as independent Bernoulli trials or used for a Wilson confidence interval.
    """
    second = {}
    for record in second_records:
        key = record["deal_key"]
        require(key not in second, "duplicate_deal_key", deal_key=key)
        second[key] = record
    margins = []
    seen = set()
    for record in first_records:
        key = record["deal_key"]
        require(key not in seen, "duplicate_deal_key", deal_key=key)
        seen.add(key)
        if key not in second:
            continue
        other = second[key]
        require({str(record.get("model_team")), str(other.get("model_team"))} == {"0", "1"},
                "duplicate_pair_requires_complementary_model_teams", deal_key=key)
        one = record["team_scores"]
        two = other["team_scores"]
        first_sign = 1 if str(record["model_team"]) == "0" else -1
        second_sign = 1 if str(other["model_team"]) == "0" else -1
        margins.append(first_sign * (one[0] - one[1]) + second_sign * (two[0] - two[1]))
    return {"pairs": len(margins), "unmatched_first": len(first_records) - len(margins),
            "unmatched_second": len(second_records) - len(margins),
            "positive_pairs": sum(value > 0 for value in margins),
            "tied_pairs": sum(value == 0 for value in margins),
            "negative_pairs": sum(value < 0 for value in margins),
            "total_model_point_margin": sum(margins),
            "mean_model_point_margin_per_pair": sum(margins) / len(margins) if margins else None,
            "inference": "descriptive paired points only; no strength threshold or confidence claim"}


def expected_allocation(oracle, log, initdata, player, stage):
    if stage != "play":
        return initdata["allocation"]
    scope = oracle.module()
    full = {"initdata": copy.deepcopy(initdata), "log": copy.deepcopy(log) +
            [{str(player): {"response": [[], []]}}]}
    return scope["recover_state"](full)


def expected_hand(oracle, log, initdata, player, stage):
    return expected_allocation(oracle, log, initdata, player, stage)[player]


def run_game(oracle, probe, seed, level, tribute, first, last, executable=None, allocation=None,
             *, model=None, model_team="all", require_model=False, expected_model_sha="", process_cwd=None,
             embedded_process=False, strategy="raw"):
    init = {"seed": str(seed), "level": [level, level], "tribute": tribute, "first": first, "last": last}
    if allocation is not None:
        init["allocation"] = allocation
    log, requests, responses = [], [[] for _ in range(4)], [[] for _ in range(4)]
    initdata, full_input, output, response = init, None, None, None
    data = [None] * 4
    latencies, play_latencies = [], []
    stages, policies, play_policies, model_statuses, model_shas = (collections.Counter() for _ in range(5))
    fallbacks, budget_exceedances = collections.Counter(), collections.Counter()
    player = -1
    turn = -1
    try:
        for turn in range(1000):
            output = oracle.step({"initdata": initdata, "log": log})
            if "initdata" in output:
                initdata = output["initdata"]
            if output["command"] == "finish":
                errors = output.get("display", {}).get("error", [])
                if isinstance(errors, dict):
                    errors = list(errors.values())
                require(not any(errors), "referee_rejected_action", output=output)
                scores = score_teams(output["content"])
                contract = {"allocation": initdata["allocation"], "level": level, "tribute": tribute,
                            "first": first, "last": last}
                deal_key = hashlib.sha256(json.dumps(contract, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
                return {"seed": seed, "level": level, "tribute": tribute, "first": first, "last": last,
                        "selection_strategy": strategy,
                        "deal_key": deal_key, "model_team": str(model_team) if model else None,
                        "turns": turn, "scores": output["content"], "team_scores": scores,
                        "winner_team": 0 if scores[0] else 1, "ms": latencies, "play_ms": play_latencies,
                        "stages": dict(stages), "policies": dict(policies), "play_policies": dict(play_policies),
                        "model_statuses": dict(model_statuses), "model_shas": dict(model_shas),
                        "fallbacks": dict(fallbacks), "local_budget_exceedances": dict(budget_exceedances)}
            require(output.get("command") == "request", "unexpected_referee_command", output=output)
            require(isinstance(output.get("content"), dict) and len(output["content"]) == 1,
                    "invalid_referee_request", output=output)
            log.append({"output": output})
            player_text, request = next(iter(output["content"].items()))
            player = int(player_text)
            require(0 <= player < 4, "invalid_requested_player", player=player)
            requests[player].append(copy.deepcopy(request))
            full_input = {"requests": requests[player], "responses": responses[player], "data": data[player]}
            stage = request["stage"]
            stages[stage] += 1
            state = probe.call(command="state", input=full_input)
            require(state.get("ok") is True, "state_rebuild_failed", state=state)
            expected = expected_allocation(oracle, log, initdata, player, stage)
            require(sorted(state["hand"]) == sorted(expected[player]), "private_hand_mismatch",
                    expected=expected[player], actual=state)
            if "remaining_counts" in state:
                require(state["remaining_counts"] == [len(hand) for hand in expected], "remaining_counts_mismatch",
                        expected=[len(hand) for hand in expected], actual=state["remaining_counts"])

            selected_model = model_for_seat(model, model_team, player)
            process_model = ":embedded:" if embedded_process and selected_model else ""
            start = time.perf_counter()
            if executable:
                # --model "" disables the executable's compiled default on
                # baseline seats. Absolute model paths make cwd irrelevant.
                completed = subprocess.run([str(executable), "--model", process_model if embedded_process else selected_model,
                                            "--strategy", strategy],
                                           input=json.dumps(full_input) + "\n", encoding="utf-8",
                                           capture_output=True, cwd=process_cwd,
                                           timeout=2 if not responses[player] else 1)
                response = parse_process_response(completed.returncode, completed.stdout, completed.stderr)
            else:
                response = probe.call(command="bot", input=full_input, model=selected_model, strategy=strategy)
            elapsed_ms = (time.perf_counter() - start) * 1000
            latencies.append(elapsed_ms)
            if stage == "play":
                play_latencies.append(elapsed_ms)
            budget_ms = 2000 if not responses[player] else 1000
            if elapsed_ms > budget_ms:
                budget_exceedances["first_decision" if not responses[player] else "later_decision"] += 1
            require(isinstance(response, dict) and not response.get("error"), "adapter_failed", response=response)
            require(isinstance(response.get("response"), list), "response_missing_or_not_array", response=response)
            debug = debug_fields(response.get("debug"))
            require(debug.get("selection_strategy") == strategy + "-v1", "wrong_selection_strategy_used",
                    expected=strategy + "-v1", actual=debug.get("selection_strategy"), debug=debug)
            policy = debug.get("policy", "missing")
            policies[policy] += 1
            model_statuses[debug.get("model_status", "missing")] += 1
            if debug.get("model_sha"):
                model_shas[debug["model_sha"]] += 1
                if selected_model and expected_model_sha:
                    require(debug["model_sha"] == expected_model_sha[:12], "wrong_model_payload_used",
                            expected=expected_model_sha[:12], actual=debug["model_sha"])
            if debug.get("legality_fallback") == "1":
                fallbacks["legality_recheck_fallback"] += 1
            if stage == "play":
                require(policy in ("model", "rule_fallback"), "invalid_play_policy", debug=debug)
                play_policies[policy] += 1
                if policy == "model":
                    require(bool(selected_model), "unexpected_model_on_rule_seat", debug=debug)
                    require(bool(debug.get("model_sha")), "model_policy_missing_payload_sha", debug=debug)
                if policy == "rule_fallback":
                    fallbacks["rule_policy_play"] += 1
                    fallbacks["model_requested_but_not_used" if selected_model else "expected_rule_baseline_play"] += 1
                if selected_model and require_model:
                    require(policy == "model", "required_model_fell_back", debug=debug)
            else:
                require(policy == "stage_rules", "invalid_nonplay_policy", debug=debug)
            data[player] = response.get("data")
            responses[player].append(response["response"])
            log.append({player_text: {"response": response["response"]}})
        raise GameFailure("decision_limit_exceeded")
    except Exception as exc:
        details = {"seed": seed, "level": level, "tribute": tribute, "first": first, "last": last,
                   "turn": turn, "player": player, "model_team": str(model_team) if model else None,
                   "initdata": initdata, "full_input": full_input, "last_output": output, "last_response": response}
        if isinstance(exc, GameFailure):
            details.update(exc.details)
            raise GameFailure(exc.reason, details) from exc
        details.update({"exception_type": type(exc).__name__, "exception": str(exc)[:2000]})
        raise GameFailure("execution_exception", details) from exc


def probe_peak_rss_kib(probe):
    if platform.system() != "Linux":
        return None
    try:
        for line in Path(f"/proc/{probe.process.pid}/status").read_text().splitlines():
            if line.startswith("VmHWM:"):
                return int(line.split()[1])
    except (OSError, ValueError):
        pass
    return None


def game_specs(count, seed):
    rng = random.Random(seed)
    for index in range(count):
        first = rng.randrange(4)
        last = (first + rng.choice((1, 3))) % 4
        yield {"index": index, "seed": seed + index, "level": "234567890JQKA"[index % 13],
               "tribute": index % 3, "first": first, "last": last}


def run_batch(config, specs):
    games = []
    result = {"games": games, "peak_probe_rss_kib": None, "failure": None}
    try:
        oracle = Oracle(Path(config["judge"]))
        with Probe(Path(config["probe"]), timeout=config["probe_timeout"]) as probe:
            for spec in specs:
                game = run_game(oracle, probe, spec["seed"], spec["level"], spec["tribute"], spec["first"], spec["last"],
                                Path(config["process"]) if config["process"] else None,
                                model=config["model"], model_team=config["model_team"],
                                require_model=config["require_model"], expected_model_sha=config["model_payload_sha"],
                                process_cwd=config["process_cwd"], embedded_process=config["embedded_process"],
                                strategy=config["strategy"])
                game["index"] = spec["index"]
                games.append(game)
            result["peak_probe_rss_kib"] = probe_peak_rss_kib(probe)
    except Exception as exc:
        result["failure"] = {"reason": exc.reason if isinstance(exc, GameFailure) else type(exc).__name__,
                             "details": exc.details if isinstance(exc, GameFailure) else {"exception": str(exc)[:2000]},
                             "traceback": traceback.format_exc(limit=8)}
    return result


def run_batches(config, specs, workers, batch_size):
    batches = [specs[index:index + batch_size] for index in range(0, len(specs), batch_size)]
    if workers == 1:
        for batch in batches:
            result = run_batch(config, batch)
            yield result
            if result["failure"]:
                return
        return
    # Spawn avoids inheriting a live Probe reader thread or pipe.  Every batch
    # owns an Oracle and Probe and closes them before its result is returned.
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers,
            mp_context=multiprocessing.get_context("spawn")) as executor:
        waiting = iter(batches)
        pending = set()
        for _ in range(workers):
            batch = next(waiting, None)
            if batch is not None:
                pending.add(executor.submit(run_batch, config, batch))
        while pending:
            ready, pending = concurrent.futures.wait(pending, return_when=concurrent.futures.FIRST_COMPLETED)
            for future in ready:
                result = future.result()
                yield result
                if result["failure"]:
                    for other in pending:
                        other.cancel()
                    return
                batch = next(waiting, None)
                if batch is not None:
                    pending.add(executor.submit(run_batch, config, batch))


def latency_summary(values):
    if not values:
        return {"count": 0, "p50": None, "p95": None, "p99": None, "max": None}
    ordered = sorted(values)
    percentile = lambda p: ordered[min(len(ordered) - 1, max(0, math.ceil(len(ordered) * p) - 1))]
    return {"count": len(ordered), "p50": percentile(.5), "p95": percentile(.95), "p99": percentile(.99), "max": ordered[-1]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20261001)
    parser.add_argument("--workers", type=int, default=1, help="independent CPU workers, 1..8; no GPU work")
    parser.add_argument("--batch-size", type=int, default=25)
    parser.add_argument("--probe-timeout", type=float, default=5)
    parser.add_argument("--probe", type=Path, default=Path("bin/core_probe"))
    parser.add_argument("--judge", type=Path, default=DEFAULT_JUDGE)
    parser.add_argument("--process", type=Path, help="actual one-request executable supporting --model PATH")
    parser.add_argument("--embedded-process", action="store_true",
                        help="with --process/--model, pass :embedded: for model seats; external --model is identity only")
    parser.add_argument("--process-cwd", type=Path, default=ROOT)
    parser.add_argument("--model", type=Path, help="explicit versioned model binary; absent means rule baseline")
    parser.add_argument("--model-team", choices=("0", "1", "all"), default="all", help="team 0=seats 0/2, team 1=seats 1/3")
    parser.add_argument("--require-model", action="store_true", help="every configured model seat must use the model on play")
    parser.add_argument("--strategy", choices=("raw", "raw-pass-bias", "group-logmeanexp"), default="raw",
                        help="model candidate selection; raw preserves v1 behavior")
    parser.add_argument("--paired-report", type=Path, help="optional prior complementary-team report with identical deal keys")
    parser.add_argument("--report", type=Path, default=Path("reports/local_judge.json"))
    args = parser.parse_args()
    if args.games <= 0 or args.batch_size <= 0 or not 1 <= args.workers <= 8 or args.probe_timeout <= 0:
        parser.error("games/batch-size/probe-timeout must be positive; workers must be 1..8")
    if (args.require_model or args.model_team != "all" or args.paired_report) and not args.model:
        parser.error("--require-model, --model-team and --paired-report require --model")
    if args.embedded_process and (not args.process or not args.model):
        parser.error("--embedded-process requires both --process and --model")
    if args.paired_report and args.model_team == "all":
        parser.error("--paired-report requires --model-team 0 or 1")
    report_path = args.report.resolve()
    if args.paired_report and args.paired_report.resolve() == report_path:
        parser.error("--paired-report must differ from --report")
    failure_path = report_path.with_name(report_path.stem + ".failure.json")
    started = time.monotonic()
    counts, stages, policies, play_policies, statuses, shas, fallbacks, budgets, winners = (collections.Counter() for _ in range(9))
    ms, play_ms, records, peak_rss = [], [], [], []
    report = {"schema": "oxbot-local-judge-v2", "status": "running", "requested_games": args.games,
              "seed": args.seed, "workers": args.workers, "batch_size": args.batch_size,
              "mode": "embedded_short_process" if args.embedded_process else ("short_process" if args.process else "offline_probe"),
              "embedded_process": args.embedded_process, "model_team": args.model_team if args.model else None,
              "require_model": args.require_model, "selection_strategy": args.strategy,
              "selection_strategy_version": args.strategy + "-v1", "online_compatibility": "unverified",
              "validation_scope": "rule_and_state_stress_only" if not args.model else "model_legality_state_and_local_timing",
              "strength_claim": "none; training-model launch and win-rate gates are separate",
              "timing": {"unit": "ms", "clock": "local host wall clock", "workers": args.workers,
                         "includes": "process startup/model load/decision/IPC" if args.process else "decision/IPC; process and model reused",
                         "excludes": "referee execution and the separate state-oracle check",
                         "platform_equivalence": "not BotZone CPU-time measurements or a 1-second compliance claim"},
              "host": {"system": platform.system(), "release": platform.release(), "machine": platform.machine(),
                       "logical_cpus": os.cpu_count()}, "failure": None}

    def snapshot(final=False):
        report.update({"seconds": time.monotonic() - started, "counts": dict(counts), "stages": dict(stages),
                       "policies": dict(policies), "play_policies": dict(play_policies), "model_status_counts": dict(statuses),
                       "observed_model_payload_sha_prefix_counts": dict(shas), "fallback_counts": dict(fallbacks),
                       "local_wall_clock_budget_exceedances": dict(budgets), "winning_teams": dict(winners),
                       "max_probe_rss_kib": max(peak_rss) if peak_rss else None})
        if final:
            report["latency_ms"] = latency_summary(ms)
            report["play_latency_ms"] = latency_summary(play_ms)
            report["game_records"] = sorted(records, key=lambda item: item["index"])
            if args.model and args.model_team != "all":
                team = int(args.model_team)
                wins = sum(record["winner_team"] == team for record in records)
                margins = [record["team_scores"][team] - record["team_scores"][1 - team] for record in records]
                report["model_vs_rule"] = {"games": len(records), "wins": wins,
                    "win_rate": wins / len(records) if records else None,
                    "total_model_point_margin": sum(margins),
                    "inference": "descriptive single-seat-assignment results; use matching paired reports before a strength conclusion"}
        atomic_json(report_path, report)

    exit_code = 0
    try:
        oracle = Oracle(args.judge)
        identity = model_identity(args.model)
        report.update({"oracle_sha256": oracle.sha256, "oracle": str(oracle.path), "model": identity,
                       "probe": str(args.probe.resolve()), "probe_sha256": sha256_file(args.probe),
                       "process": str(args.process.resolve()) if args.process else None,
                       "process_sha256": sha256_file(args.process) if args.process else None})
        config = {"judge": str(oracle.path), "probe": str(args.probe.resolve()), "probe_timeout": args.probe_timeout,
                  "process": str(args.process.resolve()) if args.process else None, "process_cwd": str(args.process_cwd.resolve()),
                  "model": identity["path"] if identity else "", "model_team": args.model_team,
                  "require_model": args.require_model, "model_payload_sha": identity["payload_sha256"] if identity else "",
                  "embedded_process": args.embedded_process, "strategy": args.strategy}
        snapshot()
        last_progress = time.monotonic()
        last_printed = 0
        for batch in run_batches(config, list(game_specs(args.games, args.seed)), args.workers, args.batch_size):
            if batch["peak_probe_rss_kib"] is not None:
                peak_rss.append(batch["peak_probe_rss_kib"])
            for game in batch["games"]:
                counts["games"] += 1
                counts["decisions"] += game["turns"]
                stages.update(game["stages"])
                policies.update(game["policies"])
                play_policies.update(game["play_policies"])
                statuses.update(game["model_statuses"])
                shas.update(game["model_shas"])
                fallbacks.update(game["fallbacks"])
                budgets.update(game["local_budget_exceedances"])
                winners[str(game["winner_team"])] += 1
                ms.extend(game["ms"])
                play_ms.extend(game["play_ms"])
                records.append({key: game[key] for key in ("index", "seed", "level", "tribute", "first", "last", "deal_key",
                                                          "model_team", "turns", "team_scores", "winner_team")})
            if batch["failure"]:
                counts["failures"] += 1
                atomic_json(failure_path, batch["failure"])
                report["failure"] = {"path": str(failure_path), "reason": batch["failure"]["reason"],
                                     "seed": batch["failure"]["details"].get("seed")}
                report["status"] = "failed"
                exit_code = 1
                break
            now = time.monotonic()
            if counts["games"] - last_printed >= 100 or now - last_progress >= 15:
                print(f"{counts['games']}/{args.games} games, {counts['decisions']} decisions, {now - started:.1f}s", flush=True)
                snapshot()
                last_progress, last_printed = now, counts["games"]
        if not exit_code:
            require(counts["games"] == args.games, "incomplete_game_count")
            report["status"] = "passed"
        if args.paired_report and not exit_code:
            comparison = json.loads(args.paired_report.read_text(encoding="utf-8"))
            require(comparison.get("status") == "passed", "paired_report_not_passed")
            require(comparison.get("oracle_sha256") == report["oracle_sha256"], "paired_report_different_oracle")
            require(comparison.get("probe_sha256") == report["probe_sha256"] and
                    comparison.get("process_sha256") == report["process_sha256"], "paired_report_different_bot_binary")
            require(comparison.get("model", {}).get("payload_sha256") == identity["payload_sha256"], "paired_report_different_model")
            report["paired_scores"] = paired_scores(records, comparison.get("game_records", []))
    except Exception as exc:
        exit_code = 1
        report["status"] = "failed"
        failure = {"reason": exc.reason if isinstance(exc, GameFailure) else type(exc).__name__,
                   "details": exc.details if isinstance(exc, GameFailure) else {"exception": str(exc)[:2000]},
                   "traceback": traceback.format_exc(limit=8)}
        atomic_json(failure_path, failure)
        report["failure"] = {"path": str(failure_path), "reason": failure["reason"]}
    snapshot(final=True)
    summary = {key: value for key, value in report.items() if key != "game_records"}
    summary["game_records_count"] = len(records)
    summary["report"] = str(report_path)
    print(json.dumps(summary, ensure_ascii=False, allow_nan=False))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
