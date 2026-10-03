"""Generate auditable self-play/candidate-Q rollouts with the official judge.

This is an offline data generator only.  It deliberately reuses the supplied
referee through :mod:`local_judge` and the C++ ``core_probe`` feature/rule
contract.  It never changes a release model or a BotZone package.

The normal trajectory stores the acting information set, every legal move,
the C++ feature vectors, and the selected move.  With ``--counterfactuals``
greater than zero, a bounded number of candidates at each recorded state are
forced once and the rest of the game is completed with the same fixed policy.
Those sibling trajectories produce candidate-level Monte-Carlo targets in the
acting team's point-margin perspective.  They are policy-evaluation labels,
not claims of an exact optimal Q function.

Example (from the repository root)::

    python tools/generate_selfplay.py --games 2 --seed 20261020 \
      --output data/selfplay/candidate_rollouts.jsonl \
      --report reports/selfplay_candidate_rollouts.json \
      --counterfactuals 2 --max-states 8

The default policy is the deterministic rule policy.  Pass ``--model`` to
evaluate a versioned C++ model instead.  ``--policy random`` is available for
coverage smoke tests; it is seeded and remains legal because choices come
from the C++ candidate generator.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
import tempfile
import time
from typing import Any, Iterable

# ``python tools/generate_selfplay.py`` should work without PYTHONPATH setup.
ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

from local_judge import DEFAULT_JUDGE, GameFailure, Oracle, Probe, score_teams  # noqa: E402


SCHEMA = "oxbot-selfplay-candidate-v1"
FEATURE_VERSION = "oxbot-observation-v1"
RULES_CONTRACT = "botzone-corrected-fa63589d-v1"
# ``rules_contract`` above is the model/C++ ABI contract and intentionally
# remains unchanged.  Rollout labels also carry the exact referee identity so
# an attachment-boundary dataset cannot be mixed with an online-official one.
OFFICIAL_JUDGE_SHA256 = "910cba94244106b68535b8bee67631b476241a9924bd1789dacbbf217fb1e895"
LEGACY_ATTACHMENT_JUDGE_SHA256 = "fa63589d3f69ce9127093cec417f1635d8cc17205d80e70f03e44bc6809fc622"
OFFICIAL_ROLLOUT_RULES_CONTRACT = "botzone-official-910cba94-v1"
LEGACY_ROLLOUT_RULES_CONTRACT = "botzone-attachment-fa63589d-v1"
PLAY_LIMIT = 1000


def rollout_contract_for_sha(sha256: str) -> str:
    if sha256 == OFFICIAL_JUDGE_SHA256:
        return OFFICIAL_ROLLOUT_RULES_CONTRACT
    if sha256 == LEGACY_ATTACHMENT_JUDGE_SHA256:
        return LEGACY_ROLLOUT_RULES_CONTRACT
    raise ValueError(
        "unrecognized judge SHA256; refusing unlabeled rollout contract: " + sha256
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", dir=path.parent, prefix=path.name + ".", suffix=".tmp", delete=False
        ) as stream:
            temporary = Path(stream.name)
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def card_array(value: Any) -> list[int]:
    if not isinstance(value, list) or any(type(card) is not int or not 0 <= card < 108 for card in value):
        raise ValueError("invalid card array")
    return list(value)


def canonical_move(value: Any) -> list[list[int]]:
    """Normalize [] and [[], []] to the feature/protocol move shape."""
    if value == []:
        return [[], []]
    if not isinstance(value, list) or len(value) != 2:
        raise ValueError("play response is not a two-array move")
    return [card_array(value[0]), card_array(value[1])]


def move_key(move: Any) -> tuple[tuple[int, ...], tuple[int, ...]]:
    normalized = canonical_move(move)
    # Card order is not semantically meaningful to the referee.  Sorting here
    # also makes matching a model response to the generator robust to harmless
    # ordering differences while retaining the original move in the record.
    return tuple(sorted(normalized[0])), tuple(sorted(normalized[1]))


def response_for_move(move: Any, stage: str) -> list[Any]:
    normalized = canonical_move(move)
    if stage == "play":
        return normalized
    if stage in ("deal", "tribute", "return"):
        if stage == "deal" or not normalized[0]:
            return []
        return [normalized[0][0]]
    raise ValueError("unknown stage")


def model_path_for_probe(path: Path | None) -> str:
    """Convert a Windows model path when the ELF probe is bridged through WSL."""
    if path is None:
        return ""
    resolved = path.resolve()
    if os.name == "nt" and len(resolved.drive) == 2:
        return "/mnt/" + resolved.drive[0].lower() + resolved.as_posix()[2:]
    return str(resolved)


def selected_model_for_seat(model: str, seat: int, model_team: str) -> str:
    if not model:
        return ""
    if model_team == "all" or seat % 2 == int(model_team):
        return model
    return ""


def matching_candidate(response: Any, moves: list[Any]) -> int:
    target = move_key(response)
    matches = [index for index, move in enumerate(moves) if move_key(move) == target]
    if not matches:
        raise GameFailure("policy_response_not_legal", {"response": response, "candidate_count": len(moves)})
    return matches[0]


def game_spec(index: int, seed: int) -> dict[str, Any]:
    rng = random.Random(seed)
    # Keep the same legal first/last topology as local_judge.game_specs.  Its
    # one RNG is advanced once per earlier game, so reconstruct that prefix
    # instead of reseeding every index (the deal seed itself is seed+index).
    first = last = 0
    for _ in range(index + 1):
        first = rng.randrange(4)
        last = (first + rng.choice((1, 3))) % 4
    return {
        "index": index,
        "seed": seed + index,
        "level": "234567890JQKA"[index % 13],
        "tribute": index % 3,
        "first": first,
        "last": last,
    }


def make_init(spec: dict[str, Any]) -> dict[str, Any]:
    # A non-zero tribute requires different first/last seats.  game_spec uses
    # the same seeded choice as local_judge's stress harness.
    first = int(spec["first"])
    last = int(spec["last"])
    return {
        "seed": str(spec["seed"]),
        "level": [spec["level"], spec["level"]],
        "tribute": int(spec["tribute"]),
        "first": first,
        "last": last,
    }


class SelfPlay:
    def __init__(
        self,
        oracle: Oracle,
        probe: Probe,
        *,
        policy: str,
        model: str,
        model_team: str,
        strategy: str,
        counterfactuals: int,
        max_states: int,
        rollout_contract: str,
        oracle_sha256: str,
        rng: random.Random,
    ) -> None:
        if policy not in ("rule", "model", "random"):
            raise ValueError("policy must be rule, model, or random")
        if policy == "model" and not model:
            raise ValueError("--policy model requires --model")
        self.oracle = oracle
        self.probe = probe
        self.policy = policy
        self.model = model
        self.model_team = model_team
        self.strategy = strategy
        self.counterfactuals = counterfactuals
        self.max_states = max_states
        self.rollout_contract = rollout_contract
        self.oracle_sha256 = oracle_sha256
        self.rng = rng

    @staticmethod
    def _score_reward(scores: tuple[int, int], seat: int) -> dict[str, int]:
        margin = int(scores[seat % 2] - scores[1 - seat % 2])
        return {"team_margin": margin, "team_win": 1 if margin > 0 else -1}

    def _bot(self, full_input: dict[str, Any], seat: int, *, branch: bool = False) -> dict[str, Any]:
        # ``--policy rule`` is intentionally a real rule baseline even if a
        # model path is present in an outer experiment configuration.  The
        # model policy is opt-in and applies only to the requested team.
        selected = selected_model_for_seat(self.model, seat, self.model_team) if self.policy == "model" else ""
        # Random policy is applied only at play states.  Stage exchanges still
        # use the deterministic C++ rules so every branch remains a valid game.
        return self.probe.call(command="bot", input=full_input, model=selected, strategy=self.strategy)

    def _choose_play(
        self,
        full_input: dict[str, Any],
        state: dict[str, Any],
        moves: list[Any],
        rng: random.Random,
        ) -> tuple[int, dict[str, Any] | None]:
        if self.policy == "random":
            return rng.randrange(len(moves)), None
        response = self._bot(full_input, int(state["player"]))
        if response.get("error"):
            raise GameFailure("bot_adapter_failed", {"response": response})
        self._check_model_response(response, int(state["player"]))
        return matching_candidate(response.get("response"), moves), response

    def _check_model_response(self, response: dict[str, Any], seat: int) -> None:
        if self.policy != "model" or not selected_model_for_seat(self.model, seat, self.model_team):
            return
        debug = response.get("debug", "")
        fields = dict(item.split("=", 1) for item in str(debug).split(";") if "=" in item)
        if fields.get("policy") != "model":
            # A model run that silently falls back to rules must not be mixed
            # into candidate-Q labels advertised as model policy evaluation.
            # Baseline-team seats are intentionally exempt.
            raise GameFailure("model_policy_fallback", {"seat": seat, "debug": debug})

    def _features(self, full_input: dict[str, Any], moves: list[Any]) -> dict[str, Any]:
        result = self.probe.call(command="features", input=full_input, moves=moves)
        if "error" in result:
            raise GameFailure("feature_encoding_failed", {"error": result.get("error")})
        state = result.get("state", [])
        tokens = result.get("tokens", [])
        if (len(state) != 128 or len(tokens) > 256 or
                any(type(value) not in (int, float) or not math.isfinite(float(value)) for value in state) or
                any(type(value) is not int or not 0 < value < 128 for value in tokens)):
            raise GameFailure("feature_shape_invalid", {"state": len(result.get("state", [])), "tokens": len(result.get("tokens", []))})
        actions = result.get("actions")
        if (not isinstance(actions, list) or len(actions) != len(moves) or any(len(row) != 128 for row in actions) or
                any(type(value) not in (int, float) or not math.isfinite(float(value)) for row in actions for value in row)):
            raise GameFailure("candidate_feature_shape_invalid", {"candidate_count": len(moves), "action_rows": len(actions or [])})
        if not isinstance(result.get("types"), list) and result.get("types") is not None:
            raise GameFailure("candidate_type_shape_invalid")
        return result

    def _stage_response(
        self,
        full_input: dict[str, Any],
        request: dict[str, Any],
        stage: str,
        *,
        forced_move: Any | None = None,
        capture: bool = False,
        rng: random.Random,
    ) -> tuple[list[Any], int | None, dict[str, Any] | None]:
        state = self.probe.call(command="state", input=full_input)
        if state.get("ok") is not True:
            raise GameFailure("state_rebuild_failed", {"state": state})
        if stage == "play":
            # During a continuation branch (or after the optional recording
            # cap) a deterministic model/rule response is enough; avoid
            # re-encoding every candidate when no row will be emitted.  A
            # random policy or forced action still needs the legal generator.
            if forced_move is None and not capture and self.policy != "random":
                response = self._bot(full_input, int(state["player"]))
                if response.get("error"):
                    raise GameFailure("bot_adapter_failed", {"response": response})
                self._check_model_response(response, int(state["player"]))
                return canonical_move(response.get("response")), None, None
            generated = self.probe.call(
                command="generate",
                hand=state["hand"],
                level=request["global"]["level"],
                leading=state["leading"],
                previous=state.get("previous"),
                metadata=True,
            )
            moves = generated.get("moves")
            if not isinstance(moves, list) or not moves:
                raise GameFailure("no_legal_candidates", {"state": state})
            features = self._features(full_input, moves) if capture else None
            if forced_move is not None:
                selected = matching_candidate(forced_move, moves)
                response = None
            else:
                selected, response = self._choose_play(full_input, state, moves, rng)
            payload = None
            if capture:
                payload = {
                    "state": features["state"],
                    "tokens": features["tokens"],
                    "actions": features["actions"],
                    "types": generated.get("types", []),
                    "moves": moves,
                    "selected": selected,
                    "player": state.get("player"),
                    "leading": state.get("leading"),
                    "remaining_counts": state.get("remaining_counts"),
                    "previous": state.get("previous"),
                    "policy_debug": (response or {}).get("debug"),
                }
            return response_for_move(moves[selected], stage), selected, payload
        # Deal/tribute/return are delegated to the C++ stage rules.  A branch
        # never forces these stages because candidate-Q labels target play.
        response = self._bot(full_input, int(state["player"]))
        if response.get("error"):
            raise GameFailure("bot_adapter_failed", {"response": response})
        raw = response.get("response")
        if not isinstance(raw, list):
            raise GameFailure("stage_response_not_array", {"response": response})
        return raw, None, None

    def _counterfactual_indices(self, count: int, selected: int, state_index: int) -> list[int]:
        if self.counterfactuals <= 0:
            return []
        # Always include the base action, then deterministic sibling samples.
        indexes = [selected]
        candidates = [i for i in range(count) if i != selected]
        local = random.Random((state_index + 1) * 0x9E3779B1 + self.counterfactuals)
        local.shuffle(candidates)
        indexes.extend(candidates[: max(0, self.counterfactuals - 1)])
        return indexes

    def _finish_or_step(
        self,
        initdata: dict[str, Any],
        log: list[dict[str, Any]],
    ) -> tuple[dict[str, Any], dict[str, Any] | None]:
        output = self.oracle.step({"initdata": copy.deepcopy(initdata), "log": copy.deepcopy(log)})
        return output, output.get("initdata") if isinstance(output, dict) else None

    def _continue_branch(
        self,
        initdata: dict[str, Any],
        log: list[dict[str, Any]],
        requests: list[list[dict[str, Any]]],
        responses: list[list[Any]],
        data: list[Any],
        player: int,
        response: list[Any],
        branch_seed: int,
    ) -> tuple[int, int]:
        """Force one pending response, then replay to finish.

        The first request has already been emitted and is present in ``log``;
        subsequent requests are obtained from the isolated official judge.
        ``requests/responses/data`` are private per-seat protocol state, so a
        branch can never mutate the parent trajectory.
        """
        local_init = copy.deepcopy(initdata)
        local_log = copy.deepcopy(log)
        local_requests = copy.deepcopy(requests)
        local_responses = copy.deepcopy(responses)
        local_data = copy.deepcopy(data)
        local_responses[player].append(copy.deepcopy(response))
        local_data[player] = None
        local_log.append({str(player): {"response": copy.deepcopy(response)}})
        rng = random.Random(branch_seed)
        for _ in range(PLAY_LIMIT):
            output, next_init = self._finish_or_step(local_init, local_log)
            if next_init is not None:
                local_init = next_init
            if output.get("command") == "finish":
                errors = output.get("display", {}).get("error", [])
                if isinstance(errors, dict):
                    errors = list(errors.values())
                if any(errors):
                    raise GameFailure("referee_rejected_branch", {"output": output})
                return score_teams(output["content"])
            if output.get("command") != "request" or not isinstance(output.get("content"), dict) or len(output["content"]) != 1:
                raise GameFailure("invalid_branch_referee_request", {"output": output})
            local_log.append({"output": output})
            key, request = next(iter(output["content"].items()))
            seat = int(key)
            local_requests[seat].append(copy.deepcopy(request))
            full = {"requests": local_requests[seat], "responses": local_responses[seat], "data": local_data[seat]}
            response, _, _ = self._stage_response(full, request, request["stage"], capture=False, rng=rng)
            local_responses[seat].append(copy.deepcopy(response))
            local_log.append({key: {"response": copy.deepcopy(response)}})
        raise GameFailure("branch_decision_limit_exceeded")

    def run_game(self, spec: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        init = make_init(spec)
        local_init = copy.deepcopy(init)
        log: list[dict[str, Any]] = []
        requests: list[list[dict[str, Any]]] = [[] for _ in range(4)]
        responses: list[list[Any]] = [[] for _ in range(4)]
        data: list[Any] = [None] * 4
        records: list[dict[str, Any]] = []
        turns = 0
        for turns in range(PLAY_LIMIT):
            output, next_init = self._finish_or_step(local_init, log)
            if next_init is not None:
                local_init = next_init
            if output.get("command") == "finish":
                errors = output.get("display", {}).get("error", [])
                if isinstance(errors, dict):
                    errors = list(errors.values())
                if any(errors):
                    raise GameFailure("referee_rejected_action", {"output": output})
                scores = score_teams(output["content"])
                for record in records:
                    reward = self._score_reward(scores, int(record["seat"]))
                    record["terminal"] = reward
                    record["terminal_scores"] = list(scores)
                return records, {
                    "index": spec["index"],
                    "seed": spec["seed"],
                    "level": spec["level"],
                    "tribute": spec["tribute"],
                    "first": spec["first"],
                    "last": init["last"],
                    "turns": turns,
                    "team_scores": list(scores),
                    "winner_team": 0 if scores[0] else 1,
                    "play_records": len(records),
                }
            if output.get("command") != "request" or not isinstance(output.get("content"), dict) or len(output["content"]) != 1:
                raise GameFailure("invalid_referee_request", {"output": output})
            log.append({"output": output})
            player_text, request = next(iter(output["content"].items()))
            player = int(player_text)
            requests[player].append(copy.deepcopy(request))
            full_input = {"requests": requests[player], "responses": responses[player], "data": data[player]}
            stage = request.get("stage")
            if stage == "play" and (self.max_states <= 0 or len(records) < self.max_states):
                # The same helper used by branches returns exact legal moves,
                # exact feature vectors, and the selected candidate.
                response, selected, payload = self._stage_response(full_input, request, stage, capture=True, rng=self.rng)
                assert selected is not None and payload is not None
                candidate_indices = self._counterfactual_indices(len(payload["moves"]), selected, len(records))
                q_values: dict[str, dict[str, int]] = {}
                for candidate in candidate_indices:
                    branch_scores = self._continue_branch(
                        local_init,
                        log,
                        requests,
                        responses,
                        data,
                        player,
                        response_for_move(payload["moves"][candidate], "play"),
                        branch_seed=int(spec["seed"]) * 1000003 + len(records) * 97 + candidate,
                    )
                    q_values[str(candidate)] = self._score_reward(branch_scores, player)
                records.append({
                    "schema": SCHEMA,
                    "rollout_rules_contract": self.rollout_contract,
                    "oracle_sha256": self.oracle_sha256,
                    "game_index": spec["index"],
                    "game_seed": spec["seed"],
                    "seat": player,
                    "team": player % 2,
                    "event_index": len(records),
                    "stage": "play",
                    "level": request["global"]["level"],
                    "tribute": request["global"].get("tribute", 0),
                    "leading": bool(payload.get("leading")),
                    "player_remaining_counts": payload.get("remaining_counts"),
                    "previous": payload.get("previous"),
                    "state": payload["state"],
                    "tokens": payload["tokens"],
                    "moves": payload["moves"],
                    "types": payload["types"],
                    "actions": payload["actions"],
                    "selected": selected,
                    "counterfactual_indices": candidate_indices,
                    "candidate_q": q_values,
                    "policy": self.policy,
                    "policy_debug": payload.get("policy_debug"),
                })
            else:
                response, _, _ = self._stage_response(full_input, request, stage, capture=False, rng=self.rng)
            data[player] = None
            responses[player].append(copy.deepcopy(response))
            log.append({player_text: {"response": copy.deepcopy(response)}})
        raise GameFailure("decision_limit_exceeded", {"turns": turns})


def write_record(stream: Any, record: dict[str, Any]) -> None:
    stream.write(json.dumps(record, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20261020)
    parser.add_argument("--policy", choices=("rule", "model", "random"), default="rule")
    parser.add_argument("--model", type=Path, help="versioned model binary; required for --policy model")
    parser.add_argument("--model-team", choices=("0", "1", "all"), default="all")
    parser.add_argument("--strategy", choices=("raw", "raw-pass-bias", "group-logmeanexp"), default="raw")
    parser.add_argument("--counterfactuals", type=int, default=0,
                        help="number of candidates per state to force (0 disables branches)")
    parser.add_argument("--max-states", type=int, default=0,
                        help="maximum recorded play states per game; 0 means all")
    parser.add_argument("--probe", type=Path, default=ROOT / "bin" / "core_probe")
    parser.add_argument("--judge", type=Path,
                        default=ROOT / "reports" / "official_judge_botzone_2026-10-02.py",
                        help="exact referee source; default is the verified BotZone official copy")
    parser.add_argument("--output", type=Path, default=ROOT / "data" / "selfplay" / "candidate_rollouts.jsonl")
    parser.add_argument("--report", type=Path, default=ROOT / "reports" / "selfplay_candidate_rollouts.json")
    args = parser.parse_args()
    if args.games <= 0 or args.counterfactuals < 0 or args.max_states < 0:
        parser.error("games must be positive; counterfactuals/max-states must be non-negative")
    if args.policy == "model" and args.model is None:
        parser.error("--policy model requires --model")
    if args.model is not None and not args.model.is_file():
        parser.error(f"model does not exist: {args.model}")
    if not args.probe.is_file():
        parser.error(f"probe does not exist: {args.probe}")
    if not args.judge.is_file():
        parser.error(f"judge does not exist: {args.judge}")

    oracle = Oracle(args.judge)
    rollout_contract = rollout_contract_for_sha(oracle.sha256)
    probe_path = args.probe.resolve()
    args.output = args.output.resolve()
    args.report = args.report.resolve()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    model = model_path_for_probe(args.model) if args.model else ""
    started = random.Random(args.seed)
    game_summaries: list[dict[str, Any]] = []
    counters: dict[str, int] = {"games": 0, "play_records": 0, "counterfactual_labels": 0, "failures": 0}
    generation_started = time.monotonic()
    temp_output = args.output.with_suffix(args.output.suffix + ".tmp")
    try:
        with temp_output.open("w", encoding="utf-8", newline="\n") as stream:
            with Probe(probe_path, timeout=10) as probe:
                for index in range(args.games):
                    spec = game_spec(index, args.seed)
                    try:
                        runner = SelfPlay(
                            oracle,
                            probe,
                            policy=args.policy,
                            model=model,
                            model_team=args.model_team,
                            strategy=args.strategy,
                            counterfactuals=args.counterfactuals,
                            max_states=args.max_states,
                            rollout_contract=rollout_contract,
                            oracle_sha256=oracle.sha256,
                            rng=random.Random(started.randrange(1 << 62)),
                        )
                        records, summary = runner.run_game(spec)
                        for record in records:
                            write_record(stream, record)
                        stream.flush()
                        counters["games"] += 1
                        counters["play_records"] += len(records)
                        counters["counterfactual_labels"] += sum(len(record["candidate_q"]) for record in records)
                        game_summaries.append(summary)
                    except Exception as exc:
                        counters["failures"] += 1
                        game_summaries.append({"index": index, "status": "failed", "reason": str(exc)[:1000]})
                        # Do not publish a partial dataset as a successful
                        # training source.  The report remains useful for
                        # diagnosis, and the temporary JSONL is removed below.
                        raise
        os.replace(temp_output, args.output)
    except Exception as exc:
        if temp_output.exists():
            temp_output.unlink()
        report = {
            "schema": SCHEMA,
            "status": "failed",
            "failure": {"type": type(exc).__name__, "message": str(exc)[:2000]},
            "oracle_sha256": oracle.sha256,
            "rollout_rules_contract": rollout_contract,
            "probe": str(probe_path),
            "probe_sha256": sha256_file(probe_path),
            "games_requested": args.games,
            "counters": counters,
            "elapsed_seconds": time.monotonic() - generation_started,
            "policy": args.policy,
            "model": str(args.model.resolve()) if args.model else None,
            "test_used": False,
        }
        atomic_json(args.report, report)
        print(json.dumps({"status": "failed", "report": str(args.report), "error": str(exc)}, ensure_ascii=False))
        return 1

    output_sha = sha256_file(args.output)
    report = {
        "schema": SCHEMA,
        "status": "complete",
        "feature_version": FEATURE_VERSION,
        "rules_contract": RULES_CONTRACT,
        "oracle": str(args.judge.resolve()),
        "oracle_sha256": oracle.sha256,
        "rollout_rules_contract": rollout_contract,
        "probe": str(probe_path),
        "probe_sha256": sha256_file(probe_path),
        "policy": args.policy,
        "model": str(args.model.resolve()) if args.model else None,
        "model_sha256": sha256_file(args.model.resolve()) if args.model else None,
        "model_team": args.model_team if args.model else None,
        "selection_strategy": args.strategy,
        "model_fallback_fail_closed": args.policy == "model",
        "games_requested": args.games,
        "seed": args.seed,
        "counterfactuals": args.counterfactuals,
        "max_states_per_game": args.max_states,
        "candidate_q_definition": (
            "fixed-policy continuation from the same information set and deal; "
            "team_margin=final team score minus opponent team score, team_win=sign(team_margin)"
        ),
        "counterfactual_limit_is_sampling": True,
        "test_used": False,
        "output": str(args.output),
        "output_sha256": output_sha,
        "counters": counters,
        "elapsed_seconds": time.monotonic() - generation_started,
        "games": game_summaries,
    }
    atomic_json(args.report, report)
    print(json.dumps({"status": report["status"], "output": str(args.output), "report": str(args.report),
                      "games": counters["games"], "play_records": counters["play_records"],
                      "counterfactual_labels": counters["counterfactual_labels"],
                      "output_sha256": output_sha}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
