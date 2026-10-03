"""Read-only diagnostic of frozen-model decisions in fresh live oracle games.

No training or parameter selection occurs. This reads only train/validation
demonstrations and fresh online states; held-out test shards are never opened.
"""
from __future__ import annotations

import argparse
import collections
import copy
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "train"))
from features import action_features, history_tokens, state_features
from model import CandidateModel, ModelConfig
from selection import choose_group_logmeanexp
from local_judge import atomic_json, debug_fields, expected_allocation, game_specs, model_identity, sha256_file
from oracle import Oracle
from probe import Probe


def referee_context(request, player):
    """Public prefix semantics used verbatim by the attached play referee."""
    leading = True
    for item in reversed(request["history"]):
        if isinstance(item, dict) and item.get("player") == player:
            break
        if isinstance(item, dict) and item.get("response") and item["response"][0]:
            leading = False
            break
    previous = next((item["response"] for item in reversed(request["history"])
                     if isinstance(item, dict) and item.get("response") and item["response"][0]), None)
    return leading, previous


def move_key(move):
    return tuple(sorted(move[0])), tuple(sorted(move[1]))


def demonstration_distribution(root):
    output = {}
    for split in ("train", "validation"):
        counts = collections.Counter()
        with (root / f"{split}.jsonl").open(encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                if row["stage"] != "play":
                    continue
                count = len(row["label"]["cards"])
                leading = row["features"]["leading"]
                counts["play_decisions"] += 1
                counts[f"cards_{count}"] += 1
                counts["cards_played"] += count
                counts["leading" if leading else "following"] += 1
                counts[("leading_" if leading else "following_") + f"cards_{count}"] += 1
        output[split] = dict(counts)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "models/bc-v1/best.pt")
    parser.add_argument("--model", type=Path, default=ROOT / "models/oxbot-bc-v1.bin")
    parser.add_argument("--probe", type=Path, default=ROOT / "bin/strength_probe")
    parser.add_argument("--network-probe", type=Path, default=ROOT / "bin/strength_network_probe")
    parser.add_argument("--games", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20263001)
    parser.add_argument("--report", type=Path, default=ROOT / "reports/strength_bc_v1_diagnostic.json")
    args = parser.parse_args()
    torch.set_num_threads(1)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model = CandidateModel(ModelConfig(**checkpoint["config"]))
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval()
    identity = model_identity(args.model)
    digest = hashlib.sha256()
    for tensor in model.state_dict().values():
        digest.update(tensor.detach().cpu().numpy().astype("<f4").tobytes())
    assert digest.hexdigest() == identity["payload_sha256"], "checkpoint differs from frozen deployed weights"
    counts = collections.Counter()
    choices = {"model": collections.Counter(), "rule": collections.Counter()}
    errors = {"state": 0.0, "action": 0.0, "score": 0.0}
    mismatches, examples, results = [], [], []
    oracle = Oracle()
    with Probe(args.probe, timeout=20) as probe, Probe(args.network_probe, timeout=20) as network:
        for spec in game_specs(args.games, args.seed):
            team = spec["index"] % 2
            init = {"seed": str(spec["seed"]), "level": [spec["level"]] * 2,
                    "tribute": spec["tribute"], "first": spec["first"], "last": spec["last"]}
            log, requests, responses, history = [], [[] for _ in range(4)], [[] for _ in range(4)], []
            data = [None] * 4
            for turn in range(1000):
                output = oracle.step({"initdata": init, "log": log})
                if "initdata" in output:
                    init = output["initdata"]
                if output["command"] == "finish":
                    assert not output.get("display", {}).get("error"), output
                    results.append({"seed": spec["seed"], "model_team": team, "scores": output["content"], "turns": turn})
                    break
                log.append({"output": output})
                player_text, request = next(iter(output["content"].items()))
                player = int(player_text)
                requests[player].append(copy.deepcopy(request))
                full = {"requests": requests[player], "responses": responses[player], "data": data[player]}
                role = "model" if player % 2 == team else "rule"
                response = probe.call(command="bot", input=full, model=str(args.model.resolve()) if role == "model" else "")
                assert "error" not in response, response
                debug = debug_fields(response["debug"])
                if request["stage"] == "play":
                    leading, previous = referee_context(request, player)
                    action = response["response"]
                    choice = choices[role]
                    size = len(action[0])
                    choice["decisions"] += 1
                    choice[f"cards_{size}"] += 1
                    choice["cards_played"] += size
                    choice["leading" if leading else "following"] += 1
                    choice[("leading_" if leading else "following_") + f"cards_{size}"] += 1
                    state = probe.call(command="state", input=full)
                    assert state["ok"], state
                    expected = expected_allocation(oracle, log, init, player, "play")
                    assert sorted(state["hand"]) == sorted(expected[player])
                    assert state["remaining_counts"] == [len(hand) for hand in expected]
                    counts["leading_checks"] += 1
                    if state["leading"] != leading:
                        mismatches.append({"kind": "leading", "seed": spec["seed"], "turn": turn})
                    if role == "model":
                        assert debug["policy"] == "model" and debug["model_sha"] == identity["payload_sha256"][:12], debug
                        generated = probe.call(command="generate", hand=state["hand"], level=request["global"]["level"],
                                               leading=leading, previous=previous, metadata=True)
                        moves = generated["moves"]
                        features = probe.call(command="features", input=full, moves=moves)
                        observation = {"hand": state["hand"], "player": player, "level": request["global"]["level"],
                                       "leading": leading, "remaining_counts": state["remaining_counts"], "history": history,
                                       "tribute": request["global"]["tribute"], "resist": request["global"]["resist"]}
                        state_py = state_features(observation)
                        tokens_py = history_tokens(observation)
                        action_py = np.asarray([action_features(move, meta["kind"], meta["key"], meta["secondary"],
                                                               observation["level"], len(state["hand"]))
                                                for move, meta in zip(moves, generated["types"], strict=True)])
                        errors["state"] = max(errors["state"], float(np.max(np.abs(state_py - features["state"]))))
                        errors["action"] = max(errors["action"], float(np.max(np.abs(action_py - features["actions"]))))
                        if tokens_py.tolist() != features["tokens"]:
                            mismatches.append({"kind": "tokens", "seed": spec["seed"], "turn": turn})
                        with torch.inference_mode():
                            scores_py = model(torch.from_numpy(tokens_py[None]), torch.tensor([len(tokens_py)]),
                                              torch.from_numpy(state_py[None]), torch.from_numpy(action_py[None]))[0].numpy()
                        actual = network.call(path=str(args.model.resolve()), tokens=features["tokens"],
                                              state=features["state"], actions=features["actions"])
                        assert actual["ok"] and actual["sha"] == identity["payload_sha256"], actual
                        scores_cpp = np.asarray(actual["scores"])
                        errors["score"] = max(errors["score"], float(np.max(np.abs(scores_py - scores_cpp))))
                        py_best, cpp_best = int(np.argmax(scores_py)), int(np.argmax(scores_cpp))
                        selected = next(i for i, move in enumerate(moves) if move_key(move) == move_key(action))
                        counts["model_decisions"] += 1
                        counts["candidates"] += len(moves)
                        counts["same_cpp_python_argmax"] += py_best == cpp_best
                        counts["same_online_python_argmax"] += selected == py_best
                        counts["same_online_cpp_argmax"] += selected == cpp_best
                        counts["pass_with_nonpass_available"] += size == 0 and any(move[0] for move in moves)
                        counts["single_with_multi_available"] += size == 1 and any(len(move[0]) > 1 for move in moves)
                        counts["finish_available"] += any(len(move[0]) == len(state["hand"]) for move in moves)
                        counts["finish_chosen"] += size == len(state["hand"])
                        if selected != py_best:
                            mismatches.append({"kind": "argmax", "seed": spec["seed"], "turn": turn,
                                               "selected": selected, "python": py_best, "cpp": cpp_best,
                                               "python_score_gap": float(scores_py[py_best] - scores_py[selected])})
                        # The group experiment is evaluated on the exact same
                        # state/candidate list without changing the game log.
                        # This is the online C++ vs Python selection parity
                        # guardrail; it does not alter frozen raw play.
                        group_response = probe.call(command="bot", input=full,
                                                    model=str(args.model.resolve()),
                                                    strategy="group-logmeanexp")
                        assert "error" not in group_response, group_response
                        group_debug = debug_fields(group_response["debug"])
                        assert group_debug.get("selection_strategy") == "group-logmeanexp-v1", group_debug
                        group_move = group_response["response"]
                        group_key = move_key(group_move)
                        group_selected = next(i for i, move in enumerate(moves) if move_key(move) == group_key)
                        group_py_best = choose_group_logmeanexp(moves, scores_py.tolist(), generated["types"],
                                                                observation["level"])[0]
                        counts["group_decisions"] += 1
                        counts["same_group_cpp_python"] += group_selected == group_py_best
                        if group_selected != group_py_best:
                            mismatches.append({"kind": "group_argmax", "seed": spec["seed"], "turn": turn,
                                               "selected": group_selected, "python": group_py_best})
                        if len(examples) < 12 and (leading or (size == 0 and any(move[0] for move in moves))):
                            counterfactual = probe.call(command="bot", input=full, model="")["response"]
                            top = sorted(range(len(moves)), key=lambda i: -scores_py[i])[:5]
                            examples.append({"seed": spec["seed"], "turn": turn, "player": player, "leading": leading,
                                             "hand": state["hand"], "remaining_counts": state["remaining_counts"],
                                             "model_move": action, "rule_move_same_state": counterfactual,
                                             "top_model_candidates": [{"move": moves[i], "kind": generated["types"][i]["kind"],
                                                                       "score": float(scores_py[i])} for i in top]})
                    history.append({"player": player, "action": action[0], "claim": action[1]})
                data[player] = response.get("data")
                responses[player].append(response["response"])
                log.append({player_text: {"response": response["response"]}})
            else:
                raise RuntimeError("game exceeded turn limit")
    result = {"status": "complete", "scope": "frozen model; fresh online diagnostic games and train/validation descriptive statistics only",
              "weights_changed": False, "test_data_opened": False, "seed": args.seed, "games": results,
              "model": identity, "checkpoint_sha256": sha256_file(args.checkpoint), "probe_sha256": sha256_file(args.probe),
              "network_probe_sha256": sha256_file(args.network_probe), "oracle_sha256": oracle.sha256,
              "checks": dict(counts), "maximum_absolute_errors": errors, "mismatches": mismatches,
              "online_action_counts": {key: dict(value) for key, value in choices.items()}, "examples": examples,
              "demonstration_counts": demonstration_distribution(ROOT / "data/processed/njupt")}
    atomic_json(args.report, result)
    print(json.dumps({key: result[key] for key in ("status", "checks", "maximum_absolute_errors", "mismatches", "online_action_counts", "demonstration_counts")}))


if __name__ == "__main__":
    main()
