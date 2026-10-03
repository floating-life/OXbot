"""Frozen-policy, paired-deal evaluation for post-training candidates.

No weights are trained, overwritten, or promoted here. Promotion is a report
gate, not an upload action. Pass counters are descriptive review signals, never
labels that a pass was wrong. The local engine still needs official-judge QA.
"""
from __future__ import annotations

import argparse
from collections import Counter
import copy
import hashlib
import json
from pathlib import Path
import random

import numpy as np

from .agents import RuleAgent
from .cards import is_wildcard
from .combos import PASS, SINGLE
from .encode import FEAT_DIM, encode_decision
from .engine import GuandanRound
from .ring import sample_setting


MIN_PROMOTION_PAIRS = 200


def _model_identity(model, backend):
    digest = hashlib.sha256()
    cfg = getattr(model, "cfg", {})
    if hasattr(cfg, "to_dict"):
        cfg = cfg.to_dict()
    digest.update(json.dumps(cfg, sort_keys=True, separators=(",", ":")).encode())
    weights = model.w if backend == "numpy" else model.state_dict()
    for name, value in sorted(weights.items()):
        if backend == "torch":
            value = value.detach().cpu().contiguous().numpy()
        value = np.ascontiguousarray(value)
        digest.update(name.encode() + b"\0" + str(value.dtype).encode() + b"\0")
        digest.update(json.dumps(value.shape).encode() + b"\0" + value.tobytes())
    return digest.hexdigest()


class GreedyAgentPolicy:
    """Batch-compatible adapter for deterministic existing agents."""

    def __init__(self, agent):
        self.agent = copy.deepcopy(agent)
        self.source = {"kind": type(agent).__name__}
        self.feature_dim = getattr(agent, "feature_dim", FEAT_DIM)

    def act(self, observation):
        return self.agent.act(observation) if hasattr(self.agent, "act") else self.agent.act_batch([observation])[0]

    def act_batch(self, observations):
        return self.agent.act_batch(observations) if hasattr(self.agent, "act_batch") else [self.act(observation) for observation in observations]


class BatchModelPolicy:
    """Greedy frozen snapshot, with one Torch forward per observation batch."""

    def __init__(self, model, *, backend, device="cpu", source=None):
        if backend not in ("numpy", "torch"):
            raise ValueError("model backend must be numpy or torch")
        self.backend = backend
        self.device = device
        self.source = dict(source or {"kind": "in_memory_snapshot"})
        self.model = copy.deepcopy(model)
        cfg = self.model.cfg
        self.feature_dim = int(cfg.get("feat_dim", FEAT_DIM) if isinstance(cfg, dict) else cfg.feat_dim)
        self.source["model_sha256"] = _model_identity(self.model, backend)
        self.source["backend"] = backend
        if backend == "torch":
            self.model.to(device).eval()
            for parameter in self.model.parameters():
                parameter.requires_grad_(False)

    def act(self, observation):
        return self.act_batch([observation])[0]

    def act_batch(self, observations):
        if not observations:
            return []
        encoded = [encode_decision(obs, feat_dim=self.feature_dim) for obs in observations]
        if self.backend == "numpy":
            scores = [self.model.q_values(tokens, features) for tokens, features in encoded]
        else:
            import torch

            lengths = np.asarray([len(tokens) for tokens, _ in encoded], dtype=np.int64)
            counts = [len(features) for _, features in encoded]
            tokens = np.zeros((len(encoded), int(lengths.max())), dtype=np.int64)
            features = np.zeros((len(encoded), max(counts), self.feature_dim), dtype=np.float32)
            for row, (seq, actions) in enumerate(encoded):
                tokens[row, :len(seq)] = seq
                features[row, :len(actions)] = actions
            with torch.inference_mode():
                q, _ = self.model(torch.as_tensor(tokens, device=self.device),
                                  torch.as_tensor(lengths, device=self.device),
                                  torch.as_tensor(features, device=self.device))
                q = q.cpu().numpy()
            # Padding is excluded before argmax, even if every real Q is negative.
            scores = [q[row, :count] for row, count in enumerate(counts)]
        selected = []
        for values, observation in zip(scores, observations):
            values = np.asarray(values)
            if values.shape != (len(observation["legal"]),) or not np.isfinite(values).all():
                raise ValueError("policy returned invalid or nonfinite action scores")
            selected.append(int(np.argmax(values)))
        return selected


def _file_identity(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return {"kind": "checkpoint", "path": str(Path(path).resolve()), "sha256": digest.hexdigest()}


def make_policy(spec_or_model, *, backend="auto", device="cpu"):
    """Accept a .npz/.pt path, model, existing greedy agent, or 'rule'.

    Torch can load exported NPZ weights for GPU evaluation. Models are copied
    before eval()/to(), so evaluating a live learner never changes its mode,
    device, weights, or requires_grad flags. Custom agents must be deterministic.
    """
    if backend not in ("auto", "numpy", "torch"):
        raise ValueError("backend must be auto, numpy or torch")
    if isinstance(spec_or_model, (BatchModelPolicy, GreedyAgentPolicy)):
        return spec_or_model
    if isinstance(spec_or_model, (str, Path)):
        if str(spec_or_model) == "rule":
            return GreedyAgentPolicy(RuleAgent())
        path = Path(spec_or_model)
        source = _file_identity(path)
        selected = ("numpy" if path.suffix == ".npz" else "torch") if backend == "auto" else backend
        if selected == "numpy":
            if path.suffix != ".npz":
                raise ValueError("numpy evaluation requires exported .npz weights")
            from .model_np import NumpyModel
            model = NumpyModel(path)
        else:
            from .model_torch import FableDanNet, ModelConfig, load_ckpt
            if path.suffix == ".npz":
                import torch
                from .model_np import NumpyModel
                exported = NumpyModel(path)
                # Initialization is irrelevant: every inference tensor is loaded.
                with torch.random.fork_rng(devices=[]):
                    model = FableDanNet(ModelConfig.from_dict(exported.cfg))
                missing, unexpected = model.load_state_dict(
                    {key: torch.from_numpy(value) for key, value in exported.w.items()}, strict=False)
                if unexpected or any(not key.startswith(("ntp_head.", "belief_head.")) for key in missing):
                    raise ValueError("NPZ checkpoint is missing inference tensors")
            elif path.suffix == ".pt":
                import torch
                with torch.random.fork_rng(devices=[]):
                    model, _ = load_ckpt(path, device="cpu")
            else:
                raise ValueError("expected a .npz or .pt checkpoint")
        return BatchModelPolicy(model, backend=selected, device=device, source=source)
    if hasattr(spec_or_model, "act") or hasattr(spec_or_model, "act_batch"):
        if getattr(spec_or_model, "eps", 0.0) != 0:
            raise ValueError("evaluation agents must use greedy eps=0")
        if hasattr(spec_or_model, "model"):
            return make_policy(spec_or_model.model, backend=backend, device=device)
        return GreedyAgentPolicy(spec_or_model)
    if hasattr(spec_or_model, "state_dict"):
        if backend == "numpy":
            raise ValueError("export a Torch model to NPZ for numpy evaluation")
        return BatchModelPolicy(spec_or_model, backend="torch", device=device)
    if hasattr(spec_or_model, "q_values"):
        if backend == "torch":
            raise ValueError("load the NPZ path to select the Torch backend")
        return BatchModelPolicy(spec_or_model, backend="numpy", device=device)
    raise TypeError("expected an agent, FableDan model, checkpoint path, or 'rule'")


def record_decision(metrics, observation, chosen_index):
    """Count situations, not tactical mistakes; only sees public observation."""
    legal = observation["legal"]
    move = legal[chosen_index]
    me = observation["player"]
    partner = (me + 2) % 4
    owner = observation["lead_owner"]
    following = observation["lead"] is not None and observation["lead"].type != PASS
    nonpass = [candidate for candidate in legal if candidate.type != PASS]
    ordinary = [candidate for candidate in nonpass if not candidate.is_bombish()]
    is_pass = move.type == PASS
    metrics["choice_decisions"] += 1
    metrics["chosen_passes"] += int(is_pass)
    metrics["chosen_nonpasses"] += int(not is_pass)
    metrics["chosen_cards_played"] += move.size
    if not following:
        metrics["lead_decisions"] += 1
        multi_available = any(candidate.size > 1 and not candidate.is_bombish() for candidate in legal)
        metrics["lead_with_ordinary_multicard_option"] += int(multi_available)
        metrics["lead_single_with_multicard_option"] += int(multi_available and move.type == SINGLE)
        metrics["lead_multicard_chosen"] += int(move.size > 1)
        return
    if owner == partner:
        metrics["partner_control_decisions"] += 1
        metrics["partner_control_passes"] += int(is_pass)
        metrics["partner_control_overtakes"] += int(not is_pass)
        # A partner already out can require 接风 handling, so keep it separate.
        if observation["done"][partner]:
            metrics["finished_partner_control_decisions"] += 1
            metrics["finished_partner_control_passes"] += int(is_pass)
    elif owner is not None and owner % 2 != me % 2:
        metrics["enemy_control_decisions"] += 1
        metrics["enemy_control_with_reply"] += int(bool(nonpass))
        metrics["enemy_control_pass_with_reply"] += int(is_pass and bool(nonpass))
        metrics["enemy_control_with_ordinary_reply"] += int(bool(ordinary))
        metrics["enemy_control_pass_with_ordinary_reply"] += int(is_pass and bool(ordinary))
        active_enemies = [p for p in range(4) if p % 2 != me % 2 and not observation["done"][p]]
        danger = any(observation["left"][p] <= 2 for p in active_enemies)
        if danger:
            metrics["enemy_endgame_with_reply"] += int(bool(nonpass))
            metrics["enemy_endgame_pass_with_reply"] += int(is_pass and bool(nonpass))
            metrics["enemy_endgame_with_ordinary_reply"] += int(bool(ordinary))
            metrics["enemy_endgame_pass_with_ordinary_reply"] += int(is_pass and bool(ordinary))


def _ratio(counts, numerator, denominator):
    return counts.get(numerator, 0) / counts[denominator] if counts.get(denominator, 0) else None


def _metrics_report(counts, games):
    return {"counts": dict(sorted(counts.items())), "rates": {
        "all_action_pass_rate": _ratio(counts, "all_passes", "all_actions"),
        "partner_control_pass_rate": _ratio(counts, "partner_control_passes", "partner_control_decisions"),
        "enemy_ordinary_reply_pass_rate": _ratio(counts, "enemy_control_pass_with_ordinary_reply", "enemy_control_with_ordinary_reply"),
        "enemy_endgame_ordinary_reply_pass_rate": _ratio(counts, "enemy_endgame_pass_with_ordinary_reply", "enemy_endgame_with_ordinary_reply"),
        "single_lead_with_multicard_option_rate": _ratio(counts, "lead_single_with_multicard_option", "lead_with_ordinary_multicard_option"),
    }, "mean_team_terminal_cards": counts.get("terminal_team_cards", 0) / games,
        "mean_terminal_wildcards": counts.get("terminal_wildcards", 0) / games,
        "mean_cards_per_nonpass": _ratio(counts, "all_cards_played", "all_nonpasses")}


def _paired_jobs(pairs, seed, ladder_frac):
    master = random.Random(seed)
    for pair_id in range(pairs):
        pair_seed = master.getrandbits(63)
        rng = random.Random(pair_seed)
        level, tribute = sample_setting(rng, ladder_frac)
        deck = list(range(108))
        rng.shuffle(deck)
        deal = [deck[p * 27:(p + 1) * 27] for p in range(4)]
        deal_digest = hashlib.sha256(bytes(deck)).hexdigest()
        for parity in (0, 1):
            yield {"pair_id": pair_id, "pair_seed": pair_seed, "candidate_parity": parity,
                   "level": level, "tribute_mode": tribute, "deal": deal, "deal_sha256": deal_digest}


def _mixed_jobs(deals, seed, ladder_frac):
    for job in _paired_jobs(deals, seed, ladder_frac):
        if job["candidate_parity"] == 0:
            for seat in range(4):
                yield {**job, "candidate_seat": seat, "candidate_parity": seat % 2}


def _play_jobs(candidate, opponent, jobs, batch_size, check_stop):
    rows = []
    jobs = iter(jobs)
    active = []

    def finish(game, result):
        rewards, ranking = result
        parity = game["job"]["candidate_parity"]
        candidate_seats = (parity, parity + 2)
        metrics = game["metrics"]
        rnd = game["round"]
        for event in rnd.events:
            if event[1] not in candidate_seats or event[0] not in ("pass", "play"):
                continue
            metrics["all_actions"] += 1
            metrics["all_passes"] += int(event[0] == "pass")
            metrics["all_nonpasses"] += int(event[0] == "play")
            if event[0] == "play":
                metrics["all_cards_played"] += event[2].size
        metrics["terminal_team_cards"] = sum(len(rnd.hands[p]) for p in candidate_seats)
        metrics["terminal_wildcards"] = sum(is_wildcard(card, rnd.lv) for p in candidate_seats for card in rnd.hands[p])
        # Engine's signed +/-1..3 reward equals the winner's game-level score
        # minus the loser's zero; don't sum duplicate rewards for teammates.
        row = {key: value for key, value in game["job"].items() if key != "deal"}
        row.update(team_score_difference=int(rewards[parity]), win=int(rewards[parity] > 0),
                   own_official_points=max(int(rewards[parity]), 0),
                   opponent_official_points=max(-int(rewards[parity]), 0),
                   ranking=ranking, terminal_counts=[len(hand) for hand in rnd.hands], metrics=dict(metrics))
        rows.append(row)

    exhausted = False
    while active or not exhausted:
        if check_stop is not None:
            check_stop()
        while len(active) < batch_size and not exhausted:
            job = next(jobs, None)
            if job is None:
                exhausted = True
                break
            candidate_seats = ([job["candidate_seat"]] if "candidate_seat" in job else
                               [job["candidate_parity"], job["candidate_parity"] + 2])
            feature_dims = [getattr(candidate if p in candidate_seats else opponent,
                                    "feature_dim", FEAT_DIM) for p in range(4)]
            rnd = GuandanRound(job["level"], rng=random.Random(job["pair_seed"]),
                               tribute_mode=job["tribute_mode"], deal=job["deal"],
                               feature_dim=feature_dims)
            game = {"job": job, "round": rnd, "gen": rnd.play_steps(), "metrics": Counter()}
            try:
                game["obs"] = next(game["gen"])
                active.append(game)
            except StopIteration as stop:
                finish(game, stop.value)
        for is_candidate, policy in ((True, candidate), (False, opponent)):
            def candidate_turn(game):
                if "candidate_seat" in game["job"]:
                    return game["obs"]["player"] == game["job"]["candidate_seat"]
                return game["obs"]["player"] % 2 == game["job"]["candidate_parity"]
            selected = [game for game in active if candidate_turn(game) == is_candidate]
            if not selected:
                continue
            observations = [game["obs"] for game in selected]
            indices = policy.act_batch(observations) if hasattr(policy, "act_batch") else [policy.act(obs) for obs in observations]
            if len(indices) != len(selected):
                raise ValueError("batch policy returned wrong number of decisions")
            for game, index in zip(selected, indices):
                if isinstance(index, (bool, np.bool_)) or not isinstance(index, (int, np.integer)) or not 0 <= index < len(game["obs"]["legal"]):
                    raise ValueError("policy returned an invalid action index")
                if is_candidate:
                    record_decision(game["metrics"], game["obs"], int(index))
                try:
                    game["obs"] = game["gen"].send(int(index))
                except StopIteration as stop:
                    finish(game, stop.value)
                    active.remove(game)
    return sorted(rows, key=lambda row: (row["pair_id"], row.get("candidate_seat", row["candidate_parity"])))


def paired_summary(rows, *, bootstrap_samples=2000, confidence=0.95, seed=0,
                   assignment_key="candidate_parity", assignments=(0, 1)):
    """Resample whole seat-swapped pairs, never the correlated individual games."""
    if not rows or bootstrap_samples < 100 or not 0 < confidence < 1:
        raise ValueError("nonempty paired results, >=100 bootstrap samples, and 0<confidence<1 required")
    groups = {}
    for row in rows:
        groups.setdefault(row["pair_id"], []).append(row)
    values = []
    for pair in groups.values():
        if len(pair) != len(assignments) or {row[assignment_key] for row in pair} != set(assignments):
            raise ValueError("each evaluation pair requires both seat assignments")
        if any(pair[0][key] != pair[1][key] for key in ("pair_seed", "level", "tribute_mode", "deal_sha256")):
            raise ValueError("paired games must use identical deals and settings")
        values.append([np.mean([row["team_score_difference"] for row in pair]), np.mean([row["win"] for row in pair])])
    data = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    samples = []
    for start in range(0, bootstrap_samples, 128):
        indices = rng.integers(len(data), size=(min(128, bootstrap_samples - start), len(data)))
        samples.append(data[indices].mean(axis=1))
    samples = np.concatenate(samples)
    low, high = np.quantile(samples, [(1 - confidence) / 2, (1 + confidence) / 2], axis=0)
    counters = Counter()
    for row in rows:
        counters.update(row.get("metrics", {}))
    return {"pairs": len(data), "games": len(rows), "mean_team_score_difference": float(data[:, 0].mean()),
            "mean_own_official_points": float(np.mean([max(row["team_score_difference"], 0) for row in rows])),
            "mean_opponent_official_points": float(np.mean([max(-row["team_score_difference"], 0) for row in rows])),
            "win_rate": float(data[:, 1].mean()), "team_score_difference_ci": [float(low[0]), float(high[0])],
            "win_rate_ci": [float(low[1]), float(high[1])], "confidence": confidence,
            "bootstrap_unit": "same_deal_seat_swapped_pair", "bootstrap_samples": bootstrap_samples,
            "diagnostics": _metrics_report(counters, len(rows))}


def promotion_gate(comparisons, *, min_pairs=MIN_PROMOTION_PAIRS, rule_min_score=0.0):
    """Only a recommendation; the hard 200-pair floor cannot be disabled."""
    if min_pairs < 1 or not np.isfinite(rule_min_score):
        raise ValueError("invalid promotion thresholds")
    required = max(MIN_PROMOTION_PAIRS, min_pairs)
    reasons = []
    for label in ("bc_anchor", "rule"):
        summary = comparisons.get(label, {}).get("summary")
        if summary is None:
            reasons.append(f"missing_{label}_comparison")
            continue
        if summary["pairs"] < required:
            reasons.append(f"{label}_insufficient_pairs")
        if summary.get("bootstrap_samples", 0) < 1000:
            reasons.append(f"{label}_insufficient_bootstrap_samples")
        if summary.get("confidence", 0) < 0.95:
            reasons.append(f"{label}_confidence_below_95pct")
        score_low = summary["team_score_difference_ci"][0]
        win_low = summary["win_rate_ci"][0]
        if not np.isfinite([score_low, win_low]).all():
            reasons.append(f"{label}_invalid_interval")
        elif label == "bc_anchor" and score_low <= 0:
            reasons.append("bc_anchor_no_confident_score_improvement")
        elif label == "rule" and score_low < rule_min_score:
            reasons.append("rule_score_floor_not_met")
    return {"passed": not reasons, "release_eligible": False,
            "scope": "local_strength_gate_only_requires_official_judge_and_runtime_validation",
            "required_pairs_per_opponent": required, "rule_min_score": rule_min_score,
            "reasons": reasons}


def evaluate_posttrain(candidate, anchor, *, pairs=200, seed=42, batch_size=16,
                       ladder_frac=1.0, min_pairs=200, bootstrap_samples=2000,
                       confidence=0.95, check_stop=None, backend="auto", device="cpu",
                       rule_min_score=0.0, mixed_pairs=0):
    """Run candidate-team vs frozen BC-team and rule-team on common paired deals."""
    if pairs <= 0 or batch_size <= 0 or mixed_pairs < 0 or not 0 <= ladder_frac <= 1:
        raise ValueError("pairs/batch_size must be positive and ladder_frac in [0,1]")
    if bootstrap_samples < 100 or not 0 < confidence < 1:
        raise ValueError("bootstrap_samples>=100 and 0<confidence<1 required")
    candidate = make_policy(candidate, backend=backend, device=device)
    anchor = make_policy(anchor, backend=backend, device=device)
    comparisons = {}
    for index, (label, opponent) in enumerate((("bc_anchor", anchor), ("rule", RuleAgent()))):
        rows = _play_jobs(candidate, opponent, _paired_jobs(pairs, seed, ladder_frac), batch_size, check_stop)
        comparisons[label] = {"summary": paired_summary(rows, bootstrap_samples=bootstrap_samples,
                                                        confidence=confidence, seed=seed + index + 1),
                              "games": rows}
    mixed = {"measured": False, "promotion_gate_input": False}
    if mixed_pairs:
        rows = _play_jobs(candidate, anchor, _mixed_jobs(mixed_pairs, seed, ladder_frac), batch_size, check_stop)
        summary = paired_summary(rows, bootstrap_samples=bootstrap_samples, confidence=confidence,
                                 seed=seed + 3, assignment_key="candidate_seat", assignments=(0, 1, 2, 3))
        summary["deal_blocks"] = summary.pop("pairs")
        summary["bootstrap_unit"] = "same_deal_four_candidate_seats"
        mixed.update(measured=True, summary=summary, games=rows,
                     composition="one candidate with BC anchor partner vs two BC anchor opponents",
                     choice_metrics_subject="candidate_only", all_action_metrics_subject="candidate_and_anchor_partner")
    return {"schema": "fabledan-posttrain-eval-v1", "seed": seed, "pairs_per_opponent": pairs,
            "total_games": 4 * (pairs + mixed_pairs), "ladder_frac": ladder_frac, "batch_size": batch_size,
            "candidate": getattr(candidate, "source", {"kind": type(candidate).__name__}),
            "bc_anchor": getattr(anchor, "source", {"kind": type(anchor).__name__}),
            "comparisons": comparisons, "mixed_partner_diagnostic": mixed,
            "reward_contract": {"primary": "team_score_difference", "range": [-3, 3],
                                "formula": "own_official_points - opponent_official_points",
                                "official_points": "winner receives 1/2/3, loser receives 0; count once per team"},
            "promotion_gate": promotion_gate(comparisons, min_pairs=min_pairs, rule_min_score=rule_min_score),
            "metric_notes": ["Score difference is signed team reward in [-3,3], counted once per team.",
                             "Win and score intervals resample complete deal pairs, not individual games.",
                             "Pass/team/endgame counters describe opportunities; none labels a move wrong.",
                             "Choice counters exclude engine-forced single-option actions; all-action counters include them.",
                             "Main comparisons use two candidate teammates; optional mixed-partner diagnostics never enter promotion gate.",
                             "Local engine evaluation does not establish BotZone legality, latency, or overall ladder win rate."]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--anchor", required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--pairs", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--backend", choices=("auto", "numpy", "torch"), default="auto")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--ladder-frac", type=float, default=1.0)
    parser.add_argument("--min-pairs", type=int, default=200)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--mixed-pairs", type=int, default=0,
                        help="optional four-seat deal blocks with an anchor teammate; diagnostic only")
    args = parser.parse_args()
    report = evaluate_posttrain(args.candidate, args.anchor, pairs=args.pairs, seed=args.seed,
                               batch_size=args.batch_size, backend=args.backend, device=args.device,
                               ladder_frac=args.ladder_frac, min_pairs=args.min_pairs,
                               bootstrap_samples=args.bootstrap_samples, mixed_pairs=args.mixed_pairs)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(args.report), "promotion_gate": report["promotion_gate"]}))


if __name__ == "__main__":
    main()
