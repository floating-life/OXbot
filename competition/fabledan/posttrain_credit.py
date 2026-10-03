"""Offline action credit from paired continuations, followed by Q/rank training.

The simulator knows the deal; every policy sees only its own normal observation.
Targets estimate this deal under specified continuation policies. They are not
proof of optimal play, nor labels that every pass or teammate overtake is wrong.
The deployment architecture, vocabulary and feature dimensions stay unchanged.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import random

import numpy as np

from .agents import RuleAgent
from .combos import PASS
from .encode import encode_decision
from .engine import GuandanRound
from .ring import sample_setting

SCHEMA = "oxbot-fabledan-counterfactual-v1"


def file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def encoder_identity():
    root = Path(__file__).resolve().parent
    return {name: file_sha(root / name) for name in
            ("cards.py", "combos.py", "encode.py", "engine.py")}


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8")
    temp.replace(path)


def category(obs):
    owner, me = obs["lead_owner"], obs["player"]
    if owner is None:
        return "lead_composition"
    if owner == (me + 2) % 4:
        return "partner_control"
    if owner % 2 != me % 2 and 0 < obs["left"][owner] <= 2:
        return "opponent_endgame"
    return "other_follow"


def select_alternatives(obs, chosen, limit, rng):
    """Include the played action, pass and different combination types.

    Equal feature vectors are indistinguishable to both deployed networks.
    Keep one physical realization per vector, including the actual choice;
    never teach contradictory preferences between those realizations.
    """
    if limit < 2:
        raise ValueError("candidate limit must be at least two")
    _, feats = encode_decision(obs, feat_dim=obs.get("feature_dim", 80))
    legal = obs["legal"]
    if not 0 <= chosen < len(legal):
        raise ValueError("chosen action is outside the legal set")
    priority = [chosen]
    priority += [i for i, move in enumerate(legal) if move.type == PASS]
    priority += [RuleAgent().act(obs)]
    # Cover sizes/types rather than restricting exploration to BC's top-k.
    families = {}
    for i, move in enumerate(legal):
        families.setdefault((move.type, move.size), []).append(i)
    keys = sorted(families, key=lambda key: (-key[1], key[0]))
    priority += [min(families[key], key=lambda i: legal[i].key) for key in keys]
    tail = list(range(len(legal)))
    rng.shuffle(tail)
    priority += tail
    seen, result = set(), []
    for i in priority:
        key = feats[i].tobytes()
        if key not in seen:
            seen.add(key)
            result.append(i)
        if len(result) == limit:
            break
    return result


def act_batch(policy, observations):
    if hasattr(policy, "act_batch"):
        actions = policy.act_batch(observations)
    else:
        actions = [policy.act(obs) for obs in observations]
    if len(actions) != len(observations):
        raise ValueError("policy returned an incomplete action batch")
    for index, obs in zip(actions, observations):
        if (isinstance(index, (bool, np.bool_)) or not isinstance(index, (int, np.integer))
                or not 0 <= index < len(obs["legal"])):
            raise ValueError("policy returned an illegal action index")
    return [int(index) for index in actions]


def review_position(rnd, obs, chosen, scenarios, *, max_candidates=8, seed=0,
                    max_decisions=1024):
    """Evaluate identical alternatives against each fixed four-seat policy set."""
    if not scenarios or max_decisions < 1:
        raise ValueError("need at least one continuation scenario and positive decision budget")
    indices = select_alternatives(obs, chosen, max_candidates, random.Random(seed))
    feature_dim = obs.get("feature_dim", 80)
    tokens, features = encode_decision(obs, feat_dim=feature_dim)
    branches = []
    outcomes = np.zeros((len(indices), len(scenarios)), dtype=np.float32)
    for j, (name, policies) in enumerate(scenarios):
        if len(policies) != 4:
            raise ValueError("each continuation scenario needs four policies")
        for k, index in enumerate(indices):
            clone, gen = rnd.fork_decision(obs)
            clone.feature_dims = [getattr(policy, "feature_dim", 80) for policy in policies]
            # The forced first action belongs to the sampled source action
            # set, while subsequent seats use their own checkpoint contract.
            clone.feature_dims[obs["player"]] = feature_dim
            initial = next(gen)
            def signature(move):
                return move.type, move.key, tuple(move.cards), tuple(move.claim_ranks)
            if list(map(signature, initial["legal"])) != list(map(signature, obs["legal"])):
                raise ValueError("fork changed the legal action set")
            try:
                next_obs = gen.send(index)
                branches.append({"gen": gen, "obs": next_obs, "policies": policies,
                                 "candidate": k, "scenario": j})
            except StopIteration as end:
                outcomes[k, j] = end.value[0][obs["player"]] / 3.0
    turns = 0
    while branches:
        if turns >= max_decisions:
            raise RuntimeError("counterfactual did not reach a terminal reward")
        turns += 1
        groups = {}
        for branch in branches:
            policy = branch["policies"][branch["obs"]["player"]]
            groups.setdefault(id(policy), (policy, []))[1].append(branch)
        pending = []
        for policy, group in groups.values():
            actions = act_batch(policy, [b["obs"] for b in group])
            for branch, action in zip(group, actions):
                try:
                    branch["obs"] = branch["gen"].send(action)
                    pending.append(branch)
                except StopIteration as end:
                    outcomes[branch["candidate"], branch["scenario"]] = (
                        end.value[0][obs["player"]] / 3.0)
        branches = pending
    targets = outcomes.mean(axis=1)
    best = int(np.argmax(targets))
    moves = obs["legal"]
    return {"category": category(obs), "player": obs["player"], "feature_dim": feature_dim,
            "remaining_counts": list(obs["left"]), "chosen": 0,
            "tokens": list(tokens), "features": features[indices].tolist(),
            "actions": [{"cards": list(moves[i].cards), "type": moves[i].type,
                         "size": moves[i].size, "key": moves[i].key,
                         "claim_ranks": list(moves[i].claim_ranks)} for i in indices],
            "scenario_names": [name for name, _ in scenarios],
            "returns": outcomes.tolist(), "targets": targets.tolist(),
            "best_reviewed": best,
            "estimated_regret": float(targets[best] - targets[0]),
            "candidate_coverage": len(indices), "legal_candidates": len(moves)}


def collect_reviews(policy, anchor, *, games=12, positions_per_game=4,
                    max_candidates=8, seed=41000, ladder_frac=0.8, check_stop=None):
    """Stratified replay: combination leads, partner control, enemy endgames.

    Split by whole source game before building labels. A state and all of its
    counterfactual branches always belong to the same split.
    """
    if games < 2 or positions_per_game < 1 or max_candidates < 2 or seed < 0:
        raise ValueError("need >=2 games, positive positions, >=2 candidates and a nonnegative seed")
    if not 0 <= ladder_frac <= 1:
        raise ValueError("ladder_frac must be in [0,1]")
    rule, rows = RuleAgent(), []
    categories = ("opponent_endgame", "partner_control", "lead_composition", "other_follow")
    for game in range(games):
        if check_stop:
            check_stop()
        game_seed = seed + game
        rng, reservoir_rng = random.Random(game_seed), random.Random(game_seed ^ 0xBC02)
        lv, tribute = sample_setting(rng, ladder_frac)
        learner_team = game % 2
        opponent = (policy, anchor, rule)[game % 3]
        seats = [policy if p % 2 == learner_team else opponent for p in range(4)]
        rnd = GuandanRound(lv, rng, tribute,
                           feature_dim=[getattr(agent, "feature_dim", 80) for agent in seats])
        pools, counts = {key: [] for key in categories}, {key: 0 for key in categories}
        gen = rnd.play_steps()
        try:
            obs = next(gen)
            decisions = 0
            while True:
                if decisions >= 1024:
                    raise RuntimeError("source game did not terminate")
                decisions += 1
                chosen = act_batch(seats[obs["player"]], [obs])[0]
                if obs["player"] % 2 == learner_team:
                    key = category(obs)
                    counts[key] += 1
                    pool = pools[key]
                    replace = (len(pool) if len(pool) < positions_per_game
                               else reservoir_rng.randrange(counts[key]))
                    if replace < positions_per_game:
                        snapshot = copy.deepcopy((rnd, obs, chosen, decisions))
                        if replace == len(pool):
                            pool.append(snapshot)
                        else:
                            pool[replace] = snapshot
                obs = gen.send(chosen)
        except StopIteration:
            pass
        selected = []
        for offset in range(positions_per_game):
            for key in categories:
                if offset < len(pools[key]):
                    selected.append(pools[key][offset])
                if len(selected) >= positions_per_game:
                    break
            if len(selected) >= positions_per_game:
                break
        for source, observation, chosen, decision in selected:
            if check_stop:
                check_stop()
            team = observation["player"] % 2
            scenarios = [
                ("self", [policy] * 4),
                ("anchor_opponents", [policy if p % 2 == team else anchor for p in range(4)]),
                ("rule_opponents", [policy if p % 2 == team else rule for p in range(4)]),
                ("anchor_partner", [anchor if p == (observation["player"] + 2) % 4
                                    else policy for p in range(4)]),
            ]
            row = review_position(source, observation, chosen, scenarios,
                                  max_candidates=max_candidates, seed=game_seed + decision)
            row.update({"game_seed": game_seed, "decision": decision,
                        "split": "validation" if game % 4 == 0 else "train"})
            rows.append(row)
        print("[review] game %d/%d, positions %d" % (game + 1, games, len(rows)), flush=True)
    return {"schema": SCHEMA, "rows": rows,
            "feature_dim": getattr(policy, "feature_dim", 80),
            "encoder_sha256": encoder_identity(),
            "settings": {"games": games, "positions_per_game": positions_per_game,
                         "max_candidates": max_candidates, "seed": seed,
                         "ladder_frac": ladder_frac},
            "reward": "signed terminal team score / 3; teammates share reward",
            "target_scope": "known simulated deal, four fixed continuation scenarios; no optimality claim",
            "feature_scope": "own hand and public history only; no private allocations stored",
            "release_eligible": False}


def preference_pairs(returns, min_gap=1 / 3):
    """Use preferences only when no reviewed continuation reverses the sign."""
    values = np.asarray(returns, dtype=np.float32)
    if values.ndim != 2 or not values.shape[1] or not np.isfinite(values).all():
        raise ValueError("returns must be a finite [candidates, scenarios] matrix")
    delta = values[:, None, :] - values[None, :, :]
    better, worse = np.where((delta.mean(axis=2) >= min_gap - 1e-6)
                             & (delta.min(axis=2) >= -1e-6)
                             & (delta.max(axis=2) > 1e-6))
    return better, worse


def validate_reviews(data):
    if data.get("schema") != SCHEMA:
        raise ValueError("unsupported review schema")
    if "encoder_sha256" in data and data["encoder_sha256"] != encoder_identity():
        raise ValueError("review rules or encoder changed; regenerate the targets")
    feature_dim = data.get("feature_dim", 80)
    if feature_dim not in (80, 224):
        raise ValueError("unsupported review feature dimension")
    seen, split_games = set(), {"train": set(), "validation": set()}
    for row in data["rows"]:
        split = row["split"]
        if split not in split_games:
            raise ValueError("only train and validation reviews are accepted")
        identity = (row["game_seed"], row["decision"])
        if identity in seen:
            raise ValueError("duplicate source decision")
        seen.add(identity)
        split_games[split].add(row["game_seed"])
        f = np.asarray(row["features"], dtype=np.float32)
        r = np.asarray(row["returns"], dtype=np.float32)
        t = np.asarray(row["tokens"])
        if (row.get("feature_dim", 80) != feature_dim
                or f.ndim != 2 or f.shape[1] != feature_dim or len(f) < 1
                or r.shape != (len(f), len(row["scenario_names"]))
                or r.shape[1] < 1 or not np.isfinite(f).all()
                or not np.isfinite(r).all() or np.abs(r).max() > 1.00001
                or t.ndim != 1 or not 1 <= len(t) <= 512
                or not np.issubdtype(t.dtype, np.integer) or t.min() < 0 or t.max() >= 48
                or not 0 <= row["chosen"] < len(f)):
            raise ValueError("malformed public features or terminal rewards")
        if len({feature.tobytes() for feature in f}) != len(f):
            raise ValueError("duplicate feature rows cannot receive conflicting action credit")
    if not all(split_games.values()) or split_games["train"] & split_games["validation"]:
        raise ValueError("train/validation must contain disjoint complete source games")


def fit_reviews(checkpoint, data_path, out, *, epochs=4, lr=1e-5, device="cpu",
                q_scale=1.0, seed=42000, rank_weight=0.5, anchor_weight=0.02):
    """Fit team return plus robust pairwise action credit, retaining an anchor.

    Validation chooses the lowest combined loss, not a claimed strongest bot.
    posttrain_eval is mandatory before promoting any resulting checkpoint.
    """
    import torch
    import torch.nn.functional as functional
    from .model_torch import load_ckpt
    from .train_fast import atomic_checkpoint, atomic_export

    if epochs < 1 or any(not np.isfinite(v) or v <= 0 for v in (lr, q_scale)):
        raise ValueError("epochs, learning rate and Q scale must be positive")
    if any(not np.isfinite(v) or v < 0 for v in (rank_weight, anchor_weight)):
        raise ValueError("loss weights must be finite and nonnegative")
    out, checkpoint, data_path = Path(out), Path(checkpoint), Path(data_path)
    if out.resolve() == checkpoint.resolve().parent:
        raise ValueError("write credit-trained candidates to a new directory")
    if any((out / name).exists() for name in ("best.pt", "latest.pt", "training.json")):
        raise ValueError("credit output already contains a run; choose a new directory")
    data = json.loads(data_path.read_text(encoding="utf-8"))
    validate_reviews(data)
    torch.manual_seed(seed)
    torch.set_num_threads(2)
    model, _ = load_ckpt(str(checkpoint), device=device)
    if model.cfg.feat_dim != data.get("feature_dim", 80):
        raise ValueError("review features do not match the checkpoint architecture")
    final = [layer for layer in model.q_head.modules() if isinstance(layer, torch.nn.Linear)][-1]
    with torch.no_grad():
        final.weight.mul_(q_scale)
        final.bias.mul_(q_scale)
    anchor = copy.deepcopy(model).eval()
    for parameter in anchor.parameters():
        parameter.requires_grad_(False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    train = [r for r in data["rows"] if r["split"] == "train"]
    valid = [r for r in data["rows"] if r["split"] == "validation"]
    rng, history, best = random.Random(seed), [], float("inf")
    out.mkdir(parents=True, exist_ok=True)
    provenance = {"source_sha256": file_sha(checkpoint), "reviews_sha256": file_sha(data_path),
                  "feature_dim": model.cfg.feat_dim, "feature_version": model.cfg.feature_version,
                  "training_kind": "posttrain_credit", "q_scale": q_scale,
                  "objective": "terminal team-return Huber + robust pairwise preference + centered anchor",
                  "release_eligible": False}

    def loss_for(row):
        tokens = torch.tensor([row["tokens"]], dtype=torch.long, device=device)
        length = torch.tensor([len(row["tokens"])], device=device)
        features = torch.tensor([row["features"]], dtype=torch.float32, device=device)
        target = torch.tensor(np.mean(row["returns"], axis=1), dtype=torch.float32, device=device)
        q, _ = model(tokens, length, features)
        q = q[0].float()
        loss = functional.smooth_l1_loss(q, target)
        better, worse = preference_pairs(row["returns"])
        if len(better):
            loss = loss + rank_weight * functional.softplus(
                -(q[torch.as_tensor(better, device=device)]
                  - q[torch.as_tensor(worse, device=device)])).mean()
        with torch.no_grad():
            old, _ = anchor(tokens, length, features)
            centered = old[0].float() - old[0].float().mean()
        return loss + anchor_weight * functional.mse_loss(q - q.mean(), centered)

    for epoch in range(1, epochs + 1):
        rng.shuffle(train)
        model.train()
        train_loss = 0.0
        for row in train:
            optimizer.zero_grad(set_to_none=True)
            loss = loss_for(row)
            if not torch.isfinite(loss):
                raise RuntimeError("nonfinite post-training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += float(loss.detach())
        model.eval()
        with torch.no_grad():
            val_loss = float(np.mean([float(loss_for(row)) for row in valid]))
        item = {"epoch": epoch, "train_loss": train_loss / len(train),
                "validation_loss": val_loss, "selected_best": val_loss < best}
        history.append(item)
        meta = dict(provenance, epoch=epoch, seed=seed)
        atomic_checkpoint(model, optimizer, meta, str(out / "latest.pt"))
        atomic_export(model, str(out / "latest.npz"))
        if val_loss < best:
            best = val_loss
            atomic_checkpoint(model, optimizer, meta, str(out / "best.pt"))
            atomic_export(model, str(out / "best.npz"))
        print("[credit] epoch %d train %.4f validation %.4f" %
              (epoch, item["train_loss"], val_loss), flush=True)
    report = dict(provenance, epochs=history, train_positions=len(train),
                  validation_positions=len(valid),
                  preference_pairs=sum(len(preference_pairs(r["returns"])[0]) for r in train),
                  candidate_sha256=file_sha(out / "best.npz"))
    write_json(out / "training.json", report)
    return report


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="command", required=True)
    collect = sub.add_parser("collect")
    collect.add_argument("--candidate", required=True)
    collect.add_argument("--anchor", required=True)
    collect.add_argument("--out", required=True)
    collect.add_argument("--games", type=int, default=12)
    collect.add_argument("--positions", type=int, default=4)
    collect.add_argument("--candidates", type=int, default=8)
    collect.add_argument("--seed", type=int, default=41000)
    collect.add_argument("--ladder-frac", type=float, default=0.8)
    collect.add_argument("--device", default="cpu")
    fit = sub.add_parser("fit")
    fit.add_argument("--checkpoint", required=True)
    fit.add_argument("--reviews", required=True)
    fit.add_argument("--out", required=True)
    fit.add_argument("--epochs", type=int, default=4)
    fit.add_argument("--lr", type=float, default=1e-5)
    fit.add_argument("--q-scale", type=float, default=1.0)
    fit.add_argument("--device", default="cpu")
    fit.add_argument("--seed", type=int, default=42000)
    args = ap.parse_args()
    if args.command == "collect":
        from .posttrain_eval import make_policy
        policy = make_policy(args.candidate, device=args.device)
        anchor = make_policy(args.anchor, device=args.device)
        data = collect_reviews(policy, anchor, games=args.games, positions_per_game=args.positions,
                               max_candidates=args.candidates, seed=args.seed,
                               ladder_frac=args.ladder_frac)
        data["provenance"] = {"candidate_sha256": file_sha(args.candidate),
                              "anchor_sha256": file_sha(args.anchor)}
        write_json(args.out, data)
    else:
        fit_reviews(args.checkpoint, args.reviews, args.out, epochs=args.epochs,
                    lr=args.lr, device=args.device, q_scale=args.q_scale, seed=args.seed)


if __name__ == "__main__":
    main()
