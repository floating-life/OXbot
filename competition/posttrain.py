"""Finite BC -> team-return DMC -> counterfactual credit -> strength evaluation.

Run from competition/. --smoke exercises real updates, never qualifies a release.
Existing deployment files and the BC checkpoint directory are never overwritten.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

from fabledan.posttrain_credit import (collect_reviews, file_sha, fit_reviews,
                                       write_json)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--warm-start", type=Path, default=Path("ckpts/real-v2/best.pt"))
    ap.add_argument("--out", type=Path, default=Path("ckpts/posttrain-v1"))
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--feature-dim", type=int, choices=(80, 224), default=224,
                    help="224 upgrades BC to suit and afterstate features; 80 is legacy")
    ap.add_argument("--seed", type=int, default=52000)
    ap.add_argument("--cycles", type=int, default=10)
    ap.add_argument("--actors", type=int, default=6)
    ap.add_argument("--ring", type=int, default=16)
    ap.add_argument("--max-hours", type=float, default=1.0,
                    help="DMC phase deadline; review/evaluation have finite game counts")
    ap.add_argument("--review-games", type=int, default=16)
    ap.add_argument("--positions", type=int, default=4)
    ap.add_argument("--candidates", type=int, default=8)
    ap.add_argument("--credit-epochs", type=int, default=4)
    ap.add_argument("--eval-pairs", type=int, default=200)
    ap.add_argument("--mixed-pairs", type=int, default=50)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if any(getattr(args, key) <= 0 for key in
           ("cycles", "actors", "ring", "positions", "credit_epochs", "eval_pairs")):
        ap.error("cycle, actor, ring, position, epoch and evaluation counts must be positive")
    if args.review_games < 2 or args.candidates < 2 or args.seed < 0 or args.mixed_pairs < 0:
        ap.error("need >=2 review games/candidates, nonnegative seed and mixed pairs")
    source, out = args.warm_start.resolve(), args.out.resolve()
    if not source.is_file():
        ap.error("warm-start checkpoint does not exist")
    if out == source.parent or out in source.parents:
        ap.error("output must be separate from the source checkpoint directory")
    if out.exists() and any(out.iterdir()):
        ap.error("output is not empty; choose a new run directory (DMC resume is available separately)")
    if args.smoke:
        args.cycles, args.actors, args.ring = 1, 2, 4
        args.review_games, args.positions, args.candidates = 4, 2, 3
        args.credit_epochs, args.eval_pairs, args.mixed_pairs = 1, 2, 1
    out.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    report = {"schema": "oxbot-posttrain-pipeline-v1", "status": "running",
              "stage": "dmc", "settings": vars(args).copy(),
              "source_sha256": file_sha(source), "release_eligible": False,
              "training_objective": "signed terminal team score difference / 3",
              "smoke_only": args.smoke}
    report["settings"] = {k: str(v) if isinstance(v, Path) else v
                          for k, v in report["settings"].items()}

    def record(stage):
        report["stage"] = stage
        report["elapsed_seconds"] = time.monotonic() - started
        write_json(out / "pipeline.json", report)

    dmc = out / "dmc"
    command = [sys.executable, "-m", "fabledan.train_fast", "--warm-start", str(source),
               "--out", str(dmc), "--cycles", str(args.cycles),
               "--actors", str(args.actors), "--ring", str(args.ring),
               "--device", args.device, "--infer-device", args.device,
               "--feature-dim", str(args.feature_dim),
               "--seed", str(args.seed), "--lr", "0.00002", "--eps", "0.15",
               "--top-k", "1", "--ladder-frac", "0.8", "--belief-weight", "0",
               "--buffer", "128" if args.smoke else "32768", "--diversity", "2",
               "--steps-per-cycle", "2" if args.smoke else "32",
               "--batch", "32" if args.smoke else "512",
               "--micro-batch", "16" if args.smoke else "128",
               "--eval-cycles", str(args.cycles + 1), "--snapshot-cycles", "0",
               "--ckpt-cycles", "1", "--export-cycles", "1",
               "--max-hours", str(args.max_hours)]
    report["dmc_command"] = command
    record("dmc")
    try:
        subprocess.run(command, check=True)
        import torch
        from fabledan.posttrain_eval import evaluate_posttrain, make_policy
        torch.set_num_threads(2)
        checkpoint = torch.load(dmc / "latest.pt", map_location="cpu", weights_only=False)
        if checkpoint["meta"].get("optimizer_steps", 0) < 1:
            raise RuntimeError("DMC stopped before any update; candidate is not post-trained")
        report["dmc_completed_cycles"] = checkpoint["meta"]["cycle"]
        report["dmc_optimizer_steps"] = checkpoint["meta"]["optimizer_steps"]
        del checkpoint
        record("counterfactual_review")
        policy = make_policy(dmc / "latest.pt", device=args.device)
        anchor = make_policy(source, device=args.device)
        data = collect_reviews(policy, anchor, games=args.review_games,
                               positions_per_game=args.positions,
                               max_candidates=args.candidates, seed=args.seed + 10000)
        data["provenance"] = {"candidate_sha256": file_sha(dmc / "latest.pt"),
                              "anchor_sha256": file_sha(source)}
        review_path = out / "reviews.json"
        write_json(review_path, data)
        report["review_positions"] = len(data["rows"])
        del policy, anchor, data
        record("credit_training")
        training = fit_reviews(dmc / "latest.pt", review_path, out / "credit",
                               epochs=args.credit_epochs, device=args.device, seed=args.seed + 20000)
        report["preference_pairs"] = training["preference_pairs"]
        record("held_out_evaluation")
        evaluation = evaluate_posttrain(out / "credit" / "best.pt", source,
                                         pairs=args.eval_pairs, seed=args.seed + 30000,
                                         mixed_pairs=args.mixed_pairs, backend="torch",
                                         device=args.device)
        write_json(out / "evaluation.json", evaluation)
        report["promotion_gate"] = evaluation["promotion_gate"]
        report["candidate"] = str(out / "credit" / "best.npz")
        report["candidate_sha256"] = file_sha(report["candidate"])
        if file_sha(source) != report["source_sha256"]:
            raise RuntimeError("source checkpoint changed during the run")
        report["status"] = "complete"
        record("complete")
        print(json.dumps({"report": str(out / "pipeline.json"),
                          "promotion_gate": report["promotion_gate"],
                          "release_eligible": False}, ensure_ascii=False), flush=True)
    except BaseException as exc:
        report["status"] = "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
        report["error"] = f"{type(exc).__name__}: {exc}"
        record(report["stage"])
        raise


if __name__ == "__main__":
    main()
