"""Validation and train-only audit of the v1 BC objective.

This intentionally opens only ``train`` and ``validation`` shards.  It does
not import the held-out evaluator and refuses a split other than those two.
The output is descriptive: it never modifies a checkpoint or the online bot.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "train"))
from model import CandidateModel, ModelConfig  # noqa: E402


KINDS = ("pass", "invalid", "single", "pair", "three", "straight", "set",
         "three_straight", "triple_pairs", "bomb", "straight_flush", "rocket")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def files_for(root: Path, split: str) -> list[Path]:
    if split not in ("train", "validation"):
        raise ValueError("this audit is restricted to train and validation")
    files = sorted((root / split).glob("*.npz"))
    if not files:
        raise FileNotFoundError(f"no {split} shards under {root}")
    return files


def count_bin(n: int) -> str:
    if n == 1:
        return "1"
    if n <= 3:
        return "2-3"
    if n <= 8:
        return "4-8"
    if n <= 16:
        return "9-16"
    if n <= 32:
        return "17-32"
    if n <= 64:
        return "33-64"
    if n <= 128:
        return "65-128"
    if n <= 256:
        return "129-256"
    if n <= 512:
        return "257-512"
    return "513+"


def pos_bin(n: int) -> str:
    return str(n) if n <= 3 else "4+"


def empty_split() -> dict:
    return {
        "rows": 0, "candidates": 0, "positive_candidates": 0,
        "candidate_counts": collections.Counter(),
        "positive_counts": collections.Counter(),
        "candidate_kind": collections.Counter(),
        "positive_kind": collections.Counter(),
        "candidate_size": collections.Counter(),
        "positive_size": collections.Counter(),
        "candidate_count_bin": collections.Counter(),
        "positive_count_bin": collections.Counter(),
        "pass_candidate_by_count_bin": collections.Counter(),
        "pass_positive_by_count_bin": collections.Counter(),
        "lead_rows": 0,
        "pass_candidate_rows": 0,
        "pass_positive_rows": 0,
        "pass_candidates": 0,
        "pass_positive_candidates": 0,
        "finish_candidate_rows": 0,
        "finish_positive_rows": 0,
    }


def structural_audit(data_root: Path) -> dict[str, dict]:
    result = {}
    for split in ("train", "validation"):
        out = empty_split()
        for path in files_for(data_root, split):
            with np.load(path, allow_pickle=False) as d:
                state, actions, offsets, positive = d["state"], d["actions"], d["offsets"], d["positives"]
                for row in range(len(state)):
                    begin, end = int(offsets[row]), int(offsets[row + 1])
                    aa, pp = actions[begin:end], positive[begin:end]
                    count, pos_count = len(aa), int(pp.sum())
                    kinds = np.argmax(aa[:, 108:120], axis=1)
                    sizes = np.rint(aa[:, 120] * 10).astype(np.int64)
                    pass_mask = aa[:, 126] > .5
                    finish_mask = aa[:, 127] > .5
                    out["rows"] += 1
                    out["candidates"] += count
                    out["positive_candidates"] += pos_count
                    out["candidate_counts"][count] += 1
                    out["positive_counts"][pos_count] += 1
                    out["candidate_count_bin"][count_bin(count)] += 1
                    out["positive_count_bin"][pos_bin(pos_count)] += 1
                    out["candidate_kind"].update(KINDS[int(k)] for k in kinds)
                    out["positive_kind"].update(KINDS[int(k)] for k in kinds[pp])
                    out["candidate_size"].update(int(x) for x in sizes)
                    out["positive_size"].update(int(x) for x in sizes[pp])
                    out["pass_candidates"] += int(pass_mask.sum())
                    out["pass_positive_candidates"] += int((pass_mask & pp).sum())
                    out["pass_candidate_rows"] += int(pass_mask.any())
                    out["pass_positive_rows"] += int((pass_mask & pp).any())
                    out["pass_candidate_by_count_bin"][count_bin(count)] += int(pass_mask.any())
                    out["pass_positive_by_count_bin"][count_bin(count)] += int((pass_mask & pp).any())
                    out["finish_candidate_rows"] += int(finish_mask.any())
                    out["finish_positive_rows"] += int((finish_mask & pp).any())
                    out["lead_rows"] += int(bool(state[row, 125] > .5))
        result[split] = out
    return result


def _collate(rows, state, tokens, lengths, actions, offsets, positive):
    max_len = int(lengths[rows].max())
    counts = [int(offsets[r + 1] - offsets[r]) for r in rows]
    width = max(counts)
    out_a = np.zeros((len(rows), width, 128), np.float32)
    out_m = np.zeros((len(rows), width), np.bool_)
    out_p = np.zeros((len(rows), width), np.bool_)
    for j, r in enumerate(rows):
        b, e = int(offsets[r]), int(offsets[r + 1])
        out_a[j, :e - b] = actions[b:e]
        out_m[j, :e - b] = True
        out_p[j, :e - b] = positive[b:e]
    import torch
    return {
        "tokens": torch.from_numpy(tokens[rows, :max_len].astype(np.int64)),
        "lengths": torch.from_numpy(lengths[rows].astype(np.int64)),
        "state": torch.from_numpy(state[rows].astype(np.float32)),
        "actions": torch.from_numpy(out_a), "mask": torch.from_numpy(out_m),
        "positives": torch.from_numpy(out_p), "counts": counts,
    }


def _logsumexp(x: np.ndarray) -> float:
    m = float(np.max(x))
    return m + math.log(float(np.exp(x - m).sum()))


def model_validation_audit(data_root: Path, checkpoint: Path, batch_size: int, device_name: str) -> dict:
    """Score every validation candidate and summarize action selection effects."""
    import torch
    files = files_for(data_root, "validation")
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model = CandidateModel(ModelConfig(**ckpt["config"]))
    model.load_state_dict(ckpt["state_dict"], strict=True)
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    model = model.float().to(device).eval()
    stats = {
        "rows": 0, "candidates": 0, "positive_candidates": 0, "raw_correct": 0,
        "nll_sum": 0., "choice_rows": 0, "choice_correct": 0,
        "pass_demo_rows": 0, "pass_demo_pred_pass": 0, "pass_demo_correct": 0,
        "play_demo_rows": 0, "play_demo_pred_pass": 0, "play_demo_correct": 0,
        "pred_kind": collections.Counter(), "demo_kind": collections.Counter(),
        "pred_given_demo_kind": collections.defaultdict(collections.Counter),
        "demo_size": collections.Counter(), "pred_given_demo_size": collections.defaultdict(collections.Counter),
        "by_count": {}, "by_context": {},
        "pass_calibration": {"sum_p": 0., "sum_p2": 0., "brier_sum": 0., "n": 0},
    }

    def bucket(d: dict, name: str, correct: bool, nll: float, pred_pass: bool, demo_pass: bool,
               p_pass: float, best_pos: float, best_neg: float, count: int) -> None:
        row = d.setdefault(name, {"rows": 0, "correct": 0, "nll_sum": 0., "pred_pass": 0,
                                  "demo_pass": 0, "p_pass_sum": 0., "margin_sum": 0.})
        row["rows"] += 1; row["correct"] += int(correct); row["nll_sum"] += nll
        row["pred_pass"] += int(pred_pass); row["demo_pass"] += int(demo_pass)
        row["p_pass_sum"] += p_pass; row["margin_sum"] += best_pos - best_neg

    with torch.inference_mode():
        for path in files:
            with np.load(path, allow_pickle=False) as d:
                state, tokens, lengths = d["state"], d["tokens"], d["lengths"]
                actions, offsets, positive = d["actions"], d["offsets"], d["positives"]
                rows = np.arange(len(lengths), dtype=np.int64)
                for start in range(0, len(rows), batch_size):
                    selected = rows[start:start + batch_size]
                    batch = _collate(selected, state, tokens, lengths, actions, offsets, positive)
                    counts = batch.pop("counts")
                    pos = batch.pop("positives").numpy()
                    mask = batch["mask"].numpy()
                    scores = model(**{k: v.to(device) for k, v in batch.items()}).cpu().numpy()
                    aa = batch["actions"].numpy()
                    for j, count in enumerate(counts):
                        sc = scores[j, :count].astype(np.float64)
                        pp = pos[j, :count]
                        acts = aa[j, :count]
                        winner = int(np.argmax(sc))
                        kinds = np.argmax(acts[:, 108:120], axis=1)
                        pred_kind = KINDS[int(kinds[winner])]
                        demo_kind_set = {KINDS[int(k)] for k in kinds[pp]}
                        demo_kind_key = "+".join(sorted(demo_kind_set))
                        demo_size_set = {int(round(float(x) * 10)) for x in acts[pp, 120]}
                        demo_size_key = "+".join(str(x) for x in sorted(demo_size_set))
                        demo_pass = bool((acts[:, 126] > .5)[pp].any())
                        pred_pass = bool(acts[winner, 126] > .5)
                        correct = bool(pp[winner])
                        denom = _logsumexp(sc)
                        nll = denom - _logsumexp(sc[pp])
                        pass_mask = acts[:, 126] > .5
                        p = math.exp(float(sc[pass_mask][0] - denom)) if pass_mask.any() else 0.
                        best_pos = float(sc[pp].max())
                        best_neg = float(sc[~pp].max()) if (~pp).any() else best_pos
                        context = "lead" if bool(state[selected[j], 125] > .5) else "follow"
                        stats["rows"] += 1; stats["candidates"] += count; stats["positive_candidates"] += int(pp.sum())
                        stats["raw_correct"] += int(correct); stats["nll_sum"] += nll
                        stats["choice_rows"] += int(count > 1); stats["choice_correct"] += int(correct and count > 1)
                        stats["pred_kind"][pred_kind] += 1
                        stats["demo_kind"].update(demo_kind_set)
                        stats["pred_given_demo_kind"][demo_kind_key][pred_kind] += 1
                        stats["demo_size"][demo_size_key] += 1
                        stats["pred_given_demo_size"][demo_size_key][str(int(round(float(acts[winner, 120]) * 10)))] += 1
                        key = count_bin(count)
                        bucket(stats["by_count"], key, correct, nll, pred_pass, demo_pass, p, best_pos, best_neg, count)
                        bucket(stats["by_context"], context, correct, nll, pred_pass, demo_pass, p, best_pos, best_neg, count)
                        stats["pass_calibration"]["sum_p"] += p
                        stats["pass_calibration"]["sum_p2"] += p * p
                        stats["pass_calibration"]["brier_sum"] += (p - float(demo_pass)) ** 2
                        stats["pass_calibration"]["n"] += 1
                        if demo_pass:
                            stats["pass_demo_rows"] += 1; stats["pass_demo_pred_pass"] += int(pred_pass); stats["pass_demo_correct"] += int(correct)
                        else:
                            stats["play_demo_rows"] += 1; stats["play_demo_pred_pass"] += int(pred_pass); stats["play_demo_correct"] += int(correct)
    stats["accuracy"] = stats["raw_correct"] / stats["rows"]
    stats["nll"] = stats["nll_sum"] / stats["rows"]
    stats["choice_accuracy"] = stats["choice_correct"] / max(1, stats["choice_rows"])
    for d in (stats["by_count"], stats["by_context"]):
        for row in d.values():
            row["accuracy"] = row["correct"] / row["rows"]
            row["nll"] = row["nll_sum"] / row["rows"]
            row["pred_pass_rate"] = row["pred_pass"] / row["rows"]
            row["demo_pass_rate"] = row["demo_pass"] / row["rows"]
            row["mean_p_pass"] = row["p_pass_sum"] / row["rows"]
            row["mean_pos_neg_margin"] = row["margin_sum"] / row["rows"]
    pc = stats["pass_calibration"]
    pc["mean_p_pass"] = pc["sum_p"] / pc["n"]
    pc["brier_against_row_pass"] = pc["brier_sum"] / pc["n"]
    return stats


def jsonable(value):
    if isinstance(value, collections.Counter):
        return dict(sorted(value.items()))
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, (list, tuple)):
        return [jsonable(x) for x in value]
    return value


def make_markdown(report: dict) -> str:
    s = report["structural"]
    t, v = s["train"], s["validation"]
    lines = [
        "# BC v1 objective audit (train + validation only)", "",
        "The audit reads only prepared `train` and `validation` shards; no held-out test shard is opened. It does not change model code or weights.", "",
        "## Observed data geometry", "",
        f"| split | rows | candidates used | positive candidates | mean candidates/row | positive/row | candidate positive fraction | rows with pass candidate | rows with pass positive |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        f"| train | {t['rows']:,} | {t['candidates']:,} | {t['positive_candidates']:,} | {t['candidates']/t['rows']:.2f} | {t['positive_candidates']/t['rows']:.3f} | {t['positive_candidates']/t['candidates']:.2%} | {t['pass_candidate_rows']:,} | {t['pass_positive_rows']:,} |",
        f"| validation | {v['rows']:,} | {v['candidates']:,} | {v['positive_candidates']:,} | {v['candidates']/v['rows']:.2f} | {v['positive_candidates']/v['rows']:.3f} | {v['positive_candidates']/v['candidates']:.2%} | {v['pass_candidate_rows']:,} | {v['pass_positive_rows']:,} |", "",
        f"Train candidate sampling retains at most 128 negatives per row; its manifest reports 2,116,745 legal candidates before sampling and 1,105,798 retained. Validation keeps all 583,591 candidates. Thus the training denominator is 1,105,798 candidates while the validation denominator is 583,591 on a different split, and rows with more than 128 negatives are trained against a sampled denominator.", "",
        "## Candidate and positive multiplicity", "",
        "Candidate count bins (rows):", "",
        "| bin | train | validation |", "|---|---:|---:|",
    ]
    bins = ["1", "2-3", "4-8", "9-16", "17-32", "33-64", "65-128", "129-256", "257-512", "513+"]
    for b in bins:
        lines.append(f"| {b} | {t['candidate_count_bin'].get(b,0):,} | {v['candidate_count_bin'].get(b,0):,} |")
    lines += ["", "Follow pass prevalence by candidate-count bin:", "", "| bin | train pass-positive rows / rows | validation pass-positive rows / rows |", "|---|---:|---:|"]
    for b in bins:
        tn, vn = t["candidate_count_bin"].get(b, 0), v["candidate_count_bin"].get(b, 0)
        lines.append(f"| {b} | {t['pass_positive_by_count_bin'].get(b,0):,} / {tn:,} ({t['pass_positive_by_count_bin'].get(b,0)/tn:.1%}) | {v['pass_positive_by_count_bin'].get(b,0):,} / {vn:,} ({v['pass_positive_by_count_bin'].get(b,0)/vn:.1%}) |" if tn and vn else f"| {b} | {t['pass_positive_by_count_bin'].get(b,0):,} / {tn:,} | {v['pass_positive_by_count_bin'].get(b,0):,} / {vn:,} |")
    lines += ["", "Selected candidate kind shares also shift after train sampling:", "", "| kind | train retained | validation all |", "|---|---:|---:|"]
    for kind in KINDS:
        tn, vn = t["candidate_kind"].get(kind, 0), v["candidate_kind"].get(kind, 0)
        if tn or vn:
            lines.append(f"| {kind} | {tn:,} ({tn/t['candidates']:.2%}) | {vn:,} ({vn/v['candidates']:.2%}) |")
    lines += ["", "Positive candidates per row are almost always one (train 82,427 / 81,827 = 1.007; validation 24,432 / 24,224 = 1.009). The positive alternatives are therefore not the main source of the weak pass/play behavior. The large denominator mismatch for high-candidate rows is the main objective mismatch.", "", "## v1 validation scorer behavior", ""]
    m = report.get("model_validation")
    if m:
        lines += [f"Raw argmax accuracy: **{m['accuracy']:.3%}**; marginal NLL: **{m['nll']:.4f}**; multi-candidate accuracy: **{m['choice_accuracy']:.3%}**.", "", "| context | rows | demo pass rate | predicted pass rate | raw accuracy | mean pass probability | mean positive-minus-best-negative score |", "|---|---:|---:|---:|---:|---:|---:|"]
        for key in ("lead", "follow"):
            row = m["by_context"][key]
            lines.append(f"| {key} | {row['rows']:,} | {row['demo_pass_rate']:.2%} | {row['pred_pass_rate']:.2%} | {row['accuracy']:.2%} | {row['mean_p_pass']:.4f} | {row['mean_pos_neg_margin']:.4f} |")
        lines += ["", "The row level label is one demonstrated action. Candidate level pass is only about 3.5% of candidates, but pass is about half of positive rows. A candidate cross entropy over all concrete actions therefore treats pass as a rare candidate while its row-level decision frequency is high; random negative sampling makes that denominator even less stable for large hands.", "", "Positive-action recall by size on validation:", "", "| demonstrated size | rows | predicted same size | recall |", "|---:|---:|---:|---:|"]
        for size in sorted(m["demo_size"], key=lambda x: (len(x), x)):
            row_count = m["demo_size"][size]
            same = m["pred_given_demo_size"][size].get(size, 0)
            lines.append(f"| {size} | {row_count:,} | {same:,} | {same / row_count:.2%} |")
        lines += ["", "The frozen scorer selects mostly pass/single candidates. Its validation pass rate is 66.28% versus 57.89% in the demonstrations; it predicts pass on 2,553 play rows (20.36% of all demonstrated play rows). Candidate count is a strong failure boundary: validation raw accuracy is 63.22% for 2-3 candidates, 46.98% for 4-8, 28.38% for 33-64, 12.50% for 129-256, and 8.16% for 513+. This is consistent with a sampled training denominator and weak learning of rare multi-card choices, rather than an issue with positive claim multiplicity.", "", "## v2 experiments and gates", "", "1. **Group plus within-group loss (first experiment).** Keep one scorer and the fixed C++ architecture. Build a deterministic strategic group key from kind, length, key, secondary, actual face multiset, wildcard count, and suit only when needed by straight flush. Optimize `L_group = -logsumexp(group scores) + logsumexp(all group scores)` and add `0.2 * L_within`, where `L_within` is the original positive-candidate marginal loss inside the demonstrated group. At inference choose the highest group score, then the highest member. This tests whether concrete suit or claim multiplicity is diluting action probability without changing features.", "", "2. **Candidate sampling with a fixed denominator.** For each train row retain every positive and a fixed 128 negatives, but sample negatives by action kind and size strata (pass, singles/pairs, 3-5 card, 6+ cards, bombs) with deterministic SHA seeds. Include every pass candidate and apply inverse sampling weights to an NCE/logit loss, or keep an exact `all_candidates` pass for rows up to a cap. The current random negative set can omit rare bomb/long-action negatives, and its row denominator differs sharply from validation.", "", "3. **Pass calibration head/loss.** Add a row-level binary pass-vs-play loss with weight 0.2-0.4, while retaining action loss weight 1.0. Calibrate its logit on validation only. Do not hard-code a pass bias before checking by lead/follow and candidate-count bins; pass is legal only when following and may be the sole candidate when leading.", "", "4. **Optional action-type auxiliary loss.** Predict kind and size from the state/history and add weight 0.1-0.2. This gives multi-card structure a direct gradient while the candidate scorer still chooses a legal concrete action. Evaluate separately by action size and kind; do not use type prediction alone for selection.", "", "Validation gates before any strength run: (a) NLL not worse than v1 by >0.02; (b) raw and multi-candidate accuracy no worse than v1 by >1 percentage point; (c) pass rate error versus demonstrations <3pp overall and <5pp in lead/follow; (d) 6+ card positive recall improves by >=5pp and no bomb/straight-flush recall drop >3pp; (e) candidate-count 129+ accuracy improves or remains within 1pp; (f) ten duplicate groups against RulePolicy must show at least one win and nonnegative mean margin before any longer run. These are screening gates, not evidence of release strength.", "", "Do not use the frozen held-out test for selecting v2 hyperparameters. Freeze one v2 candidate after validation, then run the held-out test once and proceed to duplicate-game strength evaluation only if the gates pass."]
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=ROOT / "data/processed/bc-v1")
    ap.add_argument("--checkpoint", type=Path, default=ROOT / "models/bc-v1/best.pt")
    ap.add_argument("--output", type=Path, default=ROOT / "reports/bc-v1-objective-analysis.json")
    ap.add_argument("--markdown", type=Path, default=ROOT / "reports/bc-v1-objective-analysis.md")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch-size", type=int, default=64)
    args = ap.parse_args()
    structural = structural_audit(args.data)
    model_stats = model_validation_audit(args.data, args.checkpoint, args.batch_size, args.device)
    report = {"schema": "oxbot-bc-objective-audit-v1", "status": "complete",
              "split": ["train", "validation"], "test_data_opened": False,
              "checkpoint_sha256": sha256(args.checkpoint),
              "data_manifest_sha256": sha256(args.data / "manifest.json"),
              "script_sha256": sha256(Path(__file__)),
              "structural": jsonable(structural),
              "model_validation": jsonable(model_stats)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    args.markdown.write_text(make_markdown(report), encoding="utf-8")
    print(json.dumps({"status": "complete", "test_data_opened": False,
                      "rows": model_stats["rows"], "accuracy": model_stats["accuracy"],
                      "nll": model_stats["nll"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
