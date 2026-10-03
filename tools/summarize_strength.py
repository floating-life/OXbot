"""Audit duplicate deals and bootstrap complete two-leg groups, never legs.

Each deal is played twice with complementary model teams. A group's point
statistic is the average model-minus-RulePolicy score across its two legs;
its win statistic is 0, 0.5 or 1. Bootstrap draws resample these joint groups.
"""
from __future__ import annotations

import argparse
import collections
import json
import math
from pathlib import Path
import random
import re
import sys

from local_judge import atomic_json, model_identity, sha256_file

SCHEMA = "oxbot-duplicate-strength-v1"
EXPECTED_MODEL_PAYLOAD = "13ad6ba714fe39fe5c5b3bb36d21c8b450e981ff59547f6c9d01a2edf7d5b7ec"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def valid_sha(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def audited_pairs(first, second, expected_pairs=None, expected_seed=None):
    reports = (first, second)
    for report in reports:
        require(report.get("schema") == "oxbot-local-judge-v2", "unsupported local-judge report schema")
        require(report.get("status") == "passed" and not report.get("failure"), "both reports must have passed")
        require(report.get("require_model") is True, "reports must require the model on configured model seats")
        require(str(report.get("model_team")) in ("0", "1"), "each report must configure one model team")
        for key in ("oracle_sha256", "probe_sha256"):
            require(valid_sha(report.get(key)), f"missing/invalid {key}")
        for key in ("file_sha256", "payload_sha256"):
            require(valid_sha(report.get("model", {}).get(key)), f"missing/invalid model {key}")
        fallback = report.get("fallback_counts", {})
        require(not fallback.get("model_requested_but_not_used", 0), "a required model seat used the fallback")
        require(not fallback.get("legality_recheck_fallback", 0), "a model move needed legality fallback")
        require(report.get("play_policies", {}).get("model", 0) > 0, "no actual model play recorded")
        prefix = report["model"]["payload_sha256"][:12]
        observed = report.get("observed_model_payload_sha_prefix_counts", {})
        require(set(observed) == {prefix} and observed[prefix] > 0, "wrong or missing observed model payload")
        records = report.get("game_records", [])
        count = report.get("requested_games")
        require(type(count) is int and count > 0 and len(records) == count == report.get("counts", {}).get("games"),
                "report game counts are incomplete")
        require({row.get("index") for row in records} == set(range(count)), "record indices are missing or duplicated")
        require(not report.get("counts", {}).get("failures", 0), "report contains failed games")

    require({str(report["model_team"]) for report in reports} == {"0", "1"}, "model teams must be complementary")
    for key in ("seed", "requested_games", "oracle_sha256", "probe_sha256", "process_sha256", "mode"):
        require(first.get(key) == second.get(key), f"paired reports differ: {key}")
    for key in ("file_sha256", "payload_sha256", "architecture", "feature_version", "rules_contract"):
        require(first["model"].get(key) == second["model"].get(key), f"paired model identity differs: {key}")
    if expected_pairs is not None:
        require(first["requested_games"] == expected_pairs, "unexpected number of duplicate groups")
    if expected_seed is not None:
        require(first["seed"] == expected_seed, "unexpected evaluation seed")

    maps = []
    for report in reports:
        mapping = {}
        for row in report["game_records"]:
            key = row.get("deal_key")
            require(valid_sha(key), "invalid deal key")
            require(key not in mapping, "duplicate deal key within one report")
            require(str(row.get("model_team")) == str(report["model_team"]), "record/report model team mismatch")
            require(row.get("seed") == report["seed"] + row["index"], "record seed is not the declared schedule")
            mapping[key] = row
        maps.append(mapping)
    require(set(maps[0]) == set(maps[1]), "deal-key sets do not match exactly")

    paired = []
    for key in sorted(maps[0], key=lambda value: maps[0][value]["index"]):
        legs = sorted((mapping[key] for mapping in maps), key=lambda row: int(row["model_team"]))
        for field in ("index", "seed", "level", "tribute", "first", "last"):
            require(legs[0].get(field) == legs[1].get(field), f"same deal key has different {field}")
        margins, wins = [], []
        for row in legs:
            scores = row.get("team_scores")
            require(isinstance(scores, list) and len(scores) == 2 and all(type(v) is int and 0 <= v <= 3 for v in scores),
                    "invalid team score pair")
            require((scores[0] == 0) != (scores[1] == 0), "each leg must have one nonzero team score")
            winner = 0 if scores[0] else 1
            require(row.get("winner_team") == winner, "score and winner disagree")
            model_team = int(row["model_team"])
            margins.append(scores[model_team] - scores[1 - model_team])
            wins.append(int(winner == model_team))
        paired.append({"index": legs[0]["index"], "seed": legs[0]["seed"], "deal_key": key,
                       "model_point_margins": margins, "model_wins": wins,
                       "mean_model_point_margin_per_game": sum(margins) / 2,
                       "model_win_fraction_per_game": sum(wins) / 2})
    summaries = [report["paired_scores"] for report in reports if "paired_scores" in report]
    require(summaries, "one report must have been produced with --paired-report")
    for summary in summaries:
        require(summary.get("pairs") == len(paired) and summary.get("unmatched_first") == 0 and summary.get("unmatched_second") == 0,
                "local-judge paired summary has incomplete pairing")
        require(summary.get("total_model_point_margin") == sum(sum(row["model_point_margins"]) for row in paired),
                "local-judge paired score disagrees with audited records")
    return paired


def percentile(ordered, probability):
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    return ordered[lower] * (upper - position) + ordered[upper] * (position - lower) if lower != upper else ordered[lower]


def paired_bootstrap(pairs, repetitions=20000, seed=20262001):
    require(len(pairs) > 0 and repetitions >= 100, "bootstrap requires groups and at least 100 repetitions")
    values = [(row["mean_model_point_margin_per_game"], row["model_win_fraction_per_game"]) for row in pairs]
    size = len(values)
    point_samples, win_samples = [], []
    rng = random.Random(seed)
    for _ in range(repetitions):
        points, wins = 0.0, 0.0
        for _ in range(size):
            point, win = values[rng.randrange(size)]
            points += point
            wins += win
        point_samples.append(points / size)
        win_samples.append(wins / size)
    result = {}
    for name, column, samples, reference in (("model_point_margin_per_game", 0, point_samples, 0.0),
                                              ("model_win_fraction_per_game", 1, win_samples, 0.5)):
        samples.sort()
        interval = [percentile(samples, .025), percentile(samples, .975)]
        result[name] = {"estimate": sum(row[column] for row in values) / size, "ci95_percentile": interval,
                        "equal_performance_reference": reference,
                        "relation_to_reference": "below" if interval[1] < reference else "above" if interval[0] > reference else "overlaps"}
    return result


def summarize(first, second, *, expected_pairs=None, expected_seed=None, repetitions=20000, bootstrap_seed=20262001):
    pairs = audited_pairs(first, second, expected_pairs, expected_seed)
    totals = [sum(row["model_point_margins"]) for row in pairs]
    wins = sum(sum(row["model_wins"]) for row in pairs)
    return {"schema": SCHEMA, "status": "complete", "baseline": "RulePolicy",
            "scope": "basic launch capability against the repository heuristic baseline; not a strong reference opponent",
            "evaluation_seed": first["seed"], "duplicate_groups": len(pairs), "games": 2 * len(pairs),
            "model_wins": wins, "rule_wins": 2 * len(pairs) - wins,
            "total_model_point_margin": sum(totals),
            "duplicate_group_outcomes": {"positive": sum(x > 0 for x in totals), "tie": sum(x == 0 for x in totals), "negative": sum(x < 0 for x in totals)},
            "metrics": paired_bootstrap(pairs, repetitions, bootstrap_seed),
            "bootstrap": {"seed": bootstrap_seed, "repetitions": repetitions, "confidence": .95,
                          "method": "percentile, linear interpolation", "resampling_unit": "whole two-leg duplicate group",
                          "point_statistic": "mean over groups of (model score difference in leg0 + leg1) / 2",
                          "win_statistic": "mean over groups of (model win indicator in leg0 + leg1) / 2",
                          "independent_bernoulli_interval": False,
                          "limitation": "conditional on this fixed game schedule and baseline; percentile intervals can collapse when every sampled group has the same outcome"},
            "identity": {"model": first["model"], "probe_sha256": first["probe_sha256"],
                         "oracle_sha256": first["oracle_sha256"], "process_sha256": first.get("process_sha256")},
            "leg_descriptions": [{"model_team": report["model_team"], "model_vs_rule": report.get("model_vs_rule"),
                                  "required_model_fallbacks": report.get("fallback_counts", {}).get("model_requested_but_not_used", 0),
                                  "local_wall_clock_budget_exceedances": report.get("local_wall_clock_budget_exceedances", {})}
                                 for report in (first, second)],
            "model_or_policy_adjusted_from_results": False, "pairs": pairs}


def markdown_report(report):
    points = report["metrics"]["model_point_margin_per_game"]
    wins = report["metrics"]["model_win_fraction_per_game"]
    outcome = report["duplicate_group_outcomes"]
    return f"""# BC v1 对 RulePolicy 同牌换座评测

固定 {report['duplicate_groups']} 副牌，每副互换模型所在队伍，共 {report['games']} 盘。模型获胜 {report['model_wins']} 盘，RulePolicy 获胜 {report['rule_wins']} 盘。

| 指标 | 估计 | 配对 bootstrap 95% CI |
| --- | ---: | --- |
| 每盘模型得分差（模型 − RulePolicy） | {points['estimate']:.4f} | [{points['ci95_percentile'][0]:.4f}, {points['ci95_percentile'][1]:.4f}] |
| 每盘模型胜率 | {wins['estimate']:.2%} | [{wins['ci95_percentile'][0]:.2%}, {wins['ci95_percentile'][1]:.2%}] |

配对组总分差：正 {outcome['positive']} 组，平 {outcome['tie']} 组，负 {outcome['negative']} 组。

使用固定 seed={report['bootstrap']['seed']}、{report['bootstrap']['repetitions']} 次重采样，每次整体抽取同一副牌的两盘；没有把 {report['games']} 盘当作独立 Bernoulli 试验计算 Wilson 区间。若所有组的某项结果完全相同，百分位 bootstrap 区间可能退化，不能据此声称总体概率恰好为 0 或 1。

RulePolicy 是仓库内的基础启发式策略，本报告只评估首发基础能力，不能代表对 FableDan、DanLM 或其他强参考对手的水平。两次报告的 deal_key、发牌参数、model/engine/oracle SHA 均一致；模型侧强制使用冻结权重，没有因本次结果调整模型或策略。

- 评测 seed：{report['evaluation_seed']}
- 模型 payload SHA-256：`{report['identity']['model']['payload_sha256']}`
- 引擎 SHA-256：`{report['identity']['probe_sha256']}`
- 裁判 SHA-256：`{report['identity']['oracle_sha256']}`

本次使用离线复用进程模式，同时存在其他本地压力测试；其耗时不是 BotZone 的 CPU 时间证明。完整逐组结果、两份源报告哈希及配对核查见同名 JSON。
"""


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("first", type=Path)
    parser.add_argument("second", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-pairs", type=int)
    parser.add_argument("--expected-seed", type=int)
    parser.add_argument("--bootstrap-seed", type=int, default=20262001)
    parser.add_argument("--repetitions", type=int, default=20000)
    parser.add_argument("--expected-model-payload", default=EXPECTED_MODEL_PAYLOAD)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--probe", type=Path)
    parser.add_argument("--judge", type=Path)
    parser.add_argument("--expected-harness-sha")
    args = parser.parse_args(argv)
    first, second = [json.loads(path.read_text(encoding="utf-8")) for path in (args.first, args.second)]
    result = summarize(first, second, expected_pairs=args.expected_pairs, expected_seed=args.expected_seed,
                       repetitions=args.repetitions, bootstrap_seed=args.bootstrap_seed)
    require(result["identity"]["model"]["payload_sha256"] == args.expected_model_payload, "unexpected frozen model payload")
    if args.model:
        identity = model_identity(args.model)
        require(identity["file_sha256"] == result["identity"]["model"]["file_sha256"], "model binary changed since evaluation")
    for path, field in ((args.probe, "probe_sha256"), (args.judge, "oracle_sha256")):
        if path:
            require(sha256_file(path) == result["identity"][field], f"{field} changed since evaluation")
    harness_sha = sha256_file(Path(__file__).with_name("local_judge.py"))
    if args.expected_harness_sha:
        require(harness_sha == args.expected_harness_sha, "evaluation harness changed")
    result["provenance"] = {"reports": [{"path": str(path.resolve()), "sha256": sha256_file(path)} for path in (args.first, args.second)],
                            "summary_script_sha256": sha256_file(__file__), "harness_sha256": harness_sha}
    atomic_json(args.output, result)
    args.output.with_suffix(".md").write_text(markdown_report(result), encoding="utf-8")
    print(json.dumps({"output": str(args.output.resolve()), "duplicate_groups": result["duplicate_groups"],
                      "games": result["games"], "model_wins": result["model_wins"], "metrics": result["metrics"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(2)
