# -*- coding: utf-8 -*-
"""Apply the promotion gate for a self-play/duplicate evaluation run.

The training/evaluation launcher deliberately keeps this decision in a small,
side-effect-free module.  It validates the evidence first and only then applies
the one promotion rule documented in ``docs/TRAINING_5080.md``:

* at least 1,000 paired deals;
* the candidate is stronger than ``real-v2`` (and the current champion when
  one is required), with a paired-bootstrap 95% CI whose lower endpoint is
  strictly positive; and
* the official judge completed the requested number of games with
  ``require_model`` enabled and zero errors.

The duplicate reports are produced by ``tools/duplicate_eval.py``.  The judge
runner does not know about the frozen candidate's identity, so the launcher
must add ``candidate_sha256`` (and ``baseline_sha256`` when the baseline is
frozen) to ``judge.json`` before calling this
module.  Training metadata is read from ``training_status.json`` unless the
caller explicitly supplies ``--training-ok``.

Exit codes are intentionally distinct:

``0``
    Complete, valid evidence and the candidate is promoted.
``3``
    Complete, valid evidence but a normal gate is not satisfied.
``4``
    Evidence is missing, malformed, non-finite, or has an inconsistent
    identity.  This is an abnormal run and must not be treated as a champion
    decision.
"""

from __future__ import print_function

import argparse
import hashlib
import json
import math
import os
import re
import sys


PROMOTE = 0
NO_PROMOTE = 3
INVALID_EVIDENCE = 4
MIN_DEALS = 1000
SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")


class EvidenceError(ValueError):
    """A report cannot be used as promotion evidence."""


def _reject_constant(value):
    raise EvidenceError("non-finite JSON constant: %s" % value)


def _check_finite(value, path="report"):
    """Reject NaN/Infinity even when a JSON parser accepts them."""
    if isinstance(value, float):
        if not math.isfinite(value):
            raise EvidenceError("non-finite value at %s" % path)
    elif isinstance(value, dict):
        for key, item in value.items():
            _check_finite(item, "%s.%s" % (path, key))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _check_finite(item, "%s[%d]" % (path, index))


def _read_json(path):
    if not os.path.isfile(path):
        raise EvidenceError("missing report: %s" % path)
    try:
        with open(path, "r", encoding="utf-8") as stream:
            value = json.load(stream, parse_constant=_reject_constant)
    except EvidenceError:
        raise
    except (OSError, ValueError) as exc:
        raise EvidenceError("invalid JSON %s: %s" % (path, exc))
    _check_finite(value, os.path.basename(path))
    if not isinstance(value, dict):
        raise EvidenceError("report root must be an object: %s" % path)
    return value


def _is_int(value):
    return isinstance(value, int) and not isinstance(value, bool)


def _number(value, label):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise EvidenceError("%s must be numeric" % label)
    try:
        value = float(value)
    except (OverflowError, ValueError):
        raise EvidenceError("%s must be finite" % label)
    if not math.isfinite(value):
        raise EvidenceError("%s must be finite" % label)
    return value


def _sha256_file(path):
    if not os.path.isfile(path):
        raise EvidenceError("missing frozen artifact: %s" % path)
    digest = hashlib.sha256()
    try:
        with open(path, "rb") as stream:
            for chunk in iter(lambda: stream.read(1 << 20), b""):
                digest.update(chunk)
    except OSError as exc:
        raise EvidenceError("cannot read frozen artifact %s: %s" % (path, exc))
    return digest.hexdigest()


def _report_sha(value, label):
    if not isinstance(value, str) or not SHA256_RE.match(value):
        raise EvidenceError("%s must be a 64-hex SHA-256" % label)
    return value.lower()


def _require_string(value, label):
    if not isinstance(value, str) or not value:
        raise EvidenceError("%s must be a non-empty string" % label)
    return value


def _expected_verdict(lo, hi):
    if lo > 0:
        return "A_STRONGER"
    if hi < 0:
        return "A_WEAKER"
    return "INCONCLUSIVE"


def _frozen_reference(eval_dir, kind):
    """Return (path, sha) for a frozen comparison model, when present."""
    names = {
        "realv2": ("baseline.npz",),
        "champion": ("previous_champion.npz", "champion.npz"),
    }[kind]
    for name in names:
        path = os.path.join(eval_dir, name)
        if os.path.exists(path):
            return path, _sha256_file(path)
    return None, None


def _validate_duplicate(path, name, expected_deals, expected_seed,
                        candidate_sha, reference_sha=None):
    report = _read_json(path)
    required = ("a", "b", "deals", "games", "seed", "ladder_frac",
                "avg_score_diff_per_game", "ci95_per_game", "verdict")
    for key in required:
        if key not in report:
            raise EvidenceError("%s missing key %s" % (path, key))
    if not _is_int(report["deals"]) or report["deals"] < 1:
        raise EvidenceError("%s.deals must be a positive integer" % path)
    if report["deals"] != expected_deals:
        raise EvidenceError("%s.deals=%s, expected %s" %
                            (path, report["deals"], expected_deals))
    if not _is_int(report["games"]) or report["games"] != 2 * report["deals"]:
        raise EvidenceError("%s.games must equal 2 * deals" % path)
    if not _is_int(report["seed"]) or report["seed"] != expected_seed:
        raise EvidenceError("%s.seed does not match the fixed evaluation seed" % path)
    _number(report["ladder_frac"], "%s.ladder_frac" % path)
    if not 0.0 <= float(report["ladder_frac"]) <= 1.0:
        raise EvidenceError("%s.ladder_frac outside [0, 1]" % path)
    _number(report["avg_score_diff_per_game"],
            "%s.avg_score_diff_per_game" % path)
    ci = report["ci95_per_game"]
    if not isinstance(ci, list) or len(ci) != 2:
        raise EvidenceError("%s.ci95_per_game must contain two values" % path)
    lo = _number(ci[0], "%s.ci95_per_game[0]" % path)
    hi = _number(ci[1], "%s.ci95_per_game[1]" % path)
    if lo > hi:
        raise EvidenceError("%s has reversed CI endpoints" % path)
    verdict = report["verdict"]
    if verdict != _expected_verdict(lo, hi):
        raise EvidenceError("%s.verdict disagrees with the reported CI" % path)

    a = report["a"]
    b = report["b"]
    if not isinstance(a, dict) or not isinstance(b, dict):
        raise EvidenceError("%s.a and .b must be objects" % path)
    _require_string(a.get("spec"), "%s.a.spec" % path)
    actual_candidate = _report_sha(a.get("sha256"), "%s.a.sha256" % path)
    if candidate_sha is not None and actual_candidate != candidate_sha:
        raise EvidenceError("%s candidate SHA does not match candidate.npz" % path)

    b_spec = _require_string(b.get("spec"), "%s.b.spec" % path)
    actual_reference = None
    if name == "rule":
        if b_spec != "rule":
            raise EvidenceError("%s must compare against rule" % path)
        # Rule has no file hash.  If a hash is supplied, it is still required
        # to be well formed so a malformed optional field cannot be ignored.
        if "sha256" in b:
            _report_sha(b["sha256"], "%s.b.sha256" % path)
    else:
        actual_reference = _report_sha(b.get("sha256"), "%s.b.sha256" % path)
        if candidate_sha is not None and actual_reference == candidate_sha:
            raise EvidenceError("%s compares candidate against itself" % path)
        if reference_sha is not None and actual_reference != reference_sha:
            raise EvidenceError("%s reference SHA does not match frozen artifact" % path)

    return {
        "path": os.path.abspath(path),
        "deals": report["deals"],
        "games": report["games"],
        "seed": report["seed"],
        "avg": float(report["avg_score_diff_per_game"]),
        "ci_lo": lo,
        "ci_hi": hi,
        "verdict": verdict,
        "candidate_sha256": actual_candidate,
        "reference_sha256": actual_reference,
        "ladder_frac": float(report["ladder_frac"]),
    }


def _validate_judge(path, expected_games, expected_seed, exit_code,
                    candidate_sha, baseline_sha=None, champion_sha=None):
    report = _read_json(path)
    required = ("games", "errors", "require_model", "failures",
                "candidate_sha256", "seed")
    for key in required:
        if key not in report:
            raise EvidenceError("%s missing key %s" % (path, key))
    if not _is_int(report["games"]) or report["games"] != expected_games:
        raise EvidenceError("%s.games does not match requested judge games" % path)
    if not _is_int(report["errors"]) or report["errors"] < 0:
        raise EvidenceError("%s.errors must be a nonnegative integer" % path)
    if not isinstance(report["require_model"], bool):
        raise EvidenceError("%s.require_model must be boolean" % path)
    if not isinstance(report["failures"], list):
        raise EvidenceError("%s.failures must be a list" % path)
    if not _is_int(report["seed"]) or report["seed"] != expected_seed:
        raise EvidenceError("%s.seed does not match the fixed evaluation seed" % path)
    actual_candidate = _report_sha(report["candidate_sha256"],
                                   "%s.candidate_sha256" % path)
    if candidate_sha is not None and actual_candidate != candidate_sha:
        raise EvidenceError("%s candidate SHA does not match candidate.npz" % path)

    refs = {}
    for key, expected in (("baseline_sha256", baseline_sha),
                          ("champion_sha256", champion_sha)):
        if expected is not None:
            # The judge invocation compares candidate vs baseline.  A current
            # champion is validated by vs_champion.json, but is not necessarily
            # loaded by judge_runner, so champion_sha256 is an optional
            # provenance field while baseline_sha256 is required here.
            if key not in report and key == "baseline_sha256":
                raise EvidenceError("%s missing key %s" % (path, key))
            if key not in report:
                continue
            actual = _report_sha(report[key], "%s.%s" % (path, key))
            if actual != expected:
                raise EvidenceError("%s does not match frozen reference" % path)
            refs[key] = actual
        elif key in report:
            refs[key] = _report_sha(report[key], "%s.%s" % (path, key))

    return {
        "path": os.path.abspath(path),
        "games": report["games"],
        "errors": report["errors"],
        "require_model": report["require_model"],
        "failures": len(report["failures"]),
        "exit_code": exit_code,
        "candidate_sha256": actual_candidate,
        "references": refs,
        "ok": (exit_code == 0 and report["errors"] == 0 and
               report["require_model"] is True and
               len(report["failures"]) == 0),
    }


def _validate_training(eval_dir, candidate_sha, explicit_ok):
    path = os.path.join(eval_dir, "training_status.json")
    if not os.path.isfile(path):
        if explicit_ok:
            return {"ok": True, "source": "--training-ok"}
        raise EvidenceError("missing training_status.json (or --training-ok)")
    report = _read_json(path)
    if "training_ok" not in report or not isinstance(report["training_ok"], bool):
        raise EvidenceError("training_status.json.training_ok must be boolean")
    if "candidate_sha256" not in report:
        raise EvidenceError("training_status.json missing candidate_sha256")
    actual = _report_sha(report["candidate_sha256"],
                         "training_status.json.candidate_sha256")
    if candidate_sha is not None and actual != candidate_sha:
        raise EvidenceError("training metadata candidate SHA mismatch")
    if "exit_code" in report:
        if not _is_int(report["exit_code"]):
            raise EvidenceError("training_status.json.exit_code must be integer")
        exit_ok = report["exit_code"] == 0
    else:
        exit_ok = True
    return {
        "ok": report["training_ok"] is True and exit_ok,
        "source": "training_status.json",
        "training_ok": report["training_ok"],
        "exit_code": report.get("exit_code"),
        "candidate_sha256": actual,
    }


def assess(eval_dir, deals=1000, judge_games=200, seed=20261101,
           judge_exit_code=0, require_champion=False, training_ok=False):
    """Return ``(summary, exit_code)`` without writing files."""
    eval_dir = os.path.abspath(eval_dir)
    summary = {
        "schema": 1,
        "eval_dir": eval_dir,
        "deals": deals,
        "min_deals": MIN_DEALS,
        "judge_games": judge_games,
        "seed": seed,
        "require_champion": bool(require_champion),
        "promote": False,
        "status": "INVALID_EVIDENCE",
        "errors": [],
        "gate_failures": [],
        "checks": {},
    }
    hard_errors = summary["errors"]
    if not os.path.isdir(eval_dir):
        hard_errors.append("evaluation directory not found: %s" % eval_dir)
        return summary, INVALID_EVIDENCE
    if not _is_int(deals) or deals < 1:
        hard_errors.append("--deals must be a positive integer")
    if not _is_int(judge_games) or judge_games < 1:
        hard_errors.append("--judge-games must be a positive integer")
    if not _is_int(seed):
        hard_errors.append("--seed must be an integer")
    candidate_path = os.path.join(eval_dir, "candidate.npz")
    candidate_sha = None
    try:
        candidate_sha = _sha256_file(candidate_path)
        summary["candidate_sha256"] = candidate_sha
    except EvidenceError as exc:
        hard_errors.append(str(exc))

    refs = {}
    for kind in ("realv2", "champion"):
        try:
            refs[kind] = _frozen_reference(eval_dir, kind)[1]
        except EvidenceError as exc:
            hard_errors.append(str(exc))
            refs[kind] = None
    summary["frozen_references"] = {key: value for key, value in refs.items()
                                    if value is not None}
    champion_required = bool(require_champion or refs["champion"] is not None)
    summary["champion_required"] = champion_required

    reports = {}
    if not hard_errors and _is_int(deals) and _is_int(seed):
        for name in ("realv2", "rule"):
            try:
                reports[name] = _validate_duplicate(
                    os.path.join(eval_dir, "vs_%s.json" % name), name,
                    deals, seed, candidate_sha,
                    refs["realv2"] if name == "realv2" else None)
            except EvidenceError as exc:
                hard_errors.append(str(exc))
        champion_path = os.path.join(eval_dir, "vs_champion.json")
        # A frozen previous champion makes the comparison mandatory even when
        # a caller forgets --require-champion.  The explicit flag also records
        # that a champion is required when its frozen artifact is missing.
        if os.path.isfile(champion_path) or champion_required:
            try:
                reports["champion"] = _validate_duplicate(
                    champion_path, "champion", deals, seed, candidate_sha,
                    refs["champion"])
            except EvidenceError as exc:
                hard_errors.append(str(exc))
    summary["checks"]["duplicate"] = reports

    try:
        training = _validate_training(eval_dir, candidate_sha, training_ok)
        summary["checks"]["training"] = training
    except EvidenceError as exc:
        hard_errors.append(str(exc))

    judge = None
    if not hard_errors and _is_int(judge_games) and _is_int(seed):
        try:
            judge = _validate_judge(
                os.path.join(eval_dir, "judge.json"), judge_games, seed,
                judge_exit_code, candidate_sha, refs["realv2"], refs["champion"])
            summary["checks"]["judge"] = judge
        except EvidenceError as exc:
            hard_errors.append(str(exc))

    if hard_errors:
        return summary, INVALID_EVIDENCE

    # A report can be complete and valid while a normal statistical/legality
    # gate is not met.  Keep these distinct from malformed evidence above.
    if deals < MIN_DEALS:
        summary["gate_failures"].append(
            "duplicate deals %d below required minimum %d" % (deals, MIN_DEALS))
    if reports["realv2"]["ci_lo"] <= 0:
        summary["gate_failures"].append("real-v2 CI lower endpoint is not > 0")
    if "champion" in reports and reports["champion"]["ci_lo"] <= 0:
        summary["gate_failures"].append("champion CI lower endpoint is not > 0")
    if "champion" not in reports and champion_required:
        # This is normally caught as a missing-report evidence error; retain a
        # defensive gate so callers cannot accidentally promote if validation
        # logic is changed later.
        summary["gate_failures"].append("current champion comparison is missing")
    if not summary["checks"].get("training", {}).get("ok", False):
        summary["gate_failures"].append("training did not complete successfully")
    if not summary["checks"].get("judge", {}).get("ok", False):
        summary["gate_failures"].append("official judge gate failed")

    summary["promote"] = not summary["gate_failures"]
    summary["status"] = "PROMOTED" if summary["promote"] else "NOT_PROMOTED"
    return summary, PROMOTE if summary["promote"] else NO_PROMOTE


def _write_summary(eval_dir, summary):
    path = os.path.join(eval_dir, "summary.json")
    temporary = path + ".tmp"
    with open(temporary, "w", encoding="utf-8") as stream:
        json.dump(summary, stream, ensure_ascii=False, indent=2,
                  allow_nan=False, sort_keys=True)
        stream.write("\n")
    os.replace(temporary, path)
    return path


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("eval_dir", help="directory containing frozen reports")
    parser.add_argument("--judge-exit-code", type=int, default=0)
    parser.add_argument("--deals", type=int, default=1000)
    parser.add_argument("--judge-games", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20261101)
    parser.add_argument("--require-champion", action="store_true",
                        help="require vs_champion.json instead of allowing a first run")
    parser.add_argument("--training-ok", action="store_true",
                        help="assert training succeeded when training_status.json is absent")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    summary, code = assess(
        args.eval_dir, deals=args.deals, judge_games=args.judge_games,
        seed=args.seed, judge_exit_code=args.judge_exit_code,
        require_champion=args.require_champion, training_ok=args.training_ok)
    try:
        summary_path = _write_summary(os.path.abspath(args.eval_dir), summary)
    except (OSError, ValueError) as exc:
        print("INVALID EVIDENCE: cannot write summary: %s" % exc,
              file=sys.stderr)
        return INVALID_EVIDENCE
    for name, row in summary.get("checks", {}).get("duplicate", {}).items():
        print("vs %-9s %+.3f/game  95%% CI [%+.3f, %+.3f]  %s" %
              (name, row["avg"], row["ci_lo"], row["ci_hi"], row["verdict"]))
    print("summary: %s" % summary_path)
    if code == PROMOTE:
        print("PROMOTE")
    elif code == NO_PROMOTE:
        print("KEEP CURRENT CHAMPION")
        for reason in summary.get("gate_failures", []):
            print("  gate: %s" % reason)
    else:
        print("INVALID EVIDENCE")
        for reason in summary.get("errors", []):
            print("  error: %s" % reason, file=sys.stderr)
    return code


if __name__ == "__main__":
    sys.exit(main())
