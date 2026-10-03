"""Evaluate the factorized candidate on prepared validation shards only."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "train"))
sys.path.insert(0, str(ROOT / "tools"))
from compare_bc_policies import eval_policy, md, sha  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=ROOT / "data" / "processed" / "bc-v2-full")
    parser.add_argument("--candidate", type=Path, default=ROOT / "models" / "bc-v3-factorized005" / "best.pt")
    parser.add_argument("--candidate-name", default="factorized005",
                        help="label used for the candidate policy and provenance")
    parser.add_argument("--training-report", type=Path,
                        help="training.json sidecar recorded in the validation report")
    parser.add_argument("--anchor", type=Path, default=ROOT / "models" / "bc-v2-full-fp32" / "best.pt")
    parser.add_argument("--output", type=Path, default=ROOT / "reports" / "factorized005_validation.json")
    parser.add_argument("--markdown", type=Path, default=ROOT / "reports" / "factorized005_validation.md")
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()
    files = sorted((args.data / "validation").glob("*.npz"))
    if not files:
        raise ValueError("validation shards required")
    specs = {"anchor": args.anchor, args.candidate_name: args.candidate}
    report = {
        "schema": "oxbot-factorized-validation-v1", "status": "complete",
        "split": "validation", "test_data_opened": False,
        "validation_shards_sha256": {path.name: sha(path) for path in files},
        "policies": {},
    }
    for name, path in specs.items():
        report["policies"][name] = eval_policy(path, files, args)
    training_report = args.training_report
    if training_report is None:
        training_report = args.candidate.parent / "training.json"
    report["candidate_training_report"] = str(training_report.resolve())
    report["script_sha256"] = sha(__file__)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    # Reuse the established comparison renderer; it intentionally reports no
    # test metrics and keeps the anchor/candidate metrics side by side.
    args.markdown.write_text(md({"policies": report["policies"]}) + "\n", encoding="utf-8")
    print(json.dumps({name: value["all"] for name, value in report["policies"].items()}, ensure_ascii=False))


if __name__ == "__main__":
    main()
