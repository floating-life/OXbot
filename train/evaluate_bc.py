"""One held-out BC test evaluation after validation has frozen a checkpoint.

These metrics measure agreement with demonstrations over all legal candidates.
They are not match results, win rates, or estimates of BotZone playing strength.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path
import time

import torch

from export_model import RULES_CONTRACT
from features import FEATURE_VERSION
from model import CandidateModel, ModelConfig
from train_bc import atomic_json, measure, sha256_file, shard_paths


def evaluate(checkpoint_path: Path, expected_checkpoint_sha: str, data: Path, output: Path,
             *, training_report: Path | None = None, batch_size: int = 64,
             device_name: str = "cuda") -> dict:
    checkpoint_path, data, output = checkpoint_path.resolve(), data.resolve(), output.resolve()
    if output.exists():
        raise FileExistsError("evaluation output already exists; a held-out evaluation is not overwritten or rerun here")
    if batch_size < 1:
        raise ValueError("batch size must be positive")
    if sha256_file(checkpoint_path) != expected_checkpoint_sha:
        raise ValueError("checkpoint does not match the explicitly frozen SHA256")
    report_path = (training_report or checkpoint_path.with_name("training.json")).resolve()
    training = json.loads(report_path.read_text(encoding="utf-8"))
    if training.get("status") != "completed" or training.get("checkpoint_sha256") != expected_checkpoint_sha:
        raise ValueError("training must be completed and must identify this frozen checkpoint")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if checkpoint.get("config") != asdict(ModelConfig()):
        raise ValueError("checkpoint architecture differs from the fixed C++ v1 model contract")
    provenance = checkpoint.get("provenance", {})
    if provenance.get("test_used") is not False:
        raise ValueError("checkpoint provenance does not establish test-independent selection")
    if provenance.get("selected_epoch") != training.get("best_epoch"):
        raise ValueError("checkpoint is not the training report's validation-selected best epoch")

    data_manifest_path = data / "manifest.json"
    data_manifest_sha = sha256_file(data_manifest_path)
    manifest = json.loads(data_manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "complete" or manifest.get("feature_version") != FEATURE_VERSION:
        raise ValueError("prepared data manifest is incomplete or uses different features")
    if provenance.get("data_manifest_sha256") != data_manifest_sha:
        raise ValueError("checkpoint and held-out dataset do not share the same prepared-data manifest")
    files = shard_paths(data, "test")
    expected_hashes = {item["path"]: item["sha256"] for item in manifest["splits"]["test"]["shards"]}
    actual_hashes = {path.relative_to(data).as_posix(): sha256_file(path) for path in files}
    if expected_hashes != actual_hashes:
        raise ValueError("test shard set or SHA256 differs from the prepared manifest")
    distributions = manifest["splits"]["test"]["candidate_distributions"]
    if distributions["all_candidates"] != distributions["kept_candidates"]:
        raise ValueError("test split must retain all legal candidates")

    torch.set_num_threads(4)
    device = torch.device(device_name)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA evaluation requested but unavailable")
        torch.backends.cuda.matmul.allow_tf32 = False
    model = CandidateModel(ModelConfig(**checkpoint["config"]))
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model = model.float().to(device)
    model.eval()
    identity = {
        "checkpoint": str(checkpoint_path), "checkpoint_sha256": expected_checkpoint_sha,
        "selected_epoch": provenance["selected_epoch"], "selection_metric": "validation NLL only",
        "training_report_sha256": sha256_file(report_path), "data_manifest_sha256": data_manifest_sha,
        "test_shards": actual_hashes, "feature_version": FEATURE_VERSION,
        "rules_contract": RULES_CONTRACT, "model_config": checkpoint["config"],
        "script_sha256": sha256_file(__file__),
        "measure_script_sha256": sha256_file(Path(__file__).with_name("train_bc.py")),
        "torch": str(torch.__version__), "device": str(device), "batch_size": batch_size,
    }
    report = {
        "schema": "oxbot-bc-heldout-evaluation-v1", "status": "running", "provenance": identity,
        "evaluation_passes": 1, "split": "test", "all_candidates": True,
        "positive_policy": "any legal claim compatible with the demonstrated action is correct",
        "metric_interpretation": "demonstration agreement and marginal NLL; not gameplay win rate",
        "used_for_checkpoint_selection": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    # Reserve a durable audit record before the first model pass over held-out
    # samples. A failed run stays visible rather than silently being retried.
    with output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    started = time.monotonic()
    try:
        metrics = measure(model, files, batch_size, device)
        if metrics["samples"] != manifest["splits"]["test"]["counts"]["accepted_records"]:
            raise ValueError("evaluation sample count differs from manifest")
        if not all(math.isfinite(value) for value in metrics.values() if isinstance(value, float)):
            raise ValueError("evaluation produced non-finite metrics")
        if sha256_file(checkpoint_path) != expected_checkpoint_sha or sha256_file(data_manifest_path) != data_manifest_sha:
            raise ValueError("checkpoint or prepared manifest changed during evaluation")
        report.update(status="complete", metrics=metrics, seconds=time.monotonic() - started)
    except Exception as error:
        report.update(status="failed", error=str(error), seconds=time.monotonic() - started)
        atomic_json(output, report)
        raise
    atomic_json(output, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path("models/bc-v1/best.pt"))
    parser.add_argument("--expected-checkpoint-sha", required=True)
    parser.add_argument("--training-report", type=Path)
    parser.add_argument("--data", type=Path, default=Path("data/processed/bc-v1"))
    parser.add_argument("--output", type=Path, default=Path("reports/bc-v1-test.json"))
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    report = evaluate(args.checkpoint, args.expected_checkpoint_sha, args.data, args.output,
                      training_report=args.training_report, batch_size=args.batch_size, device_name=args.device)
    print(json.dumps({"status": report["status"], "checkpoint_sha256": args.expected_checkpoint_sha,
                      "metrics": report["metrics"], "seconds": report["seconds"],
                      "interpretation": report["metric_interpretation"]}), flush=True)


if __name__ == "__main__":
    main()
