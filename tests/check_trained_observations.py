"""Compare a frozen trained model on 128 real validation observations.

All candidates are retained. The sample is deterministic and covers history
length and candidate-count strata plus global extrema. This checks deployment
numerics, not demonstration agreement, gameplay strength, or hidden test data.
"""
from __future__ import annotations

import argparse
from bisect import bisect_left
from collections import Counter, defaultdict, deque
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import struct
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "tools"), str(ROOT / "train")]
from export_model import RULES_CONTRACT
from features import FEATURE_VERSION
from model import CandidateModel, ModelConfig
from probe import Probe

SAMPLE_COUNT = 128
TOLERANCE = 1e-4
HISTORY_BOUNDS = (32, 64, 128, 192, 256)
CANDIDATE_BOUNDS = (1, 4, 16, 64, 256)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stratum(record: dict) -> tuple[int, int]:
    return (bisect_left(HISTORY_BOUNDS, record["history_length"]),
            bisect_left(CANDIDATE_BOUNDS, record["candidate_count"]))


def select_records(records: list[dict], seed: int) -> list[dict]:
    """Seeded stable ordering, global extrema, then round-robin across strata."""
    if len(records) < SAMPLE_COUNT:
        raise ValueError(f"at least {SAMPLE_COUNT} validation records are required")
    rank = lambda record: (hashlib.sha256(f"{seed}:{record['id']}".encode("utf-8")).digest(),
                           record["shard"], record["row"])
    ordered = sorted(records, key=rank)
    selected, seen = [], set()

    def add(record):
        identity = (record["shard"], record["row"])
        if identity not in seen:
            seen.add(identity)
            selected.append(record)

    # min/max preserve the seeded order when several observations tie.
    for key in ("history_length", "candidate_count"):
        add(min(ordered, key=lambda record: record[key]))
        add(max(ordered, key=lambda record: record[key]))
    groups = defaultdict(deque)
    for record in ordered:
        groups[stratum(record)].append(record)
    while len(selected) < SAMPLE_COUNT:
        added = False
        for key in sorted(groups):
            group = groups[key]
            while group and (group[0]["shard"], group[0]["row"]) in seen:
                group.popleft()
            if group:
                add(group.popleft())
                added = True
            if len(selected) == SAMPLE_COUNT:
                break
        if not added:
            raise ValueError("insufficient distinct validation records")
    return selected


def inspect_validation(data: Path, manifest: dict) -> tuple[list[dict], dict[str, str]]:
    split = manifest["splits"]["validation"]
    distributions = split["candidate_distributions"]
    if distributions["all_candidates"] != distributions["kept_candidates"]:
        raise ValueError("validation must retain all legal candidates")
    descriptors = split["shards"]
    expected_paths = {item["path"] for item in descriptors}
    if expected_paths != {path.relative_to(data).as_posix() for path in (data / "validation").glob("shard-*.npz")}:
        raise ValueError("validation shard set differs from the prepared manifest")
    records, hashes, ids_seen = [], {}, set()
    for item in descriptors:
        relative = Path(item["path"])
        if relative.is_absolute() or relative.parts[0] != "validation" or ".." in relative.parts:
            raise ValueError("only validation shards may be loaded")
        path = data / relative
        digest = sha256_file(path)
        if digest != item["sha256"]:
            raise ValueError(f"validation shard SHA256 differs: {relative}")
        hashes[item["path"]] = digest
        with np.load(path, allow_pickle=False) as shard:
            lengths, offsets, ids = shard["lengths"], shard["offsets"], shard["ids"]
            count = item["records"]
            if lengths.shape != (count,) or offsets.shape != (count + 1,) or ids.shape != (count,):
                raise ValueError(f"invalid validation index shapes: {relative}")
            if lengths.dtype != np.int16 or offsets.dtype != np.int64 or ids.dtype.kind != "U":
                raise ValueError(f"invalid validation index dtypes: {relative}")
            counts = np.diff(offsets)
            if offsets[0] != 0 or offsets[-1] != item["candidates"] or (counts < 1).any():
                raise ValueError(f"invalid candidate offsets: {relative}")
            if (lengths < 2).any() or (lengths > 256).any():
                raise ValueError(f"invalid history lengths: {relative}")
            for row in range(count):
                identity = str(ids[row])
                if identity in ids_seen:
                    raise ValueError(f"duplicate source record ID: {identity}")
                ids_seen.add(identity)
                records.append({"shard": item["path"], "row": row, "id": identity,
                                "history_length": int(lengths[row]), "candidate_count": int(counts[row])})
    if len(records) != split["counts"]["accepted_records"]:
        raise ValueError("validation count differs from the manifest")
    return records, hashes


def inspect_model(path: Path, model: CandidateModel) -> dict:
    content = path.read_bytes()
    if content[:8] != b"OXGDQ001" or len(content) < 12:
        raise ValueError("invalid deployment artifact magic")
    header_size = struct.unpack("<I", content[8:12])[0]
    if not 0 < header_size <= 65536 or len(content) < 12 + header_size:
        raise ValueError("invalid deployment artifact header size")
    header = json.loads(content[12:12 + header_size])
    if (header["config"] != asdict(ModelConfig()) or header["feature_version"] != FEATURE_VERSION
            or header["rules_contract"] != RULES_CONTRACT or header["dtype"] != "float32-le"
            or header["architecture"] != ModelConfig().architecture):
        raise ValueError("deployment artifact contract mismatch")
    payload = content[12 + header_size:]
    tensors, digest, offset = [], hashlib.sha256(), 0
    for name, tensor in model.state_dict().items():
        array = tensor.detach().cpu().float().numpy().astype("<f4")
        if not np.isfinite(array).all():
            raise ValueError(f"non-finite checkpoint tensor: {name}")
        tensors.append({"name": name, "shape": list(array.shape), "count": array.size, "offset": offset})
        raw = array.tobytes(order="C")
        digest.update(raw)
        offset += len(raw)
    if (tensors != header["tensors"] or len(payload) != offset
            or hashlib.sha256(payload).hexdigest() != header["payload_sha256"]
            or digest.hexdigest() != header["payload_sha256"]):
        raise ValueError("deployment payload does not match the frozen checkpoint")
    return header


def top_margin(scores: np.ndarray) -> float | None:
    if scores.size == 1:
        return None
    top = np.partition(scores.astype(np.float64), -2)[-2:]
    return float(top.max() - top.min())


def run(args) -> dict:
    checkpoint_path, model_path = args.checkpoint.resolve(), args.model.resolve()
    data, probe_path = args.data.resolve(), args.probe.resolve()
    checkpoint_sha, model_sha = sha256_file(checkpoint_path), sha256_file(model_path)
    if checkpoint_sha != args.expected_checkpoint_sha:
        raise ValueError("checkpoint does not match the explicitly frozen SHA256")
    if args.expected_model_sha and model_sha != args.expected_model_sha:
        raise ValueError("deployment artifact does not match the explicitly frozen SHA256")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if checkpoint["config"] != asdict(ModelConfig()):
        raise ValueError("checkpoint configuration differs from the fixed C++ v1 contract")
    manifest_path = data / "manifest.json"
    manifest_sha = sha256_file(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "complete" or manifest.get("feature_version") != FEATURE_VERSION:
        raise ValueError("prepared manifest is incomplete or uses different features")
    if checkpoint["provenance"]["data_manifest_sha256"] != manifest_sha:
        raise ValueError("checkpoint and validation data have different manifests")
    torch.set_num_threads(1)
    model = CandidateModel(ModelConfig())
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.float().eval()
    header = inspect_model(model_path, model)
    records, hashes = inspect_validation(data, manifest)
    selected = select_records(records, args.seed)
    strata_all, strata_selected = Counter(map(stratum, records)), Counter(map(stratum, selected))
    by_shard = defaultdict(list)
    for index, record in enumerate(selected):
        by_shard[record["shard"]].append((index, record))
    cases = [None] * len(selected)
    probe_sha = sha256_file(probe_path)
    started = time.monotonic()
    with torch.inference_mode(), Probe(probe_path, timeout=15) as probe:
        for relative, group in sorted(by_shard.items()):
            with np.load(data / relative, allow_pickle=False) as shard:
                tokens, state, actions, offsets = (shard[key] for key in ("tokens", "state", "actions", "offsets"))
                count = len(shard["lengths"])
                if (tokens.shape != (count, 256) or tokens.dtype != np.uint8
                        or state.shape != (count, 128) or state.dtype != np.float32
                        or actions.shape != (int(offsets[-1]), 128) or actions.dtype != np.float32):
                    raise ValueError(f"invalid feature shape/dtype: {relative}")
                for index, record in group:
                    row, length = record["row"], record["history_length"]
                    if (tokens[row, :length] == 0).any() or (tokens[row, :length] >= 128).any() or (tokens[row, length:] != 0).any():
                        raise ValueError(f"invalid right-PAD token history: {record['id']}")
                    token_array = tokens[row, :length].astype(np.int64)
                    state_array = np.ascontiguousarray(state[row])
                    action_array = np.ascontiguousarray(actions[offsets[row]:offsets[row + 1]])
                    if not np.isfinite(state_array).all() or not np.isfinite(action_array).all():
                        raise ValueError(f"non-finite observation features: {record['id']}")
                    expected = model(torch.from_numpy(token_array)[None], torch.tensor([length]),
                                     torch.from_numpy(state_array)[None], torch.from_numpy(action_array)[None])[0].numpy()
                    actual = probe.call(path=str(model_path), tokens=token_array.tolist(), state=state_array.tolist(),
                                        actions=action_array.tolist(), trace=False)
                    if not actual.get("ok") or actual.get("sha") != header["payload_sha256"]:
                        raise ValueError(f"C++ rejected the frozen model: {actual}")
                    scores = np.asarray(actual["scores"], dtype=np.float64)
                    if scores.shape != expected.shape or not np.isfinite(scores).all() or not np.isfinite(expected).all():
                        raise ValueError(f"invalid Python/C++ score vector: {record['id']}")
                    error = float(np.max(np.abs(expected.astype(np.float64) - scores)))
                    torch_best, cpp_best = int(np.argmax(expected)), int(np.argmax(scores))
                    same = torch_best == cpp_best
                    # Two scores can reverse order within twice the observed
                    # vector error. Record this without relabeling the argmax.
                    tie_bound = 2 * error + 1e-12
                    torch_gap = float(expected[torch_best]) - float(expected[cpp_best])
                    cpp_gap = float(scores[cpp_best] - scores[torch_best])
                    cases[index] = dict(record, max_abs_error=error, torch_argmax=torch_best, cpp_argmax=cpp_best,
                                        same_argmax=same, torch_top2_margin=top_margin(expected),
                                        cpp_top2_margin=top_margin(scores), torch_winner_gap=torch_gap,
                                        cpp_winner_gap=cpp_gap, numerical_tie_bound=tie_bound,
                                        disagreement_within_numerical_tie=bool(not same and torch_gap <= tie_bound and cpp_gap <= tie_bound),
                                        cpp_milliseconds=float(actual["milliseconds"]))
    for path, original in ((checkpoint_path, checkpoint_sha), (model_path, model_sha),
                           (manifest_path, manifest_sha), (probe_path, probe_sha)):
        if sha256_file(path) != original:
            raise ValueError(f"input changed during parity check: {path}")
    for relative, digest in hashes.items():
        if sha256_file(data / relative) != digest:
            raise ValueError(f"validation shard changed during parity check: {relative}")
    maximum = max(case["max_abs_error"] for case in cases)
    agreements = sum(case["same_argmax"] for case in cases)
    numerical_ties = sum(case["disagreement_within_numerical_tie"] for case in cases)
    timings = [case["cpp_milliseconds"] for case in cases]
    status = "passed" if maximum <= TOLERANCE and agreements == SAMPLE_COUNT else "failed"
    return {
        "schema": "oxbot-trained-observation-parity-v1", "status": status,
        "scope": "CPU float32 Torch versus C++ on real validation observations; not gameplay strength",
        "split": "validation", "all_candidates": True, "test_split_read": False,
        "seed": args.seed, "samples": SAMPLE_COUNT, "validation_population": len(records),
        "absolute_error_tolerance": TOLERANCE, "max_absolute_error": maximum,
        "scores_within_tolerance": maximum <= TOLERANCE, "argmax_agreements": agreements,
        "argmax_disagreements": SAMPLE_COUNT - agreements,
        "disagreements_within_numerical_tie": numerical_ties,
        "argmax_policy": "exact first-max index; numerical ties are disclosed and still count as disagreements",
        "checkpoint": str(checkpoint_path), "checkpoint_sha256": checkpoint_sha,
        "model": str(model_path), "model_sha256": model_sha, "payload_sha256": header["payload_sha256"],
        "data_manifest_sha256": manifest_sha, "validation_shards": hashes,
        "probe_sha256": probe_sha, "script_sha256": sha256_file(Path(__file__)),
        "model_config": checkpoint["config"], "feature_version": FEATURE_VERSION,
        "rules_contract": RULES_CONTRACT, "torch": str(torch.__version__), "torch_device": "cpu",
        "torch_threads": 1, "seconds": time.monotonic() - started,
        "cpp_ms": {"p50": float(np.percentile(timings, 50)), "p95": float(np.percentile(timings, 95)),
                   "max": max(timings), "scope": "probe-reported scoring only; excludes model loading and JSON transport"},
        "selection": {"method": "SHA256(seed:ID) ordering, extrema, round-robin across nonempty strata",
                      "history_bounds_inclusive": list(HISTORY_BOUNDS),
                      "candidate_bounds_inclusive": list(CANDIDATE_BOUNDS),
                      "last_candidate_bin": ">256", "history_min": min(r["history_length"] for r in selected),
                      "history_max": max(r["history_length"] for r in selected),
                      "candidates_min": min(r["candidate_count"] for r in selected),
                      "candidates_max": max(r["candidate_count"] for r in selected),
                      "strata": [{"history_bin": key[0], "candidate_bin": key[1],
                                  "population": strata_all[key], "selected": strata_selected[key]}
                                 for key in sorted(strata_all)]},
        "cases": cases,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "models/bc-v1/best.pt")
    parser.add_argument("--expected-checkpoint-sha", required=True)
    parser.add_argument("--model", type=Path, default=ROOT / "models/oxbot-bc-v1.bin")
    parser.add_argument("--expected-model-sha")
    parser.add_argument("--data", type=Path, default=ROOT / "data/processed/bc-v1")
    parser.add_argument("--probe", type=Path, default=ROOT / "bin/network_probe")
    parser.add_argument("--report", type=Path, default=ROOT / "reports/trained-observation-parity.json")
    parser.add_argument("--seed", type=int, default=20261001)
    args = parser.parse_args()
    report = run(args)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("status", "samples", "max_absolute_error",
                      "scores_within_tolerance", "argmax_agreements", "argmax_disagreements",
                      "disagreements_within_numerical_tie", "cpp_ms", "seconds")}), flush=True)
    if report["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
