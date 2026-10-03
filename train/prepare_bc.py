"""Prepare information-set-only, multi-positive BC shards from NUPT decisions.

Inputs are read-only. Candidate legality and claim variants come from the C++
rule core; this script never uses results or hidden opponent allocations.
"""
from __future__ import annotations

import argparse
import collections
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np

TRAIN_DIR = Path(__file__).resolve().parent
ROOT = TRAIN_DIR.parent
sys.path.insert(0, str(ROOT / "tools"))
from probe import Probe
from features import FEATURE_VERSION, RANKS, action_features, history_tokens, matching_actions, state_features

SCHEMA = "oxbot-bc-shards-v1"
DECISION_SCHEMA = "njupt-decision-v3"
SPLITS = ("train", "validation", "test")
DEFAULT_SOURCE = ROOT / "data" / "processed" / "njupt"
DEFAULT_OUTPUT = ROOT / "data" / "processed" / "bc-v1"
DEFAULT_PROBE = ROOT / "bin" / "core_probe_rules"


class PreparationError(RuntimeError):
    pass


class SkipRecord(ValueError):
    def __init__(self, reason: str, detail: str = ""):
        super().__init__(detail or reason)
        self.reason = reason
        self.detail = detail


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ensure_safe_output(source: Path, output: Path) -> tuple[Path, Path]:
    source = source.resolve()
    output = output.resolve()
    if output == source or output.is_relative_to(source) or source.is_relative_to(output):
        raise PreparationError("output must be a separate sibling of the read-only source, never an ancestor or descendant")
    if output.exists() and any(output.iterdir()):
        raise PreparationError(f"output directory is not empty; refusing to overwrite: {output}")
    return source, output


def card_array(value: Any, field: str, *, unique: bool = False) -> list[int]:
    if not isinstance(value, list) or any(type(card) is not int or not 0 <= card < 108 for card in value):
        raise SkipRecord("invalid_observation", f"{field} is not a valid card array")
    if unique and len(value) != len(set(value)):
        raise SkipRecord("invalid_observation", f"{field} repeats a physical card")
    return list(value)


def observation_from_record(record: dict[str, Any]) -> dict[str, Any]:
    """Whitelist the acting player's information set; do not copy the record.

    In particular, neither ``result`` nor any opponent hand field is accessed.
    Extra fields from future ETL revisions cannot silently enter the features.
    """
    source = record.get("features")
    if not isinstance(source, dict):
        raise SkipRecord("invalid_observation", "features is not an object")
    hand = card_array(source.get("own_hand"), "own_hand", unique=True)
    if not 1 <= len(hand) <= 27:
        raise SkipRecord("invalid_observation", "acting hand size is outside 1..27")
    player = source.get("seat")
    if type(player) is not int or not 0 <= player < 4:
        raise SkipRecord("invalid_observation", "seat is outside 0..3")
    level = source.get("level_label")
    if level == "10":
        level = "0"
    if not isinstance(level, str) or len(level) != 1 or level not in RANKS:
        raise SkipRecord("invalid_observation", "unknown level_label")
    leading = source.get("leading")
    if type(leading) is not bool:
        raise SkipRecord("invalid_observation", "leading must be boolean")
    remaining = source.get("remaining_counts")
    if not isinstance(remaining, list) or len(remaining) != 4 or any(type(x) is not int or not 0 <= x <= 27 for x in remaining):
        raise SkipRecord("invalid_observation", "remaining_counts must contain four counts in 0..27")
    if remaining[player] != len(hand):
        raise SkipRecord("invalid_observation", "own hand count disagrees with remaining_counts")
    played = source.get("played_face_counts")
    if not isinstance(played, list) or len(played) != 54 or any(type(x) is not int or not 0 <= x <= 2 for x in played):
        raise SkipRecord("invalid_observation", "played_face_counts must contain 54 counts in 0..2")
    tribute = source.get("tribute", 0)
    resist = source.get("resist", False)
    if type(tribute) is not int or tribute not in (0, 1, 2) or type(resist) is not bool:
        raise SkipRecord("invalid_observation", "invalid tribute/resist context")
    history_source = source.get("history")
    if not isinstance(history_source, list):
        raise SkipRecord("invalid_observation", "history must be a list")
    if len(history_source) > 128:
        raise SkipRecord("invalid_observation", "ETL history exceeds the agreed 128 play events; refusing silent truncation")
    history = []
    for event in history_source:
        if not isinstance(event, dict):
            raise SkipRecord("invalid_observation", "history event must be an object")
        owner = event.get("player")
        if type(owner) is not int or not 0 <= owner < 4:
            raise SkipRecord("invalid_observation", "history player is outside 0..3")
        action = card_array(event.get("action"), "history.action", unique=True)
        raw_claim = event.get("claim")
        claim = None if raw_claim is None else card_array(raw_claim, "history.claim")
        if claim is not None and len(action) != len(claim):
            raise SkipRecord("invalid_observation", "history action/claim length mismatch")
        history.append({"player": owner, "action": action, "claim": claim})
    result = {
        "hand": hand, "player": player, "level": level, "leading": leading,
        "remaining_counts": list(remaining), "played_face_counts": list(played),
        "history": history, "tribute": tribute, "resist": resist,
    }
    if not leading:
        previous_claim = source.get("last_claim")
        if previous_claim is None:
            raise SkipRecord("uncertain_previous", "following an action whose wildcard claim is unobserved")
        previous_cards = card_array(source.get("last_cards"), "last_cards", unique=True)
        previous_claim = card_array(previous_claim, "last_claim")
        if not previous_cards or len(previous_cards) != len(previous_claim):
            raise SkipRecord("invalid_previous", "missing previous cards or mismatched claim length")
        result["previous"] = [previous_cards, previous_claim]
    return result


def record_id(record: dict[str, Any]) -> str:
    provenance = record.get("provenance")
    if not isinstance(provenance, dict):
        raise SkipRecord("missing_provenance", "provenance is not an object")
    source_hash = provenance.get("data_sha256")
    if not isinstance(source_hash, str) or len(source_hash) != 64:
        raise SkipRecord("missing_provenance", "data_sha256 is missing")
    deal = record.get("deal_id")
    event = record.get("event_index")
    if type(deal) is not int or type(event) is not int or deal < 0 or event < 0:
        raise SkipRecord("missing_provenance", "deal/event indices are missing")
    identity = f"{source_hash[:16]}:d{deal}:e{event}"
    if len(identity) > 80:
        raise SkipRecord("missing_provenance", "source indices do not fit the stable sample-id contract")
    return identity


def select_candidates(count: int, positives: list[int], split: str, seed: int,
                      identity: str, max_negatives: int) -> list[int]:
    if not positives:
        raise PreparationError("candidate sampling received no positive action")
    positive_set = set(positives)
    if len(positive_set) != len(positives) or min(positives) < 0 or max(positives) >= count:
        raise PreparationError("invalid positive indices")
    if split != "train":
        return list(range(count))
    negatives = np.asarray([i for i in range(count) if i not in positive_set], dtype=np.int64)
    if len(negatives) > max_negatives:
        digest = hashlib.sha256(f"{seed}:{identity}".encode("utf-8")).digest()
        local_seed = int.from_bytes(digest[:8], "little")
        rng = np.random.default_rng(local_seed)
        negatives = rng.choice(negatives, size=max_negatives, replace=False)
    selected = sorted([*positives, *(int(x) for x in negatives)])
    if not positive_set.issubset(selected):
        raise PreparationError("sampling would drop a positive label")
    return selected


def raw_history_token_length(observation: dict[str, Any]) -> int:
    count = 2  # BOS/EOS
    for event in observation["history"]:
        count += 2  # player and end-of-event
        if not event["action"]:
            count += 1
        else:
            count += len(event["action"]) + 1  # cards + action/claim separator
            count += 1 if event["claim"] is None else len(event["claim"])
    return count


def encode_record(record: dict[str, Any], split: str, probe: Probe, *, seed: int,
                  max_negatives: int) -> tuple[dict[str, Any], dict[str, int]]:
    if record.get("stage") != "play":
        raise SkipRecord("non_play_stage")
    identity = record_id(record)
    observation = observation_from_record(record)
    previous = observation.get("previous")
    if previous is not None:
        previous_type = probe.call(command="classify", claim=previous[1])
        if previous_type.get("kind") in (None, "invalid", "pass"):
            raise SkipRecord("invalid_previous", "C++ classifies previous claim as invalid")
    generated = probe.call(command="generate", hand=observation["hand"],
                           level=observation["level"], leading=observation["leading"],
                           previous=previous, metadata=True)
    if "error" in generated:
        raise PreparationError(f"C++ generator error for {identity}: {generated['error']}")
    moves = generated.get("moves")
    types = generated.get("types")
    if not isinstance(moves, list) or not isinstance(types, list) or len(moves) != len(types):
        raise PreparationError(f"C++ generator metadata mismatch for {identity}")
    if not moves:
        raise SkipRecord("no_legal_candidates")
    label = record.get("label")
    if not isinstance(label, dict):
        raise SkipRecord("invalid_label", "label is not an object")
    demonstrated = card_array(label.get("cards"), "label.cards", unique=True)
    positives = matching_actions(moves, demonstrated)
    if not positives:
        raise SkipRecord("demonstration_not_legal", "demonstrated face multiset has no legal candidate")
    selected = select_candidates(len(moves), positives, split, seed, identity, max_negatives)
    positive_set = set(positives)
    action_rows = []
    for index in selected:
        meta = types[index]
        if not isinstance(meta, dict):
            raise PreparationError(f"invalid metadata for {identity}")
        action_rows.append(action_features(moves[index], meta["kind"], meta["key"],
                                          meta["secondary"], observation["level"], len(observation["hand"])))
    state = state_features(observation)
    tokens = history_tokens(observation, limit=256)
    if state.shape != (128,) or len(tokens) > 256 or not np.isfinite(state).all():
        raise PreparationError(f"feature contract violation for {identity}")
    if len(tokens) < 2 or np.any(tokens < 0) or np.any(tokens > 255):
        raise PreparationError(f"token contract violation for {identity}")
    actions = np.asarray(action_rows, dtype=np.float32)
    if actions.shape != (len(selected), 128) or not np.isfinite(actions).all():
        raise PreparationError(f"action feature contract violation for {identity}")
    mask = np.asarray([index in positive_set for index in selected], dtype=np.bool_)
    if int(mask.sum()) != len(positives):
        raise PreparationError(f"positive labels changed while encoding {identity}")
    padded = np.zeros(256, dtype=np.uint8)
    padded[:len(tokens)] = tokens.astype(np.uint8)
    feature_source = record["features"]
    stats = {
        "all_candidates": len(moves), "kept_candidates": len(selected), "positive_candidates": len(positives),
        "raw_history_tokens": raw_history_token_length(observation),
        "token_window_truncated": int(raw_history_token_length(observation) > 256),
        "etl_history_window_truncated": int(feature_source.get("history_total", len(observation["history"])) > len(observation["history"])),
    }
    return {"state": state, "tokens": padded, "length": len(tokens), "actions": actions,
            "positives": mask, "id": identity}, stats


class ShardWriter:
    def __init__(self, output: Path, split: str, size: int):
        self.output, self.split, self.size = output, split, size
        self.pending: list[dict[str, Any]] = []
        self.files: list[dict[str, Any]] = []

    def add(self, record: dict[str, Any]) -> None:
        self.pending.append(record)
        if len(self.pending) >= self.size:
            self.flush()

    def flush(self) -> None:
        if not self.pending:
            return
        index = len(self.files)
        target = self.output / self.split / f"shard-{index:05d}.npz"
        target.parent.mkdir(parents=True, exist_ok=True)
        lengths = [len(record["actions"]) for record in self.pending]
        offsets = np.concatenate((np.zeros(1, dtype=np.int64), np.cumsum(lengths, dtype=np.int64)))
        state = np.stack([record["state"] for record in self.pending]).astype(np.float32, copy=False)
        tokens = np.stack([record["tokens"] for record in self.pending]).astype(np.uint8, copy=False)
        token_lengths = np.asarray([record["length"] for record in self.pending], dtype=np.int16)
        actions = np.concatenate([record["actions"] for record in self.pending]).astype(np.float32, copy=False)
        positives = np.concatenate([record["positives"] for record in self.pending]).astype(np.bool_, copy=False)
        ids = np.asarray([record["id"] for record in self.pending], dtype="U80")
        if len(set(ids.tolist())) != len(ids):
            raise PreparationError(f"duplicate sample IDs within {self.split} shard {index}")
        with target.open("xb") as stream:
            np.savez_compressed(stream, actions=actions, offsets=offsets, positives=positives,
                                state=state, tokens=tokens, lengths=token_lengths, ids=ids)
        self.files.append({"path": target.relative_to(self.output).as_posix(), "records": len(self.pending),
                           "candidates": int(offsets[-1]), "bytes": target.stat().st_size,
                           "sha256": sha256_file(target)})
        self.pending.clear()


def distribution(values: list[int]) -> dict[str, Any]:
    if not values:
        return {"count": 0}
    array = np.asarray(values, dtype=np.int64)
    return {"count": len(values), "min": int(array.min()), "max": int(array.max()),
            "mean": float(array.mean()), "p50": float(np.quantile(array, .5)),
            "p95": float(np.quantile(array, .95)), "p99": float(np.quantile(array, .99)),
            "sum": int(array.sum())}


def process_split(config: dict[str, Any]) -> dict[str, Any]:
    source = Path(config["source"])
    output = Path(config["output"])
    split = config["split"]
    counters: collections.Counter[str] = collections.Counter()
    examples: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    metrics: dict[str, list[int]] = collections.defaultdict(list)
    writer = ShardWriter(output, split, config["shard_size"])
    sample_ids: set[str] = set()
    match_ids: set[str] = set()
    provenance_sources: dict[str, str] = {}
    started = time.monotonic()
    last_progress = started
    with Probe(Path(config["probe"]), timeout=config["timeout"]) as probe, source.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            counters["input_records"] += 1
            try:
                record = json.loads(line)
                if not isinstance(record, dict):
                    raise SkipRecord("malformed_record", "JSON value is not an object")
                if record.get("schema") != DECISION_SCHEMA:
                    raise PreparationError(f"source row uses an unexpected schema at {source}:{line_number}")
                match_id = record.get("match_id")
                if isinstance(match_id, str):
                    match_ids.add(match_id)
                encoded, stats = encode_record(record, split, probe, seed=config["seed"],
                                               max_negatives=config["max_negatives"])
                if encoded["id"] in sample_ids:
                    raise PreparationError(f"duplicate sample ID: {encoded['id']}")
                sample_ids.add(encoded["id"])
                if not isinstance(match_id, str):
                    raise PreparationError("accepted record has no match_id")
                provenance = record["provenance"]
                provenance_sources[provenance["data_sha256"]] = str(provenance.get("source_data", ""))
                writer.add(encoded)
                counters["accepted_records"] += 1
                for name, value in stats.items():
                    if name.endswith("truncated"):
                        counters[name + "_records"] += value
                    else:
                        metrics[name].append(value)
            except json.JSONDecodeError as exc:
                raise PreparationError(f"malformed source JSON at {source}:{line_number}: {exc}") from exc
            except SkipRecord as exc:
                counters["filtered_records"] += 1
                counters["filtered_" + exc.reason] += 1
                if len(examples[exc.reason]) < 5:
                    evidence = record if isinstance(record, dict) else {}
                    examples[exc.reason].append({"line": line_number,
                                                 "game_id": evidence.get("game_id"),
                                                 "deal_id": evidence.get("deal_id"),
                                                 "event_index": evidence.get("event_index"),
                                                 "detail": exc.detail})
            now = time.monotonic()
            if now - last_progress >= 30:
                print(json.dumps({"split": split, "input": counters["input_records"],
                                  "accepted": counters["accepted_records"], "seconds": round(now - started, 1)}), flush=True)
                last_progress = now
    writer.flush()
    if counters["accepted_records"] == 0:
        raise PreparationError(f"no accepted records in {split}; refusing an empty training split")
    return {"split": split, "source": source.name, "source_sha256": config["source_sha256"],
            "counts": dict(counters), "filters": dict(examples),
            "candidate_distributions": {name: distribution(values) for name, values in metrics.items()},
            "shards": writer.files, "match_ids": sorted(match_ids),
            "provenance_sources": provenance_sources, "seconds": time.monotonic() - started}


def run(source: Path, output: Path, probe: Path, *, seed: int = 20261001,
        max_negatives: int = 128, shard_size: int = 512, workers: int = 3,
        timeout: float = 10) -> dict[str, Any]:
    source, output = ensure_safe_output(source, output)
    probe = probe.resolve()
    if not probe.is_file():
        raise PreparationError(f"probe is missing: {probe}")
    if shard_size <= 0 or max_negatives < 0 or workers <= 0:
        raise PreparationError("invalid shard size, negative limit or worker count")
    inputs = {split: source / (split + ".jsonl") for split in SPLITS}
    if any(not path.is_file() for path in inputs.values()):
        raise PreparationError("source must contain train.jsonl, validation.jsonl, and test.jsonl")
    source_manifest = source / "manifest.json"
    if not source_manifest.is_file():
        raise PreparationError("source manifest is required before preparing a training dataset")
    with source_manifest.open("r", encoding="utf-8") as stream:
        source_info = json.load(stream)
    if source_info.get("decision_schema") != DECISION_SCHEMA:
        raise PreparationError(f"source must use {DECISION_SCHEMA}; rebuild ETL with the complete history window")
    source_manifest_hash = sha256_file(source_manifest)
    hashes = {split: sha256_file(path) for split, path in inputs.items()}
    scripts = {"prepare_bc.py": sha256_file(Path(__file__)), "features.py": sha256_file(TRAIN_DIR / "features.py"),
               "probe.py": sha256_file(ROOT / "tools" / "probe.py")}
    probe_hash = sha256_file(probe)
    output.mkdir(parents=True, exist_ok=True)
    configs = [{"source": str(inputs[split]), "output": str(output), "split": split,
                "source_sha256": hashes[split], "probe": str(probe), "seed": seed,
                "max_negatives": max_negatives, "shard_size": shard_size, "timeout": timeout}
               for split in SPLITS]
    started = time.monotonic()
    if workers == 1:
        results = [process_split(config) for config in configs]
    else:
        with ProcessPoolExecutor(max_workers=min(workers, 3)) as pool:
            results = list(pool.map(process_split, configs))
    for index, left in enumerate(results):
        for right in results[index + 1:]:
            overlap = set(left["match_ids"]) & set(right["match_ids"])
            if overlap:
                raise PreparationError(f"match-group leakage across splits: {sorted(overlap)}")
            shared_sources = set(left["provenance_sources"]) & set(right["provenance_sources"])
            if shared_sources:
                raise PreparationError(f"source-data leakage across splits: {sorted(shared_sources)}")
    final_hashes = {split: sha256_file(path) for split, path in inputs.items()}
    if hashes != final_hashes:
        raise PreparationError("source changed while preparing shards; refusing a successful manifest")
    if sha256_file(probe) != probe_hash:
        raise PreparationError("probe binary changed during preparation; rebuild with a stable binary path")
    if sha256_file(source_manifest) != source_manifest_hash:
        raise PreparationError("source manifest changed during preparation")
    final_scripts = {"prepare_bc.py": sha256_file(Path(__file__)), "features.py": sha256_file(TRAIN_DIR / "features.py"),
                     "probe.py": sha256_file(ROOT / "tools" / "probe.py")}
    if final_scripts != scripts:
        raise PreparationError("preparation or feature scripts changed during preparation")
    manifest = {
        "schema": SCHEMA, "feature_version": FEATURE_VERSION, "status": "complete",
        "seed": seed, "shard_size": shard_size, "train_max_negatives": max_negatives,
        "positive_policy": "all legal candidates matching the demonstrated action face multiset",
        "sampling": "per-record SHA-derived RNG; train only; all positives retained; validation/test keep every candidate",
        "observations": "acting hand + public context only; result and opponent allocations never read",
        "history_window": {"source_events": "ETL's most recent 128 public play events", "token_length": 256,
                           "policy": "authoritative features.history_tokens: BOS + most recent 254 raw tokens + EOS; right pad 0",
                           "truncation_counts_are_reported": True},
        "arrays": {"state": "float32 [N,128]", "tokens": "uint8 [N,256] right padded 0",
                   "lengths": "int16 [N]", "actions": "float32 [sum candidates,128]",
                   "offsets": "int64 [N+1]", "positives": "bool [sum candidates]", "ids": "unicode [N]"},
        "source_manifest_sha256": source_manifest_hash,
        "source_splits_sha256": hashes, "probe_sha256": probe_hash, "scripts_sha256": scripts,
        "splits": {item["split"]: item for item in results}, "seconds": time.monotonic() - started,
    }
    with (output / "manifest.json").open("x", encoding="utf-8") as stream:
        json.dump(manifest, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--probe", type=Path, default=DEFAULT_PROBE)
    parser.add_argument("--seed", type=int, default=20261001)
    parser.add_argument("--max-negatives", type=int, default=128)
    parser.add_argument("--shard-size", type=int, default=512)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=10)
    args = parser.parse_args()
    manifest = run(args.source, args.output, args.probe, seed=args.seed, max_negatives=args.max_negatives,
                   shard_size=args.shard_size, workers=args.workers, timeout=args.timeout)
    summary = {"status": manifest["status"], "output": str(args.output), "seconds": manifest["seconds"],
               "counts": {split: result["counts"] for split, result in manifest["splits"].items()}}
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
