"""Build conservative candidate-level public-continuation targets.

This is a development-only sidecar generator.  It opens only the ``train``
and ``validation`` decision shards from the ETL source.  It never reads a
``result`` field, an opponent hand, or the held-out ``test`` split.

The target is deliberately weaker than a Q value.  For a candidate at one
observation, the script substitutes that candidate for the current action and
checks whether the *already observed* public continuation remains legal until
the next row marked ``leading=true``.  A legal candidate that is compatible
with that observed continuation receives ``+1``; a legal, face-feasible
candidate that makes the first observed reply impossible receives ``-1``.
Candidates for which the continuation is unsupported, ambiguous, or exceeds
the fixed horizon are ignored.  This is an observational consistency label,
not a counterfactual game value and not a team-win or Q label.

The generated JSON is a development sidecar for a future optional pairwise
ranking loss; the current release model does not consume it.  The model
architecture and C++ feature contract do not change.  Any malformed source
row, probe disagreement, or candidate-order mismatch is reported and prevents
a trainable sidecar rather than inventing a label.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import tempfile
from typing import Any, Iterable

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
import sys

sys.path.insert(0, str(ROOT / "tools"))
from probe import Probe  # noqa: E402

sys.path.insert(0, str(ROOT / "train"))
from prepare_bc import observation_from_record, record_id  # noqa: E402


SCHEMA = "oxbot-candidate-continuation-targets-v1"
DECISION_SCHEMA = "njupt-decision-v3"
SPLITS = ("train", "validation")
DEFAULT_SOURCE = ROOT / "data" / "processed" / "njupt"
DEFAULT_PREPARED = ROOT / "data" / "processed" / "bc-v2-full"
DEFAULT_PROBE = ROOT / "bin" / "core_probe_rules"


class TargetBuildError(RuntimeError):
    """Raised when a sidecar would be unsafe to train from."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _cards(value: Any, name: str, *, unique: bool = False) -> list[int]:
    if not isinstance(value, list) or any(type(card) is not int or not 0 <= card < 108 for card in value):
        raise ValueError(f"{name} is not a valid card array")
    if unique and len(value) != len(set(value)):
        raise ValueError(f"{name} contains a repeated physical card")
    return list(value)


def _face_counts(cards: Iterable[int]) -> tuple[int, ...]:
    counts = [0] * 54
    for card in cards:
        if type(card) is not int or not 0 <= card < 108:
            raise ValueError("invalid card while building candidate signature")
        counts[card % 54] += 1
    return tuple(counts)


def _move_signature(move: list[list[int]], metadata: dict[str, Any]) -> str:
    """Canonical signature matching the prepared feature order."""
    if not isinstance(metadata, dict):
        raise ValueError("candidate metadata is not an object")
    kind = metadata.get("kind")
    key = metadata.get("key")
    secondary = metadata.get("secondary")
    if (not isinstance(kind, str) or type(key) is not int or type(secondary) is not int):
        raise ValueError("candidate metadata is incomplete")
    if not isinstance(move, list) or len(move) != 2:
        raise ValueError("candidate move has invalid shape")
    action = _cards(move[0], "candidate.action")
    claim = _cards(move[1], "candidate.claim")
    payload = {"action": _face_counts(action), "claim": _face_counts(claim),
               "kind": kind, "key": key, "secondary": secondary}
    # Compact, deterministic and independent of dict insertion order.
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def candidate_fingerprint(signatures: Iterable[str]) -> str:
    payload = "\n".join(signatures).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _team(record: dict[str, Any]) -> tuple[int, int]:
    features = record.get("features")
    if not isinstance(features, dict):
        raise ValueError("features is not an object")
    seat = features.get("seat")
    team = features.get("team")
    if type(seat) is not int or not 0 <= seat < 4:
        raise ValueError("features.seat is outside 0..3")
    if type(team) is not int or team != seat % 2:
        raise ValueError("features.team does not equal seat modulo two")
    return seat, team


def _label_move(record: dict[str, Any]) -> list[list[int]] | None:
    """Return one unambiguous public move, or ``None`` if it is unknown."""
    label = record.get("label")
    if not isinstance(label, dict):
        return None
    cards = _cards(label.get("cards"), "label.cards", unique=True)
    if not cards:
        return [[], []]
    candidates = label.get("claim_candidates")
    if not isinstance(candidates, list):
        return None
    claims: list[list[int]] = []
    for item in candidates:
        if not isinstance(item, dict) or "claim" not in item:
            continue
        try:
            claim = _cards(item["claim"], "label.claim_candidates.claim")
        except ValueError:
            continue
        if len(claim) != len(cards):
            continue
        claims.append(claim)
    # Never guess a wildcard claim.  Multiple byte-identical claims are safe
    # to collapse because they induce the same comparison target.
    unique_claims = {tuple(claim) for claim in claims}
    if len(unique_claims) != 1:
        return None
    return [cards, list(next(iter(unique_claims)))]


def _same_group(record: dict[str, Any]) -> tuple[str, str, str, int]:
    provenance = record.get("provenance")
    if not isinstance(provenance, dict) or not isinstance(provenance.get("data_sha256"), str):
        raise ValueError("missing provenance.data_sha256")
    match_id = record.get("match_id")
    game_id = record.get("game_id")
    deal_id = record.get("deal_id")
    if not isinstance(match_id, str) or not isinstance(game_id, str) or type(deal_id) is not int or deal_id < 0:
        raise ValueError("missing public deal identity")
    return provenance["data_sha256"], match_id, game_id, deal_id


def _read_split(source: Path, split: str) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Read one development split and no other split."""
    path = source / f"{split}.jsonl"
    if not path.is_file():
        raise TargetBuildError(f"missing development split: {path}")
    rows: list[dict[str, Any]] = []
    counters: Counter[str] = Counter()
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            counters["lines"] += 1
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise TargetBuildError(f"{path}:{line_number}: malformed JSON: {exc}") from exc
            if not isinstance(record, dict) or record.get("schema") != DECISION_SCHEMA:
                raise TargetBuildError(f"{path}:{line_number}: unexpected decision schema")
            # The ETL row may carry a final result for other audits.  Remove
            # it before any downstream helper sees the record; this sidecar's
            # label contract is public-continuation-only.
            record.pop("result", None)
            if record.get("stage") != "play":
                counters["non_play"] += 1
                continue
            try:
                _team(record)
                # The authoritative observation validator is shared with BC
                # preparation and does not access ``result``.
                observation_from_record(record)
                record_id(record)
                _same_group(record)
            except (TypeError, ValueError) as exc:
                counters["invalid_play"] += 1
                if counters["invalid_play"] <= 5:
                    counters[f"invalid_play_example_{counters['invalid_play']}"] = str(exc)
                continue
            rows.append(record)
            counters["play_rows"] += 1
    return rows, dict(counters)


def _prepare_provenance(prepared: Path) -> tuple[str, dict[str, Any]]:
    manifest_path = prepared / "manifest.json"
    if not manifest_path.is_file():
        raise TargetBuildError(f"prepared manifest is missing: {manifest_path}")
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if payload.get("schema") != "oxbot-bc-shards-v1" or payload.get("status") != "complete":
        raise TargetBuildError("prepared manifest is incomplete or incompatible")
    if payload.get("feature_version") != "oxbot-observation-v1":
        raise TargetBuildError("prepared feature contract is incompatible")
    for split in SPLITS:
        if split not in payload.get("splits", {}):
            raise TargetBuildError(f"prepared manifest has no {split} split")
    # The candidate sidecar describes the full-candidate v2 shards.  A
    # sampled train denominator cannot safely consume one label per candidate.
    if int(payload.get("train_max_negatives", 0)) < 3336:
        raise TargetBuildError("prepared train split is sampled; use full-candidate shards")
    return sha256_file(manifest_path), payload


def _prepared_alignment(prepared: Path, payload: dict[str, Any]) -> dict[str, dict[str, int]]:
    """Read the prepared sample IDs and full candidate counts.

    The continuation sidecar is keyed by ``record_id``.  The source ETL can
    contain rows which the BC preparation deliberately drops (for example a
    demonstration that is no longer legal), so provenance hashes alone do
    not prove that a target row exists in the actual shards.  Read only the
    train/validation IDs and offsets here; the held-out test split is never
    opened.  Any malformed shard, duplicate ID, or offset mismatch is a hard
    build error rather than an implicit filter.
    """
    result: dict[str, dict[str, int]] = {}
    for split in SPLITS:
        split_info = payload.get("splits", {}).get(split, {})
        shard_info = split_info.get("shards")
        if not isinstance(shard_info, list) or not shard_info:
            raise TargetBuildError(f"prepared manifest has no {split} shards")
        mapping: dict[str, int] = {}
        for item in shard_info:
            if not isinstance(item, dict) or not isinstance(item.get("path"), str):
                raise TargetBuildError(f"prepared {split} shard metadata is malformed")
            path = prepared / item["path"]
            if not path.is_file():
                raise TargetBuildError(f"prepared shard is missing: {path}")
            expected_sha = item.get("sha256")
            if isinstance(expected_sha, str) and sha256_file(path) != expected_sha:
                raise TargetBuildError(f"prepared shard SHA256 differs from manifest: {path}")
            try:
                with np.load(path, allow_pickle=False) as data:
                    if "ids" not in data.files or "offsets" not in data.files:
                        raise TargetBuildError(f"prepared shard lacks ids/offsets: {path}")
                    ids = data["ids"]
                    offsets = data["offsets"]
                    if len(offsets) != len(ids) + 1 or int(offsets[0]) != 0 or int(offsets[-1]) < 0:
                        raise TargetBuildError(f"prepared shard offsets are malformed: {path}")
                    for index, value in enumerate(ids):
                        identity = str(value)
                        if not identity or identity in mapping:
                            raise TargetBuildError(f"duplicate/empty prepared sample ID in {path}: {identity!r}")
                        begin, end = int(offsets[index]), int(offsets[index + 1])
                        if begin < 0 or end <= begin or end > int(offsets[-1]):
                            raise TargetBuildError(f"invalid candidate offsets in {path} for {identity}")
                        mapping[identity] = end - begin
            except (OSError, ValueError, TypeError) as exc:
                raise TargetBuildError(f"cannot read prepared shard {path}: {exc}") from exc
        result[split] = mapping
    return result


def _alignment_digest(mapping: dict[str, int]) -> str:
    """Stable digest of prepared IDs and candidate counts."""
    payload = "\n".join(f"{identity}\t{mapping[identity]}" for identity in sorted(mapping))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _alignment_label_summary(targets: dict[str, Any]) -> dict[str, Any]:
    """Summarize label validity for the standalone alignment audit."""
    entries = positive = negative = ignored = invalid = 0
    for target in targets.values():
        labels = target.get("labels") if isinstance(target, dict) else None
        count = target.get("candidate_count") if isinstance(target, dict) else None
        entries += 1
        if type(count) is not int or count < 1 or not isinstance(labels, list) or len(labels) != count:
            invalid += 1
            continue
        if any(type(label) is not int or label not in (-1, 0, 1) for label in labels):
            invalid += 1
            continue
        positive += sum(label > 0 for label in labels)
        negative += sum(label < 0 for label in labels)
        ignored += sum(label == 0 for label in labels)
    return {"target_entries": entries, "positive_labels": positive,
            "negative_labels": negative, "ignored_labels": ignored,
            "invalid_label_entries": invalid}


def _source_provenance(source: Path) -> tuple[str, dict[str, str]]:
    manifest_path = source / "manifest.json"
    if not manifest_path.is_file():
        raise TargetBuildError(f"source manifest is missing: {manifest_path}")
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if payload.get("schema") != "njupt-etl-manifest-v3" or payload.get("decision_schema") != DECISION_SCHEMA:
        raise TargetBuildError("source manifest schema is incompatible")
    outputs = payload.get("output_files")
    if not isinstance(outputs, dict):
        raise TargetBuildError("source manifest has no output_files metadata")
    hashes: dict[str, str] = {}
    for split in SPLITS:
        path = source / f"{split}.jsonl"
        expected = outputs.get(f"{split}.jsonl", {}).get("sha256") if isinstance(outputs.get(f"{split}.jsonl"), dict) else None
        if not path.is_file() or not isinstance(expected, str):
            raise TargetBuildError(f"source manifest does not cover {path}")
        actual = sha256_file(path)
        if actual != expected:
            raise TargetBuildError(f"source split SHA256 differs from manifest: {split}")
        hashes[split] = actual
    return sha256_file(manifest_path), hashes


def _future_support(rows: list[dict[str, Any]], index: int, probe: Probe, *, max_horizon: int) -> dict[str, Any]:
    """Audit one row's observed continuation once, independent of candidates."""
    current = rows[index]
    current_features = current["features"]
    current_seat, current_team = _team(current)
    # Find the first future public row that claims the next lead.  It is the
    # target horizon; all preceding rows must be a single legal continuation.
    future_index = None
    for position in range(index + 1, len(rows)):
        if rows[position]["features"].get("leading") is True:
            future_index = position
            break
    if future_index is None:
        return {"status": "unsupported", "reason": "no_future_lead"}
    horizon = future_index - index
    if horizon > max_horizon:
        return {"status": "unsupported", "reason": "horizon_exceeded", "horizon": horizon}
    next_seat, next_team = _team(rows[future_index])
    if next_seat < 0 or next_team not in (0, 1):
        return {"status": "unsupported", "reason": "next_lead_identity", "horizon": horizon}
    current_previous = None
    if not current_features.get("leading"):
        try:
            previous_cards = _cards(current_features.get("last_cards"), "features.last_cards", unique=True)
            previous_claim = _cards(current_features.get("last_claim"), "features.last_claim")
        except ValueError:
            return {"status": "unsupported", "reason": "current_previous_invalid", "horizon": horizon}
        if not previous_cards or len(previous_cards) != len(previous_claim):
            return {"status": "unsupported", "reason": "current_previous_missing", "horizon": horizon}
        current_previous = [previous_cards, previous_claim]

    # The first future non-pass action is the only comparison that depends on
    # the substituted candidate.  Validate the tail once; this prevents a
    # per-candidate Probe IPC loop and makes unsupported rows explicit.
    first_future: list[list[int]] | None = None
    tail_previous: list[list[int]] | None = None
    future_actor_faces: Counter[int] = Counter()
    unknown_claim = False
    for position in range(index + 1, future_index):
        row = rows[position]
        seat, _ = _team(row)
        features = row["features"]
        move = _label_move(row)
        if move is None:
            unknown_claim = True
            break
        if not move[0]:
            if features.get("leading") is True:
                return {"status": "unsupported", "reason": "leading_pass", "horizon": horizon}
            continue
        if seat == current_seat:
            future_actor_faces.update(card % 54 for card in move[0])
        if first_future is None:
            first_future = move
        else:
            assert tail_previous is not None
            checked = probe.call(command="beats", previous=tail_previous, move=move,
                                 level=current_features["level_label"])
            if "error" in checked:
                return {"status": "unsupported", "reason": "probe_beats_error", "horizon": horizon}
            if checked.get("ok") is not True:
                return {"status": "unsupported", "reason": "future_chain_invalid", "horizon": horizon}
        tail_previous = move
    if unknown_claim:
        return {"status": "unsupported", "reason": "future_claim_ambiguous", "horizon": horizon}
    return {"status": "supported", "horizon": horizon, "next_team": next_team,
            "acting_team": current_team, "current_previous": current_previous,
            "first_future": first_future, "future_actor_faces": dict(future_actor_faces),
            "current_seat": current_seat, "next_seat": next_seat}


def _candidate_labels(record: dict[str, Any], moves: list[list[list[int]]], metadata: list[dict[str, Any]],
                     support: dict[str, Any], probe: Probe) -> tuple[list[int], dict[str, Any]]:
    hand = _cards(record["features"].get("own_hand"), "features.own_hand", unique=True)
    hand_faces = Counter(card % 54 for card in hand)
    labels = [0] * len(moves)
    if support.get("status") != "supported":
        return labels, {"labeled": 0, "positive": 0, "negative": 0, "pair": False}
    current_previous = support["current_previous"]
    first_future = support["first_future"]
    future_faces = Counter({int(key): int(value) for key, value in support["future_actor_faces"].items()})
    # Do not assign the same next-lead team sign to every candidate.  That
    # would make opposite-label pairs mathematically impossible.  The pairwise
    # target is continuation compatibility: +1 preserves the observed reply,
    # -1 is legal and face-feasible but cannot admit that reply.
    for index, move in enumerate(moves):
        action = _cards(move[0], "candidate.action")
        if not action and record["features"].get("leading"):
            continue
        candidate_faces = Counter(card % 54 for card in action)
        if any(hand_faces[face] < count + future_faces[face] for face, count in candidate_faces.items()):
            # Substituted action would make the actor's observed future hand
            # impossible.  Leave it unlabeled instead of guessing a value.
            continue
        compatible = True
        if first_future is not None:
            previous = current_previous if not action else move
            if previous is None:
                continue
            checked = probe.call(command="beats", previous=previous, move=first_future,
                                 level=record["features"]["level_label"])
            if "error" in checked:
                continue
            compatible = checked.get("ok") is True
        labels[index] = 1 if compatible else -1
    positive = sum(value > 0 for value in labels)
    negative = sum(value < 0 for value in labels)
    pair = positive > 0 and negative > 0
    return labels, {"labeled": positive + negative, "positive": positive,
                    "negative": negative, "pair": pair,
                    "definition": "continuation_compatible_vs_incompatible"}


def _split_targets(rows: list[dict[str, Any]], probe: Probe, *, split: str, max_horizon: int) -> tuple[dict[str, Any], dict[str, int]]:
    groups: dict[tuple[str, str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[_same_group(row)].append(row)
    for group_rows in groups.values():
        group_rows.sort(key=lambda row: int(row["event_index"]))
    targets: dict[str, Any] = {}
    counters: Counter[str] = Counter({"groups": len(groups), "rows": len(rows)})
    for group_rows in groups.values():
        for index, record in enumerate(group_rows):
            identity = record_id(record)
            try:
                observation = observation_from_record(record)
            except (TypeError, ValueError):
                counters["invalid_observation"] += 1
                continue
            generated = probe.call(command="generate", hand=observation["hand"], level=observation["level"],
                                   leading=observation["leading"], previous=observation.get("previous"), metadata=True)
            if "error" in generated:
                raise TargetBuildError(f"{split}:{identity}: candidate generation failed: {generated['error']}")
            moves = generated.get("moves")
            metadata = generated.get("types")
            if not isinstance(moves, list) or not isinstance(metadata, list) or len(moves) != len(metadata) or not moves:
                raise TargetBuildError(f"{split}:{identity}: malformed candidate generation")
            counters["rows_with_candidates"] += 1
            signatures: list[str] = []
            for move, meta in zip(moves, metadata):
                try:
                    signatures.append(_move_signature(move, meta))
                except ValueError as exc:
                    raise TargetBuildError(f"{split}:{identity}: invalid candidate metadata: {exc}") from exc
            if len(set(signatures)) != len(signatures):
                raise TargetBuildError(f"{split}:{identity}: duplicate candidate signatures")
            support = _future_support(group_rows, index, probe, max_horizon=max_horizon)
            counters["horizon_" + str(support.get("reason", "supported"))] += 1
            if support.get("status") == "supported":
                counters["rows_with_horizon"] += 1
            labels, summary = _candidate_labels(record, moves, metadata, support, probe)
            counters["labeled_candidates"] += summary["labeled"]
            counters["positive_candidates"] += summary["positive"]
            counters["negative_candidates"] += summary["negative"]
            counters["pair_rows"] += int(summary["pair"])
            # Only rows with opposite supported labels are emitted.  Missing
            # rows are interpreted as an all-zero ignore mask by train_bc.
            if not summary["pair"]:
                continue
            targets[identity] = {
                "candidate_count": len(moves),
                "candidate_fingerprint": candidate_fingerprint(signatures),
                "labels": labels,
                "horizon": int(support["horizon"]),
                "next_lead_seat": int(support["next_seat"]),
                "acting_team": int(support["acting_team"]),
                "label_definition": "supported_public_continuation_compatibility",
            }
    return targets, dict(counters)


def run(source: Path, prepared: Path, probe_path: Path, output: Path, *, max_horizon: int = 8,
        timeout: float = 20.0, max_rows_per_split: int | None = None,
        alignment_report: Path | None = None) -> dict[str, Any]:
    if max_horizon < 1 or timeout <= 0 or (max_rows_per_split is not None and max_rows_per_split < 1):
        raise TargetBuildError("max_horizon and timeout must be positive")
    source = source.resolve()
    prepared = prepared.resolve()
    probe_path = probe_path.resolve()
    output = output.resolve()
    source_manifest_sha, source_split_sha = _source_provenance(source)
    prepared_manifest_sha, prepared_info = _prepare_provenance(prepared)
    prepared_alignment = _prepared_alignment(prepared, prepared_info)
    if prepared_info.get("source_manifest_sha256") != source_manifest_sha:
        raise TargetBuildError("prepared and source manifests refer to different ETL provenance")
    if prepared_info.get("source_splits_sha256", {}).get("train") != source_split_sha["train"] or \
            prepared_info.get("source_splits_sha256", {}).get("validation") != source_split_sha["validation"]:
        raise TargetBuildError("prepared and source development split SHA256 values differ")
    if not probe_path.is_file():
        raise TargetBuildError(f"probe is missing: {probe_path}")
    rows_by_split: dict[str, list[dict[str, Any]]] = {}
    read_counts: dict[str, dict[str, int]] = {}
    for split in SPLITS:
        rows_by_split[split], read_counts[split] = _read_split(source, split)
        if max_rows_per_split is not None:
            rows_by_split[split] = rows_by_split[split][:max_rows_per_split]
            read_counts[split]["audit_row_cap"] = max_rows_per_split
    with Probe(probe_path, timeout=timeout) as probe:
        split_targets: dict[str, dict[str, Any]] = {}
        split_summaries: dict[str, dict[str, Any]] = {}
        alignment_summaries: dict[str, dict[str, Any]] = {}
        for split in SPLITS:
            targets, summary = _split_targets(rows_by_split[split], probe, split=split, max_horizon=max_horizon)
            prepared_ids = set(prepared_alignment[split])
            raw_ids = set(targets)
            extra_ids = sorted(raw_ids - prepared_ids)
            common_ids = raw_ids & prepared_ids
            count_mismatches = [
                {"id": identity, "target": int(targets[identity]["candidate_count"]),
                 "prepared": int(prepared_alignment[split][identity])}
                for identity in sorted(common_ids)
                if targets[identity].get("candidate_count") != prepared_alignment[split][identity]
            ]
            if count_mismatches:
                sample = count_mismatches[:5]
                raise TargetBuildError(f"{split}: candidate count mismatch against prepared shards: {sample}")
            # A target is meaningful only for a row that survived BC
            # preparation.  Keep sparse pair rows (unsupported rows are
            # intentionally omitted), but never leave a source-only ID in a
            # trainable sidecar.
            split_targets[split] = {identity: targets[identity] for identity in sorted(common_ids)}
            summary["read"] = read_counts[split]
            summary["target_count_before_alignment"] = len(targets)
            summary["target_count"] = len(split_targets[split])
            summary["prepared_record_count"] = len(prepared_ids)
            summary["target_extra_count"] = len(extra_ids)
            summary["target_extra_ids"] = extra_ids
            summary["target_missing_prepared_count"] = len(prepared_ids - set(split_targets[split]))
            summary["candidate_count_mismatch_count"] = len(count_mismatches)
            split_summaries[split] = summary
            alignment_summaries[split] = {
                "prepared_record_count": len(prepared_ids),
                "target_count_before_alignment": len(targets),
                "target_count_after_alignment": len(split_targets[split]),
                "common_count": len(common_ids),
                "extra_count": len(extra_ids),
                "extra_ids": extra_ids,
                "missing_prepared_count": len(prepared_ids - set(split_targets[split])),
                "candidate_count_mismatch_count": len(count_mismatches),
                "candidate_count_mismatches": count_mismatches,
                "prepared_alignment_sha256": _alignment_digest(prepared_alignment[split]),
                **_alignment_label_summary(split_targets[split]),
            }
    train_pairs = split_summaries["train"].get("pair_rows", 0)
    validation_pairs = split_summaries["validation"].get("pair_rows", 0)
    payload = {
        "schema": SCHEMA,
        "status": "complete",
        "trainable": bool(train_pairs and validation_pairs and max_rows_per_split is None),
        "source": str(source),
        "prepared": str(prepared),
        "probe": str(probe_path),
        "source_manifest_sha256": source_manifest_sha,
        "source_split_sha256": source_split_sha,
        "prepared_manifest_sha256": prepared_manifest_sha,
        "probe_sha256": sha256_file(probe_path),
        "splits_read": list(SPLITS),
        "test_used": False,
        "max_horizon": max_horizon,
        "max_rows_per_split": max_rows_per_split,
        "audit_only": max_rows_per_split is not None,
        "target_definition": {
            "candidate_substitution": "replace current action; no hidden hand or final result",
            "support": "observed non-pass continuation remains legal until next leading=true row and actor face counts remain feasible",
            "labels": "+1 for continuation-compatible candidate, -1 for legal face-feasible candidate that cannot admit the observed first reply",
            "unsupported": "pass/terminal/ambiguous claim/long horizon/illegal continuation or unknown feasibility receives ignore label 0",
            "interpretation": "observational continuation compatibility, not a team credit, causal counterfactual, or Q value",
        },
        "split_summaries": split_summaries,
        "prepared_alignment": {
            split: {"record_count": len(prepared_alignment[split]),
                    "sha256": _alignment_digest(prepared_alignment[split])}
            for split in SPLITS
        },
        "target_coverage": "paired_rows_only",
        "target_count": sum(len(items) for items in split_targets.values()),
        "targets": split_targets,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n", encoding="utf-8")
    temporary.replace(output)
    if alignment_report is None:
        alignment_report = output.parent / "candidate_continuation_alignment.json"
    audit = {
        "schema": "oxbot-candidate-continuation-alignment-v1",
        "status": "complete",
        "sidecar": str(output),
        "prepared": str(prepared),
        "prepared_manifest_sha256": prepared_manifest_sha,
        "source_manifest_sha256": source_manifest_sha,
        "test_used": False,
        "splits": alignment_summaries,
    }
    alignment_report.parent.mkdir(parents=True, exist_ok=True)
    alignment_temporary = alignment_report.with_suffix(alignment_report.suffix + ".tmp")
    alignment_temporary.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    alignment_temporary.replace(alignment_report)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--prepared", type=Path, default=DEFAULT_PREPARED)
    parser.add_argument("--probe", type=Path, default=DEFAULT_PROBE)
    parser.add_argument("--output", type=Path, default=ROOT / "reports/candidate_continuation_targets_v1.json")
    parser.add_argument("--max-horizon", type=int, default=8)
    parser.add_argument("--timeout", type=float, default=20.0)
    parser.add_argument("--max-rows-per-split", type=int, default=None,
                        help="bounded audit sample; any capped run is audit_only and not trainable")
    parser.add_argument("--alignment-report", type=Path, default=None,
                        help="standalone prepared-shard alignment audit (default: next to output)")
    args = parser.parse_args()
    payload = run(args.source, args.prepared, args.probe, args.output,
                  max_horizon=args.max_horizon, timeout=args.timeout,
                  max_rows_per_split=args.max_rows_per_split,
                  alignment_report=args.alignment_report)
    print(json.dumps({"status": payload["status"], "trainable": payload["trainable"],
                      "target_count": payload["target_count"], "test_used": payload["test_used"],
                      "split_summaries": payload["split_summaries"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
