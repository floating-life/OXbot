"""Audited real-replay BC for the competition FableDan network.

``prepare`` adapts the read-only NJUPT decision collection to the exact
competition encoder. ``train`` opens train/validation shards only, visits every
accepted train row each epoch, and selects checkpoints by validation NLL.
``evaluate`` is the separate, explicit held-out evaluation entry point.

Opponent allocations and results are never features or training targets. Full
public prefixes are joined by game/deal/event, so neither the current action nor
future events enter the sequence. Missing wildcard claims remain unknown.
"""
from __future__ import annotations

import argparse
from bisect import bisect_left
from collections import Counter, defaultdict
from contextlib import nullcontext
import hashlib
import json
import math
from pathlib import Path
import random
import sys
import time
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from fabledan.cards import is_wildcard, level_rank, rank_of
from fabledan.combos import PASS, PASS_MOVE, classify_claim, gen_moves
from fabledan.encode import (BOS_TOK, FEAT_DIM, LEVEL_BASE, MAX_SEQ, PLAYER_BASE,
                             RANK_BASE, RETURN_TOK, TRIBUTE_TOK, TYPE_BASE,
                             VOCAB, encode_decision)

SCHEMA = "oxbot-fabledan-real-shards-v2"
DECISION_SCHEMA = "njupt-decision-v3"
UNKNOWN_CLAIM_TOK = 47  # reserved in the competition vocabulary
DEVELOPMENT_SPLITS = ("train", "validation")
ENCODER_FILES = ("cards.py", "combos.py", "encode.py")


class DataError(ValueError):
    pass


class SkipRecord(DataError):
    def __init__(self, reason: str, detail: str = ""):
        super().__init__(detail or reason)
        self.reason = reason


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def encoder_hashes() -> dict[str, str]:
    return {name: sha256_file(HERE / "fabledan" / name) for name in ENCODER_FILES}


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def cards(value: Any, field: str, *, unique: bool = True) -> list[int]:
    if not isinstance(value, list) or any(type(c) is not int or not 0 <= c < 108 for c in value):
        raise SkipRecord("invalid_observation", f"{field} must be Botzone cards in 0..107")
    if unique and len(value) != len(set(value)):
        raise SkipRecord("invalid_observation", f"{field} repeats physical cards")
    return list(value)


def local_cards(faces: list[int]) -> list[int]:
    """Public duplicate faces get local IDs, independent of hidden allocations."""
    used: Counter[int] = Counter()
    result = []
    for face in faces:
        if type(face) is not int or not 0 <= face < 54 or used[face] >= 2:
            raise DataError("public action has invalid/repeated faces")
        result.append(face + 54 * used[face])
        used[face] += 1
    return result


def identity(record: dict[str, Any]) -> str:
    provenance = record.get("provenance")
    if not isinstance(provenance, dict):
        raise DataError("decision has no provenance")
    digest = provenance.get("data_sha256")
    deal, event = record.get("deal_id"), record.get("event_index")
    if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
        raise DataError("decision has no valid source SHA256")
    if type(deal) is not int or type(event) is not int or deal < 0 or event < 0:
        raise DataError("decision has invalid deal/event indices")
    return f"{digest[:16]}:d{deal}:e{event}"


def full_action_moves(action: list[int], level: int) -> list[Any]:
    """All generator-supported claims using every demonstrated physical face."""
    if not action:
        return [PASS_MOVE]
    target = Counter(c % 54 for c in action)
    return [move for move in gen_moves(action, level)
            if move.size == len(action) and Counter(c % 54 for c in move.cards) == target]


def semantic_key(move: Any) -> tuple[Any, ...]:
    """The canonical action/claim semantics, independent of card realization."""
    return move.type, move.key, move.size, tuple(sorted(move.claim_ranks))


def observation_from_record(record: dict[str, Any]) -> tuple[dict[str, Any], dict[str, int]]:
    """Whitelist the acting information set. Never copy the source record."""
    feature = record.get("features")
    if not isinstance(feature, dict):
        raise SkipRecord("invalid_observation", "features is not an object")
    hand = cards(feature.get("own_hand"), "own_hand")
    player = feature.get("seat")
    if type(player) is not int or not 0 <= player < 4 or not 1 <= len(hand) <= 27:
        raise SkipRecord("invalid_observation", "invalid acting seat/hand size")
    try:
        level = level_rank(feature.get("level_label"))
    except ValueError as exc:
        raise SkipRecord("invalid_observation", str(exc)) from exc
    left = feature.get("remaining_counts")
    if not isinstance(left, list) or len(left) != 4 or any(type(v) is not int or not 0 <= v <= 27 for v in left):
        raise SkipRecord("invalid_observation", "invalid public remaining counts")
    if left[player] != len(hand):
        raise SkipRecord("invalid_observation", "acting hand disagrees with public remaining count")
    leading = feature.get("leading")
    if type(leading) is not bool:
        raise SkipRecord("invalid_observation", "leading is not boolean")
    lead = None
    stats = {"unknown_previous_semantics_recovered": 0}
    if not leading:
        previous = cards(feature.get("last_cards"), "last_cards")
        if not previous:
            raise SkipRecord("invalid_previous", "following without a public lead")
        claim = feature.get("last_claim")
        if claim is not None:
            claim = cards(claim, "last_claim", unique=False)
            if len(claim) != len(previous):
                raise SkipRecord("invalid_previous", "lead card/claim length mismatch")
            try:
                lead = classify_claim(previous, claim, level)
            except ValueError as exc:
                raise SkipRecord("invalid_previous", str(exc)) from exc
        else:
            variants = full_action_moves(previous, level)
            semantics = {(move.type, move.key, move.size) for move in variants}
            if len(semantics) != 1:
                raise SkipRecord("uncertain_previous", "unobserved wildcard claim has multiple lead type/key/size values")
            # Only type/key/size are used for legality and lead features. This
            # recovers those public invariants, not an asserted wildcard claim.
            lead = variants[0]
            stats["unknown_previous_semantics_recovered"] = 1
        if lead.type == PASS:
            raise SkipRecord("invalid_previous", "public lead cannot be pass")
    return {"hand": hand, "player": player, "level": level, "left": list(left),
            "done": [v == 0 for v in left], "lead": lead, "events": []}, stats


class PublicHistoryIndex:
    """Store only whitelisted public events for explicitly selected games.

    The indexed event table supplies exact raw event indices. PUBLIC_EVENTS is
    cross-checked for complete P/T/B coverage, rather than relying on a bounded
    history snapshot. The result/private fields are never accessed.
    """
    def __init__(self, path: Path, selected_games: dict[str, dict[str, Any]]):
        self.events: dict[tuple[str, int], list[dict[str, Any]]] = {}
        self._cache: dict[tuple[str, int, int, int], tuple[list[int], list[int], list[int], list[int]]] = {}
        seen_games = set()
        with path.open("r", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                source = json.loads(line)
                game_id = source.get("game_id")
                if game_id not in selected_games:
                    continue
                expected = selected_games[game_id]
                if game_id in seen_games or source.get("match_id") != expected["match_id"]:
                    raise DataError(f"games:{line_number}: duplicate game or match mismatch")
                seen_games.add(game_id)
                public_rows = source.get("events")
                if not isinstance(public_rows, list):
                    raise DataError(f"games:{line_number}: no public event table")
                by_deal: dict[int, list[dict[str, Any]]] = defaultdict(list)
                compact = None
                for raw in public_rows:
                    if not isinstance(raw, dict):
                        raise DataError("malformed public event")
                    tag = raw.get("tag")
                    if tag == "PUBLIC_EVENTS":
                        if compact is not None:
                            raise DataError("duplicate PUBLIC_EVENTS")
                        compact = raw.get("events_by_deal")
                        continue
                    if tag not in ("R", "P", "T", "B", "C"):
                        continue
                    deal, event = raw.get("deal_id"), raw.get("event_index")
                    if type(deal) is not int or type(event) is not int or deal < 0 or event < 0:
                        raise DataError("invalid indexed public event")
                    item = {"tag": tag, "index": event}
                    if tag == "P":
                        item["player"] = raw.get("seat")
                        item["cards"] = local_cards(raw.get("face_cards", []))
                    elif tag in ("T", "B"):
                        item["player"] = raw.get("from")
                        item["recipient"] = raw.get("to")
                        item["cards"] = local_cards(raw.get("face_cards", []))
                        if len(item["cards"]) != 1:
                            raise DataError("public transfer does not contain one card")
                    elif tag == "R":
                        item["source_level"] = raw.get("current_level")
                    if tag in ("P", "T", "B") and (type(item["player"]) is not int or not 0 <= item["player"] < 4):
                        raise DataError("public event has invalid player")
                    by_deal[deal].append(item)
                if not isinstance(compact, dict):
                    raise DataError("full PUBLIC_EVENTS is required for history coverage")
                for deal, events in by_deal.items():
                    indices = [event["index"] for event in events]
                    if indices != sorted(set(indices)):
                        raise DataError("public event indices are not strictly increasing")
                    compact_rows = compact.get(str(deal))
                    if not isinstance(compact_rows, list):
                        raise DataError("PUBLIC_EVENTS lacks a deal")
                    expected_count = Counter(row[0] for row in compact_rows if isinstance(row, list) and row)
                    actual_count = Counter(event["tag"] for event in events)
                    if any(expected_count[tag] != actual_count[tag] for tag in ("R", "P", "T", "B", "C")):
                        raise DataError("indexed public history and PUBLIC_EVENTS disagree")
                    compact_actions = [row for row in compact_rows if row and row[0] in ("P", "T", "B", "C")]
                    indexed_actions = []
                    for item in events:
                        tag = item["tag"]
                        if tag == "P":
                            indexed_actions.append([tag, item["player"], [card % 54 for card in item["cards"]], not item["cards"]])
                        elif tag in ("T", "B"):
                            indexed_actions.append([tag, item["player"], item["recipient"], [card % 54 for card in item["cards"]]])
                        elif tag == "C":
                            indexed_actions.append([tag])
                    if indexed_actions != compact_actions:
                        raise DataError("indexed public action sequence and PUBLIC_EVENTS disagree")
                    self.events[(game_id, deal)] = events
        if seen_games != set(selected_games):
            raise DataError(f"public games missing {len(set(selected_games) - seen_games)} selected games")

    def tokens_before(self, game: str, deal: int, event: int, viewer: int, level: int) -> tuple[list[int], dict[str, int]]:
        key = (game, deal, viewer, level)
        if key not in self._cache:
            rows = self.events.get((game, deal))
            if rows is None:
                raise DataError("decision has no full public history")
            tokens = [BOS_TOK, LEVEL_BASE + level]
            indices, ends, unknown_ends = [], [], []
            unknown_count = 0
            for row in rows:
                indices.append(row["index"])
                ends.append(len(tokens))  # prefix excludes the indexed event
                unknown_ends.append(unknown_count)
                tag = row["tag"]
                if tag in ("T", "B"):
                    tokens.extend([PLAYER_BASE + (row["player"] - viewer) % 4,
                                   TRIBUTE_TOK if tag == "T" else RETURN_TOK,
                                   RANK_BASE + rank_of(row["cards"][0])])
                elif tag == "P":
                    action = row["cards"]
                    tokens.append(PLAYER_BASE + (row["player"] - viewer) % 4)
                    if not action:
                        tokens.append(TYPE_BASE + PASS)
                    elif any(is_wildcard(card, level) for card in action):
                        tokens.append(UNKNOWN_CLAIM_TOK)
                        tokens.extend(RANK_BASE + rank for rank in sorted(rank_of(card) for card in action))
                        unknown_count += 1
                    else:
                        try:
                            move = classify_claim(action, action, level)
                        except ValueError:
                            tokens.append(UNKNOWN_CLAIM_TOK)
                            tokens.extend(RANK_BASE + rank for rank in sorted(rank_of(card) for card in action))
                            unknown_count += 1
                        else:
                            tokens.append(TYPE_BASE + move.type)
                            tokens.extend(RANK_BASE + rank for rank in sorted(move.claim_ranks))
                # R is represented by BOS/LEVEL; C changes the rule state but
                # emits no token in competition/fabledan/encode.py.
            self._cache[key] = (indices, ends, tokens, unknown_ends)
        indices, ends, tokens, unknown_ends = self._cache[key]
        position = bisect_left(indices, event)
        if position == len(indices) or indices[position] != event:
            raise DataError("decision event does not occur in public history")
        prefix = tokens[:ends[position]]
        raw_length = len(prefix)
        if raw_length > MAX_SEQ:
            prefix = prefix[:2] + prefix[-(MAX_SEQ - 2):]
        return prefix, {"full_history_window_truncated": int(raw_length > MAX_SEQ),
                        "unknown_claim_history": int(unknown_ends[position] > 0)}


def encode_record(record: dict[str, Any], history: PublicHistoryIndex) -> tuple[dict[str, Any], dict[str, int]]:
    if record.get("schema") != DECISION_SCHEMA:
        raise DataError("incompatible decision schema")
    if record.get("stage") != "play":
        raise SkipRecord("rule_handled_exchange")
    sample_id = identity(record)
    obs, stats = observation_from_record(record)
    legal = gen_moves(obs["hand"], obs["level"], obs["lead"])
    if not legal:
        raise SkipRecord("no_legal_candidates")
    obs["legal"] = legal
    _, features = encode_decision(obs)
    label = record.get("label")
    if not isinstance(label, dict):
        raise SkipRecord("invalid_label")
    demonstration = cards(label.get("cards"), "label.cards")
    if Counter(c % 54 for c in demonstration) - Counter(c % 54 for c in obs["hand"]):
        raise SkipRecord("invalid_label", "demonstrated faces are outside the acting hand")
    variants = full_action_moves(demonstration, obs["level"])
    if obs["lead"] is not None:
        from fabledan.combos import beats
        variants = [move for move in variants if move.type == PASS or beats(move, obs["lead"], obs["level"])]
    elif not demonstration:
        variants = []
    if not variants:
        raise SkipRecord("demonstration_not_legal", "no legal claim using the entire demonstrated face multiset")
    positive_semantics = {semantic_key(move) for move in variants}
    positive_keys = {row.tobytes() for move, row in zip(legal, features)
                     if semantic_key(move) in positive_semantics}
    # The live generator chooses canonical deck/suit representatives. The
    # generator also chooses natural cards before optional wildcard substitutes.
    # Map full-demo claims to the canonical type/key/size/claim-rank semantics;
    # card realization and wildcard consumption are not guessed source labels.
    unique: dict[bytes, int] = {}
    kept = []
    for index, row in enumerate(features):
        key = row.tobytes()
        if key not in unique:
            unique[key] = len(kept)
            kept.append(index)
    features = features[kept]
    positive = np.asarray([row.tobytes() in positive_keys for row in features], dtype=np.bool_)
    if not positive.any():
        raise SkipRecord("demonstration_not_in_policy", "legal demonstrated semantics absent from the live generator")
    tokens, history_stats = history.tokens_before(record.get("game_id"), record.get("deal_id"),
                                                  record.get("event_index"), obs["player"], obs["level"])
    if features.shape[1:] != (FEAT_DIM,) or not np.isfinite(features).all():
        raise DataError("competition feature contract violation")
    stats.update(history_stats)
    stats["canonical_candidates_collapsed"] = len(legal) - len(kept)
    stats["ambiguous_positive_claims"] = int(positive.sum() > 1)
    return {"id": sample_id, "tokens": np.asarray(tokens, dtype=np.uint8),
            "actions": features, "positives": positive, "bc_supported": True}, stats


def public_ntp_record(record: dict[str, Any], history: PublicHistoryIndex) -> tuple[dict[str, Any], dict[str, int]]:
    """Retain an uncertain/incompatible P row for public NTP supervision.

    A single dummy candidate has no positive label and never asserts a legal
    move. The explicit mask removes it from BC/accuracy denominators.
    Its public prefix still trains the same transformer with the standard NTP
    objective. No current or future action is used to guess a missing lead.
    """
    feature = record.get("features")
    if not isinstance(feature, dict) or type(feature.get("seat")) is not int or not 0 <= feature["seat"] < 4:
        raise DataError("NTP-only record lacks valid public observation")
    try:
        level = level_rank(feature.get("level_label"))
    except ValueError as error:
        raise DataError("NTP-only record lacks valid level") from error
    tokens, stats = history.tokens_before(record.get("game_id"), record.get("deal_id"),
                                         record.get("event_index"), feature["seat"], level)
    return {"id": identity(record), "tokens": np.asarray(tokens, dtype=np.uint8),
            "actions": np.zeros((1, FEAT_DIM), dtype=np.float32),
            "positives": np.zeros(1, dtype=np.bool_), "bc_supported": False}, stats


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
        target = self.output / self.split / f"shard-{len(self.files):05d}.npz"
        target.parent.mkdir(parents=True, exist_ok=True)
        offsets = np.concatenate(([0], np.cumsum([len(row["actions"]) for row in self.pending], dtype=np.int64)))
        tokens = np.zeros((len(self.pending), MAX_SEQ), dtype=np.uint8)
        lengths = np.asarray([len(row["tokens"]) for row in self.pending], dtype=np.int16)
        for index, row in enumerate(self.pending):
            tokens[index, :lengths[index]] = row["tokens"]
        actions = np.concatenate([row["actions"] for row in self.pending]).astype(np.float32)
        positives = np.concatenate([row["positives"] for row in self.pending])
        bc_supported = np.asarray([row.get("bc_supported", True) for row in self.pending], dtype=np.bool_)
        ids = np.asarray([row["id"] for row in self.pending], dtype="U80")
        with target.open("xb") as stream:
            np.savez_compressed(stream, tokens=tokens, lengths=lengths, actions=actions,
                                offsets=offsets, positives=positives, ids=ids, bc_supported=bc_supported)
        self.files.append({"path": target.relative_to(self.output).as_posix(), "records": len(ids),
                           "candidates": int(offsets[-1]), "sha256": sha256_file(target)})
        self.pending.clear()


def verify_source(source: Path, splits: tuple[str, ...]) -> tuple[dict[str, Any], dict[str, str], dict[str, dict[str, Any]]]:
    manifest_path = source / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("decision_schema") != DECISION_SCHEMA:
        raise DataError("audited decision-v3 source manifest is required")
    source_hashes = {}
    for name in [*(f"{split}.jsonl" for split in splits), "games.jsonl"]:
        expected = manifest.get("output_files", {}).get(name, {}).get("sha256")
        actual = sha256_file(source / name)
        if actual != expected:
            raise DataError(f"source SHA256 differs from manifest: {name}")
        source_hashes[name] = actual
    # Verify source match/package boundaries before reading selected examples.
    match_splits: dict[str, str] = {}
    archive_splits: dict[str, str] = {}
    selected_games = {}
    for item in manifest.get("source_files", []):
        split, match = item.get("split"), item.get("match_id")
        if split not in ("train", "validation", "test") or not isinstance(match, str):
            raise DataError("source manifest has invalid match split")
        if match in match_splits and match_splits[match] != split:
            raise DataError("source match crosses train/validation/test boundary")
        match_splits[match] = split
        archive = item.get("archive_sha256")
        if archive:
            if archive in archive_splits and archive_splits[archive] != split:
                raise DataError("source archive crosses train/validation/test boundary")
            archive_splits[archive] = split
        if split in splits:
            game = item.get("game_id")
            if not isinstance(game, str) or game in selected_games:
                raise DataError("source manifest contains duplicate/invalid game IDs")
            selected_games[game] = item
    if not selected_games:
        raise DataError("source manifest has no selected games")
    return manifest, source_hashes, selected_games


def prepare(source: Path, output: Path, *, splits: tuple[str, ...] = DEVELOPMENT_SPLITS,
            shard_size: int = 512) -> dict[str, Any]:
    source, output = source.resolve(), output.resolve()
    if any(split not in ("train", "validation", "test") for split in splits) or len(set(splits)) != len(splits):
        raise DataError("invalid preparation splits")
    if shard_size < 1:
        raise DataError("shard size must be positive")
    if output == source or output.is_relative_to(source) or source.is_relative_to(output):
        raise DataError("prepared output must be separate from the read-only source")
    if output.exists() and any(output.iterdir()):
        raise DataError("prepared output is not empty; refusing overwrite")
    manifest, source_hashes, selected_games = verify_source(source, splits)
    initial_source_sha = sha256_file(source / "manifest.json")
    initial_encoder_hashes = encoder_hashes()
    history = PublicHistoryIndex(source / "games.jsonl", selected_games)
    output.mkdir(parents=True, exist_ok=True)
    payload = {"schema": SCHEMA, "status": "preparing", "feature_dim": FEAT_DIM, "vocab": VOCAB,
               "max_seq": MAX_SEQ, "source_manifest_sha256": initial_source_sha,
               "source_splits_sha256": {split: source_hashes[f"{split}.jsonl"] for split in splits},
               "public_games_sha256": source_hashes["games.jsonl"], "encoder_sha256": initial_encoder_hashes,
               "splits_read": list(splits), "test_used": "test" in splits, "splits": {},
               "features": "acting hand + strict public event prefix; no opponent allocations/results",
               "label_policy": "all full-demo face-matching legal claims mapped to live generator type/key/size/sorted-claim-ranks semantics; canonical realization independent of source suit/wildcard choice; collapse encoder-identical candidates",
               "unknown_claim_policy": "reserved token 47 + printed public ranks; no invented history claim",
               "history_policy": "full indexed game/deal prefix strictly before event_index; BOS/LEVEL + last 510 tokens",
               "exchange_policy": "all tribute/return records accounted as rule_handled_exchange; live bot uses forced exchange rules",
               "uncertain_follow_policy": "retain every P as public-NTP-only when prior claim semantics are ambiguous or demonstrated follow is incompatible; BC mask0, explicit reason counts"}
    write_json(output / "manifest.json", payload)
    all_ids = set()
    start = time.monotonic()
    for split in splits:
        writer = ShardWriter(output, split, shard_size)
        counts: Counter[str] = Counter()
        examples: dict[str, list[Any]] = defaultdict(list)
        ids_digest = hashlib.sha256()
        candidates = []
        with (source / f"{split}.jsonl").open("r", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                record = json.loads(line)
                if not isinstance(record, dict) or record.get("schema") != DECISION_SCHEMA:
                    raise DataError(f"{split}:{line_number}: incompatible decision")
                game = selected_games.get(record.get("game_id"))
                if game is None or game["split"] != split or game["match_id"] != record.get("match_id"):
                    raise DataError(f"{split}:{line_number}: decision does not belong to source split")
                sample_id = identity(record)
                if sample_id in all_ids:
                    raise DataError(f"duplicate/cross-split sample ID: {sample_id}")
                all_ids.add(sample_id)
                if record["provenance"]["data_sha256"] != game["data_sha256"]:
                    raise DataError("decision provenance differs from source game")
                counts["input_records"] += 1
                stage = record.get("stage")
                if stage in ("tribute", "return"):
                    counts["rule_handled_exchange"] += 1
                    counts[f"rule_handled_{stage}"] += 1
                    continue
                if stage != "play":
                    raise DataError(f"{split}:{line_number}: unknown decision stage")
                counts["play_records"] += 1
                try:
                    encoded, stats = encode_record(record, history)
                except SkipRecord as error:
                    if error.reason not in ("uncertain_previous", "demonstration_not_legal", "demonstration_not_in_policy"):
                        raise DataError(f"{split}:{line_number}: irrecoverable observation: {error}") from error
                    encoded, stats = public_ntp_record(record, history)
                    counts["public_ntp_only_records"] += 1
                    counts[f"ntp_only_{error.reason}"] += 1
                    if len(examples[error.reason]) < 20:
                        examples[error.reason].append({"line": line_number, "id": sample_id, "detail": str(error)})
                else:
                    counts["bc_supported_records"] += 1
                writer.add(encoded)
                ids_digest.update((sample_id + "\n").encode())
                candidates.append(len(encoded["actions"]))
                counts["accepted_records"] += 1
                for name, value in stats.items():
                    counts[name] += value
                if counts["input_records"] % 10000 == 0:
                    print(json.dumps({"split": split, "input": counts["input_records"],
                                      "accepted": counts["accepted_records"], "seconds": round(time.monotonic() - start, 1)}), flush=True)
        writer.flush()
        expected = manifest.get("decision_split_counts", {}).get(split)
        if counts["input_records"] != expected:
            raise DataError(f"{split}: all-record coverage differs from source manifest")
        if counts["input_records"] != counts["accepted_records"] + counts["filtered_records"] + counts["rule_handled_exchange"]:
            raise DataError(f"{split}: uncovered decisions")
        if counts["play_records"] != counts["accepted_records"] or counts["accepted_records"] != counts["bc_supported_records"] + counts["public_ntp_only_records"]:
            raise DataError(f"{split}: incomplete P-record training coverage")
        if not counts["accepted_records"]:
            raise DataError(f"{split}: no accepted rows")
        payload["splits"][split] = {"counts": dict(counts), "filters": dict(examples), "files": writer.files,
                                     "accepted_ids_sha256": ids_digest.hexdigest(), "match_ids": sorted({item["match_id"] for item in selected_games.values() if item["split"] == split}),
                                     "candidate_count": {"sum": sum(candidates), "max": max(candidates), "mean": float(np.mean(candidates))}}
        write_json(output / "manifest.json", payload)
    if sha256_file(source / "manifest.json") != initial_source_sha or encoder_hashes() != initial_encoder_hashes:
        raise DataError("source manifest or competition encoder changed during preparation")
    if any(sha256_file(source / name) != digest for name, digest in source_hashes.items()):
        raise DataError("source changed during preparation")
    payload["status"] = "complete"
    payload["preparation_seconds"] = time.monotonic() - start
    write_json(output / "manifest.json", payload)
    return payload


def load_manifest(data: Path, splits: tuple[str, ...]) -> tuple[dict[str, Any], dict[str, list[Path]]]:
    manifest = json.loads((data / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema") != SCHEMA or manifest.get("status") != "complete":
        raise DataError("complete competition real-data manifest is required")
    if (manifest.get("feature_dim"), manifest.get("vocab"), manifest.get("max_seq")) != (FEAT_DIM, VOCAB, MAX_SEQ):
        raise DataError("prepared feature contract is incompatible")
    if manifest.get("encoder_sha256") != encoder_hashes():
        raise DataError("competition encoder differs from prepared-data provenance")
    paths = {}
    split_matches: dict[str, set[str]] = {}
    for split in splits:
        entry = manifest.get("splits", {}).get(split)
        if not isinstance(entry, dict) or not entry.get("files"):
            raise DataError(f"prepared manifest lacks {split}")
        files = []
        for info in entry["files"]:
            path = (data / info["path"]).resolve()
            if not path.is_relative_to(data.resolve()) or path.parent != (data / split).resolve():
                raise DataError("prepared shard path escapes its split")
            if sha256_file(path) != info["sha256"]:
                raise DataError(f"prepared shard SHA256 differs: {path}")
            files.append(path)
        if set(files) != set((data / split).resolve().glob("shard-*.npz")):
            raise DataError(f"prepared {split} shard set differs from manifest")
        paths[split] = files
        split_matches[split] = set(entry.get("match_ids", []))
    if "train" in split_matches and "validation" in split_matches and split_matches["train"] & split_matches["validation"]:
        raise DataError("prepared match crosses development split boundary")
    return manifest, paths


def load_shard(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in ("tokens", "lengths", "actions", "offsets", "positives", "ids", "bc_supported")}
    count = len(arrays["ids"])
    offsets, lengths = arrays["offsets"], arrays["lengths"]
    if count < 1 or arrays["tokens"].shape != (count, MAX_SEQ) or lengths.shape != (count,) or arrays["bc_supported"].shape != (count,):
        raise DataError("prepared token shape is invalid")
    if offsets.shape != (count + 1,) or offsets[0] != 0 or np.any(np.diff(offsets) < 1):
        raise DataError("prepared candidate offsets are invalid")
    if arrays["actions"].shape != (int(offsets[-1]), FEAT_DIM) or arrays["positives"].shape != (int(offsets[-1]),):
        raise DataError("prepared action shapes are invalid")
    if not np.isfinite(arrays["actions"]).all() or np.any(lengths < 2) or np.any(lengths > MAX_SEQ):
        raise DataError("prepared features/lengths are invalid")
    if np.any(arrays["tokens"] >= VOCAB) or any(bool(arrays["positives"][offsets[i]:offsets[i + 1]].any()) != bool(arrays["bc_supported"][i]) for i in range(count)):
        raise DataError("prepared labels/tokens are invalid")
    if len(set(map(str, arrays["ids"]))) != count:
        raise DataError("duplicate IDs inside prepared shard")
    return arrays


def batches(files: list[Path], *, batch_size: int, candidate_budget: int, seed: int | None):
    """Every record exactly once; no replacement and no dropped tail batch."""
    rng = np.random.default_rng(seed)
    order = list(files)
    if seed is not None:
        rng.shuffle(order)
    for path in order:
        data = load_shard(path)
        indices = np.arange(len(data["ids"]))
        if seed is not None:
            rng.shuffle(indices)
        chosen: list[int] = []
        total_candidates = 0
        for raw_index in indices:
            index = int(raw_index)
            count = int(data["offsets"][index + 1] - data["offsets"][index])
            if chosen and (len(chosen) >= batch_size or total_candidates + count > candidate_budget):
                yield pack_batch(data, chosen)
                chosen, total_candidates = [], 0
            chosen.append(index)
            total_candidates += count
        if chosen:
            yield pack_batch(data, chosen)


def pack_batch(data: dict[str, np.ndarray], indices: list[int]) -> dict[str, np.ndarray]:
    lengths = data["lengths"][indices].astype(np.int64)
    ranges = [(int(data["offsets"][i]), int(data["offsets"][i + 1])) for i in indices]
    counts = np.asarray([end - begin for begin, end in ranges], dtype=np.int64)
    return {"tokens": data["tokens"][indices, :int(lengths.max())].astype(np.int64), "lengths": lengths,
            "actions": np.concatenate([data["actions"][begin:end] for begin, end in ranges]),
            "positives": np.concatenate([data["positives"][begin:end] for begin, end in ranges]),
            "counts": counts, "ids": data["ids"][indices], "bc_supported": data["bc_supported"][indices]}


def marginal_loss(scores, positives, counts, bc_supported=None):
    """Exact multi-positive likelihood over the complete per-record candidates."""
    import torch
    values = []
    start = 0
    for index, count in enumerate(counts):
        end = start + int(count)
        current = scores[start:end].float()
        positive = positives[start:end]
        if not bool(positive.any()):
            if bc_supported is None or bool(bc_supported[index]):
                raise DataError("BC-supported record has no positive candidate")
            values.append(current.sum() * 0.0)
        else:
            if bc_supported is not None and not bool(bc_supported[index]):
                raise DataError("NTP-only record unexpectedly has positive BC labels")
            values.append(torch.logsumexp(current, 0) - torch.logsumexp(current[positive], 0))
        start = end
    if start != scores.numel():
        raise DataError("batch candidate counts do not cover scores")
    return torch.stack(values)


def score_batch(model, batch, device: str, candidate_chunk: int):
    import torch
    tokens = torch.from_numpy(batch["tokens"]).to(device)
    lengths = torch.from_numpy(batch["lengths"]).to(device)
    ctx, hidden = model.encode_seq(tokens, lengths)
    owners = torch.repeat_interleave(torch.arange(len(batch["counts"]), device=device),
                                    torch.from_numpy(batch["counts"]).to(device))
    action = torch.from_numpy(batch["actions"]).to(device)
    scores = []
    for start in range(0, len(action), candidate_chunk):
        end = start + candidate_chunk
        embedded = model.hand_mlp(action[start:end])
        scores.append(model.q_head(torch.cat((ctx[owners[start:end]], embedded), dim=-1))[:, 0])
    return torch.cat(scores), tokens, hidden


def evaluate_model(model, files, device, *, batch_size=128, candidate_budget=8192, candidate_chunk=2048, use_amp=False):
    import torch
    model.eval()
    total_loss, correct, count, bc_count = 0.0, 0, 0, 0
    ids_digest = hashlib.sha256()
    with torch.no_grad():
        for batch in batches(files, batch_size=batch_size, candidate_budget=candidate_budget, seed=None):
            amp = torch.autocast("cuda", dtype=torch.bfloat16) if use_amp else nullcontext()
            with amp:
                scores, _, _ = score_batch(model, batch, device, candidate_chunk)
            positives = torch.from_numpy(batch["positives"]).to(device)
            bc_supported = torch.from_numpy(batch["bc_supported"]).to(device)
            loss = marginal_loss(scores, positives, batch["counts"], bc_supported)
            total_loss += float(loss[bc_supported].sum())
            bc_count += int(bc_supported.sum())
            start = 0
            for index, size in enumerate(batch["counts"]):
                end = start + int(size)
                if batch["bc_supported"][index]:
                    correct += int(positives[start:end][scores[start:end].argmax()].item())
                start = end
            count += len(batch["ids"])
            for value in batch["ids"]:
                ids_digest.update((str(value) + "\n").encode())
    return {"samples": count, "bc_samples": bc_count, "public_ntp_only_samples": count - bc_count,
            "nll": total_loss / max(1, bc_count), "top1_accuracy": correct / max(1, bc_count),
            "visited_ids_sha256": ids_digest.hexdigest()}


def train(args) -> dict[str, Any]:
    import torch
    from fabledan.model_torch import FableDanNet, ModelConfig
    from fabledan.train_fast import atomic_checkpoint, atomic_export
    data, output = args.data.resolve(), args.out.resolve()
    if output == data or output.is_relative_to(data) or data.is_relative_to(output):
        raise DataError("checkpoint output must be separate from the immutable prepared data")
    if args.source and output.is_relative_to(args.source.resolve()):
        raise DataError("checkpoint output must stay outside the read-only source")
    manifest, paths = load_manifest(data, DEVELOPMENT_SPLITS)
    data_sha = sha256_file(data / "manifest.json")
    if args.source:
        source_sha = sha256_file(args.source / "manifest.json")
        if source_sha != manifest["source_manifest_sha256"]:
            raise DataError("live source manifest differs from prepared-data provenance")
        source_info = json.loads((args.source / "manifest.json").read_text(encoding="utf-8"))
        for split in DEVELOPMENT_SPLITS:
            # This hashes development files only. The held-out split stays closed.
            digest = sha256_file(args.source / f"{split}.jsonl")
            if digest != manifest["source_splits_sha256"][split] or digest != source_info["output_files"][f"{split}.jsonl"]["sha256"]:
                raise DataError(f"live source {split} differs from prepared-data provenance")
    torch.set_num_threads(args.threads)
    torch.backends.cuda.matmul.allow_tf32 = True
    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise DataError("requested CUDA device is unavailable")
    use_amp = device.startswith("cuda") and not args.fp32
    output.mkdir(parents=True, exist_ok=True)
    cfg = ModelConfig(n_blocks=args.n_blocks, ntp_weight=0.02 if args.ntp_weight is None else args.ntp_weight)
    checkpoint = None
    if args.resume or args.init:
        checkpoint = torch.load(args.resume or args.init, map_location=device, weights_only=False)
        cfg = ModelConfig.from_dict(checkpoint["config"])
        if cfg.feat_dim != FEAT_DIM or cfg.vocab != VOCAB or cfg.max_seq != MAX_SEQ:
            raise DataError("checkpoint has an incompatible competition feature contract")
    if args.ntp_weight is not None:
        cfg.ntp_weight = args.ntp_weight
    effective_ntp = cfg.ntp_weight
    if effective_ntp <= 0 and any(manifest["splits"][split]["counts"].get("public_ntp_only_records", 0) for split in DEVELOPMENT_SPLITS):
        raise DataError("NTP weight must be positive to train every public-NTP-only record")
    effective_lr = 1e-4 if args.lr is None else args.lr
    effective_seed = 20261003 if args.seed is None else args.seed
    if args.resume and args.seed is None:
        effective_seed = checkpoint.get("meta", {}).get("training_args", {}).get("seed", effective_seed)
    random.seed(effective_seed)
    np.random.seed(effective_seed)
    torch.manual_seed(effective_seed)
    model = FableDanNet(cfg).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=effective_lr)
    start_epoch, steps, best_nll, history = 0, 0, math.inf, []
    best_epoch = None
    if checkpoint:
        model.load_state_dict(checkpoint["model"])
    if args.resume:
        if args.resume.resolve().parent != output:
            raise DataError("resume must retain the checkpoint output directory; use --init for a new run")
        meta = checkpoint.get("meta", {})
        if meta.get("real_data_manifest_sha256") != data_sha or meta.get("training_kind") != "real_bc":
            raise DataError("resume checkpoint refers to different real training data")
        if checkpoint.get("optimizer"):
            optimizer.load_state_dict(checkpoint["optimizer"])
        if args.lr is not None:
            for group in optimizer.param_groups:
                group["lr"] = args.lr
        effective_lr = optimizer.param_groups[0]["lr"]
        start_epoch, steps = meta.get("epoch", 0), meta.get("steps", 0)
        best_nll = meta.get("best_validation_nll", math.inf)
        previous = output / "training.json"
        if previous.exists():
            previous_payload = json.loads(previous.read_text(encoding="utf-8"))
            if previous_payload.get("data_manifest_sha256") != data_sha:
                raise DataError("output training report has different real data")
            history = previous_payload.get("epochs", [])
            best_epoch = previous_payload.get("best_epoch")
    elif any((output / name).exists() for name in ("latest.pt", "best.pt", "training.json")):
        raise DataError("output contains training artifacts; use --resume or a separate directory")
    training_args = {"epochs": args.epochs, "lr": effective_lr, "ntp_weight": effective_ntp,
                     "seed": effective_seed, "batch": args.batch, "candidate_budget": args.candidate_budget,
                     "candidate_chunk": args.candidate_chunk, "threads": args.threads,
                     "log_every": args.log_every, "device": device, "amp": use_amp,
                     "resume": str(args.resume.resolve()) if args.resume else None,
                     "init": str(args.init.resolve()) if args.init else None}
    payload = {"schema": "oxbot-fabledan-real-training-v1", "status": "running", "data": str(data),
               "data_manifest_sha256": data_sha, "source_manifest_sha256": manifest["source_manifest_sha256"],
               "test_used": False, "splits_read": list(DEVELOPMENT_SPLITS), "device": device, "amp": use_amp,
               "config": cfg.to_dict(), "epochs": history, "seed": effective_seed,
               "training_args": training_args, "best_epoch": best_epoch,
               "expected_samples": {split: manifest["splits"][split]["counts"]["accepted_records"] for split in DEVELOPMENT_SPLITS},
               "objective": "exact multi-positive full-candidate BC + public next-token prediction; no results/hidden-hand targets"}
    write_json(output / "training.json", payload)
    started = time.monotonic()
    expected_train = payload["expected_samples"]["train"]
    for epoch in range(start_epoch + 1, args.epochs + 1):
        model.train()
        count, bc_count, loss_sum, ntp_sum = 0, 0, 0.0, 0.0
        visited = set()
        epoch_started = time.monotonic()
        for batch in batches(paths["train"], batch_size=args.batch, candidate_budget=args.candidate_budget, seed=effective_seed + epoch):
            optimizer.zero_grad(set_to_none=True)
            amp = torch.autocast("cuda", dtype=torch.bfloat16) if use_amp else nullcontext()
            with amp:
                scores, tokens, hidden = score_batch(model, batch, device, args.candidate_chunk)
                positives = torch.from_numpy(batch["positives"]).to(device)
                bc_supported = torch.from_numpy(batch["bc_supported"]).to(device)
                losses = marginal_loss(scores, positives, batch["counts"], bc_supported)
                supported_losses = losses[bc_supported]
                bc_loss = supported_losses.mean() if supported_losses.numel() else losses.sum() * 0.0
                ntp = model.ntp_loss(tokens, hidden) if effective_ntp else scores.new_zeros(())
                loss = bc_loss + effective_ntp * ntp
            if not torch.isfinite(loss):
                raise DataError("real-data training produced nonfinite loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            steps += 1
            batch_count = len(batch["ids"])
            count += batch_count
            bc_count += int(bc_supported.sum())
            loss_sum += float(supported_losses.detach().sum())
            ntp_sum += float(ntp.detach()) * batch_count
            for raw_id in batch["ids"]:
                sample_id = str(raw_id)
                if sample_id in visited:
                    raise DataError("real epoch repeats a record")
                visited.add(sample_id)
            if steps % args.log_every == 0:
                print(json.dumps({"epoch": epoch, "steps": steps, "samples": count,
                                  "nll": loss_sum / max(1, bc_count), "seconds": round(time.monotonic() - epoch_started, 1)}), flush=True)
        if count != expected_train or len(visited) != expected_train:
            raise DataError("real-data epoch did not visit every accepted train record exactly once")
        validation = evaluate_model(model, paths["validation"], device, batch_size=args.batch,
                                    candidate_budget=args.candidate_budget, candidate_chunk=args.candidate_chunk, use_amp=use_amp)
        expected_validation = payload["expected_samples"]["validation"]
        if validation["samples"] != expected_validation or validation["visited_ids_sha256"] != manifest["splits"]["validation"]["accepted_ids_sha256"]:
            raise DataError("validation did not visit every accepted validation record exactly once")
        improved = validation["nll"] < best_nll
        best_nll = min(best_nll, validation["nll"])
        entry = {"epoch": epoch, "steps": steps, "train_samples": count, "train_unique_samples": len(visited),
                 "train_visited_ids_sha256": hashlib.sha256("\n".join(sorted(visited)).encode()).hexdigest(),
                 "train_bc_samples": bc_count, "train_public_ntp_only_samples": count - bc_count,
                 "train_nll": loss_sum / max(1, bc_count), "ntp_loss": ntp_sum / count,
                 "validation": validation, "seconds": time.monotonic() - epoch_started,
                 "selected_best": improved}
        payload["epochs"].append(entry)
        meta = {"training_kind": "real_bc", "epoch": epoch, "steps": steps, "cycle": 0,
                "total_samples": epoch * expected_train, "real_data_manifest_sha256": data_sha,
                "best_validation_nll": best_nll, "test_used": False, "training_args": training_args,
                "train_records": expected_train, "validation_records": expected_validation}
        atomic_checkpoint(model, optimizer, meta, str(output / "latest.pt"))
        atomic_export(model, str(output / "latest.npz"))
        if improved:
            atomic_checkpoint(model, optimizer, meta, str(output / "best.pt"))
            atomic_export(model, str(output / "best.npz"))
            payload["best_epoch"] = epoch
        write_json(output / "training.json", payload)
        print(json.dumps(entry), flush=True)
    if sha256_file(data / "manifest.json") != data_sha:
        raise DataError("prepared-data manifest changed during training")
    if len(payload["epochs"]) < args.epochs:
        raise DataError("training report lacks required completed epochs")
    payload["status"] = "complete"
    payload["total_seconds_this_run"] = time.monotonic() - started
    payload["best_validation_nll"] = best_nll
    payload["checkpoint_sha256"] = {name: sha256_file(output / name) for name in ("best.pt", "best.npz", "latest.pt", "latest.npz")}
    if device.startswith("cuda"):
        payload["cuda_peak_allocated_bytes"] = torch.cuda.max_memory_allocated(device)
    write_json(output / "training.json", payload)
    return payload


def run_evaluation(args) -> dict[str, Any]:
    import torch
    from fabledan.model_torch import load_ckpt
    manifest, paths = load_manifest(args.data.resolve(), (args.split,))
    data_sha = sha256_file(args.data / "manifest.json")
    model, checkpoint = load_ckpt(str(args.checkpoint), device=args.device)
    if checkpoint.get("meta", {}).get("real_data_manifest_sha256") != data_sha:
        raise DataError("checkpoint and held-out data have different prepared provenance")
    report = evaluate_model(model, paths[args.split], args.device, batch_size=args.batch,
                            candidate_budget=args.candidate_budget, candidate_chunk=args.candidate_chunk)
    expected = manifest["splits"][args.split]
    if report["samples"] != expected["counts"]["accepted_records"] or report["visited_ids_sha256"] != expected["accepted_ids_sha256"]:
        raise DataError("evaluation coverage differs from prepared split")
    report.update({"schema": "oxbot-fabledan-real-evaluation-v1", "split": args.split,
                   "test_used": args.split == "test", "data_manifest_sha256": data_sha,
                   "checkpoint_sha256": sha256_file(args.checkpoint)})
    write_json(args.out, report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--source", type=Path, default=ROOT / "data/processed/njupt")
    prep.add_argument("--out", type=Path, default=ROOT / "data/processed/fabledan-real-v2")
    prep.add_argument("--shard-size", type=int, default=512)
    prep.add_argument("--include-test", action="store_true", help="prepare an isolated held-out test shard set; never used by train")
    training = sub.add_parser("train")
    training.add_argument("--data", type=Path, default=ROOT / "data/processed/fabledan-real-v2")
    training.add_argument("--source", type=Path, default=ROOT / "data/processed/njupt")
    training.add_argument("--out", type=Path, default=HERE / "ckpts/real-v2")
    training.add_argument("--epochs", type=int, default=8)
    training.add_argument("--lr", type=float, default=None, help="fresh default 1e-4; resume preserves optimizer rate unless supplied")
    training.add_argument("--ntp-weight", type=float, default=None, help="fresh default 0.02; checkpoint value retained unless supplied")
    training.add_argument("--seed", type=int, default=None, help="fresh default 20261003; resume retains saved seed unless supplied")
    training.add_argument("--n-blocks", type=int, default=4)
    training.add_argument("--threads", type=int, default=4)
    training.add_argument("--log-every", type=int, default=100)
    training.add_argument("--resume", type=Path)
    training.add_argument("--init", type=Path, help="initialize compatible weights; start a new real-data training run")
    training.add_argument("--fp32", action="store_true")
    evaluation = sub.add_parser("evaluate")
    evaluation.add_argument("--data", type=Path, default=ROOT / "data/processed/fabledan-real-v2")
    evaluation.add_argument("--checkpoint", type=Path, required=True)
    evaluation.add_argument("--split", choices=("validation", "test"), default="test")
    evaluation.add_argument("--out", type=Path, required=True)
    for command in (training, evaluation):
        command.add_argument("--device", default="cuda:0")
        command.add_argument("--batch", type=int, default=128)
        command.add_argument("--candidate-budget", type=int, default=8192)
        command.add_argument("--candidate-chunk", type=int, default=2048)
    args = parser.parse_args()
    if args.command == "prepare":
        result = prepare(args.source, args.out, splits=("train", "validation", "test") if args.include_test else DEVELOPMENT_SPLITS,
                         shard_size=args.shard_size)
    else:
        for name in ("batch", "candidate_budget", "candidate_chunk"):
            if getattr(args, name) < 1:
                parser.error(f"--{name.replace('_', '-')} must be positive")
        if args.command == "train":
            for name in ("epochs", "threads", "log_every", "n_blocks"):
                if getattr(args, name) < 1:
                    parser.error(f"--{name.replace('_', '-')} must be positive")
            if args.resume and args.init:
                parser.error("--resume and --init are mutually exclusive")
            if ((args.lr is not None and (not math.isfinite(args.lr) or args.lr <= 0)) or
                    (args.ntp_weight is not None and (not math.isfinite(args.ntp_weight) or args.ntp_weight < 0))):
                parser.error("invalid learning rate / NTP weight")
            result = train(args)
        else:
            result = run_evaluation(args)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
