#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Safe, deterministic ETL for the NUPT GuanDan replay collection.

The downloaded ``.data`` files are concatenated protocol-3 pickle streams.  They
are treated as untrusted input: this module never imports classes while loading
them, validates the resulting primitive tree, and stops at the first V result
record.  The ETL output is intended for offline training/audit only; it is not
part of the BotZone submission.

Typical use::

    python train/etl_njupt.py \
      --source "D:\\coding\\RL\\训练数据\\南邮" \
      --output data/processed/njupt

The output consists of split decision JSONL files, a full public-event JSONL
file, a quarantine report, and a reproducible manifest.  UTF-8 is used for all
text files.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence


SCHEMA_VERSION = "njupt-decision-v3"
MANIFEST_SCHEMA = "njupt-etl-manifest-v3"
PLAY_HISTORY_LIMIT = 128
CANONICAL_FACE_COUNT = 54
DECK_SIZE = 108
SOURCE_MIN = 2
SOURCE_MAX = 55
DEFAULT_MAX_FILE_BYTES = 4 * 1024 * 1024
DEFAULT_MAX_EVENTS = 20_000
DEFAULT_MAX_DEPTH = 32
DEFAULT_MAX_SEQUENCE = 4096
DEFAULT_MAX_STRING = 4096
DEFAULT_MAX_NODES = 100_000

# The public competition encoding is rank-major in four suit blocks.  Audit of
# the tribute records identifies block 1 as hearts.  The other three blocks
# follow the historical NUPT order diamond, heart, spade, club.  BotZone's
# canonical face index is rank-major with suit order heart, diamond, spade,
# club, so the permutation below is deliberately explicit.
SOURCE_SUIT_TO_BOTZONE_SUIT = (1, 0, 2, 3)
SOURCE_JOKER_SMALL = 54
SOURCE_JOKER_BIG = 55

RANK_LABELS = {
    2: "2",
    3: "3",
    4: "4",
    5: "5",
    6: "6",
    7: "7",
    8: "8",
    9: "9",
    10: "0",
    11: "J",
    12: "Q",
    13: "K",
    14: "A",
}
BOTZONE_NATURAL_RANKS = ("A", "2", "3", "4", "5", "6", "7", "8", "9", "0", "J", "Q", "K")
BOTZONE_RANK_INDEX = {name: index for index, name in enumerate(BOTZONE_NATURAL_RANKS)}


class ETLError(RuntimeError):
    """Base error for malformed or unsafe ETL input."""


class UnsafePickleError(ETLError):
    """Raised when a pickle stream violates the restricted loader policy."""


class SchemaError(ETLError):
    """Raised when a replay object is not in the expected NUPT shape."""


class LimitedUnpickler(pickle.Unpickler):
    """Unpickler which cannot resolve classes or persistent objects."""

    def find_class(self, module: str, name: str) -> Any:  # pragma: no cover - defensive path
        raise UnsafePickleError(f"global pickle reference rejected: {module}.{name}")

    def persistent_load(self, pid: Any) -> Any:  # pragma: no cover - defensive path
        raise UnsafePickleError("persistent pickle reference rejected")


def _validate_primitive(value: Any, *, depth: int, max_depth: int, max_sequence: int, max_string: int, seen: set[int], budget: list[int]) -> None:
    """Validate that a loaded object is only the primitive replay tree.

    The source schema uses tuples/lists/integers/strings.  Dicts, bytes,
    floats, sets, custom objects, and recursive containers are rejected.  The
    depth and sequence limits also protect the caller from tiny pickles which
    expand into unexpectedly large structures.
    """

    budget[0] -= 1
    if budget[0] < 0:
        raise UnsafePickleError("pickle value count exceeds limit")
    if depth > max_depth:
        raise UnsafePickleError(f"pickle nesting exceeds {max_depth}")
    if value is None or isinstance(value, (int, str)):
        if isinstance(value, str) and len(value) > max_string:
            raise UnsafePickleError("pickle string exceeds limit")
        if isinstance(value, int) and value.bit_length() > 64:
            raise UnsafePickleError("pickle integer exceeds 64 bits")
        return
    if isinstance(value, (tuple, list)):
        if len(value) > max_sequence:
            raise UnsafePickleError("pickle sequence exceeds limit")
        marker = id(value)
        if marker in seen:
            raise UnsafePickleError("recursive pickle container rejected")
        seen.add(marker)
        try:
            for child in value:
                _validate_primitive(
                    child,
                    depth=depth + 1,
                    max_depth=max_depth,
                    max_sequence=max_sequence,
                    max_string=max_string,
                    seen=seen,
                    budget=budget,
                )
        finally:
            seen.remove(marker)
        return
    raise UnsafePickleError(f"unsupported pickle value type: {type(value).__name__}")


def iter_safe_pickle(path: Path, *, max_file_bytes: int = DEFAULT_MAX_FILE_BYTES, max_events: int = DEFAULT_MAX_EVENTS, max_depth: int = DEFAULT_MAX_DEPTH, max_sequence: int = DEFAULT_MAX_SEQUENCE, max_string: int = DEFAULT_MAX_STRING, max_nodes: int = DEFAULT_MAX_NODES) -> Iterator[Any]:
    """Yield a concatenated pickle stream under strict resource limits."""

    size = path.stat().st_size
    if size > max_file_bytes:
        raise UnsafePickleError(f"pickle file {path} is {size} bytes, limit is {max_file_bytes}")
    count = 0
    with path.open("rb") as handle:
        while True:
            record_offset = handle.tell()
            try:
                # Each source event is an independent pickle.dump call.
                value = LimitedUnpickler(handle).load()
            except EOFError:
                if record_offset != size:
                    raise UnsafePickleError(f"truncated pickle record at byte {record_offset}")
                return
            except (pickle.UnpicklingError, IndexError, ValueError, TypeError) as exc:
                raise UnsafePickleError(f"pickle decode failed for {path}: {exc}") from exc
            count += 1
            if count > max_events:
                raise UnsafePickleError(f"pickle event count exceeds {max_events}")
            _validate_primitive(
                value,
                depth=0,
                max_depth=max_depth,
                max_sequence=max_sequence,
                max_string=max_string,
                seen=set(),
                budget=[max_nodes],
            )
            yield value


def sha256_file(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def source_id_to_face(source_id: int) -> int:
    """Map an NUPT source card id to BotZone's canonical 0..53 face id."""

    if not isinstance(source_id, int) or isinstance(source_id, bool) or not SOURCE_MIN <= source_id <= SOURCE_MAX:
        raise SchemaError(f"source card id outside 2..55: {source_id!r}")
    if source_id == SOURCE_JOKER_SMALL:
        return 52
    if source_id == SOURCE_JOKER_BIG:
        return 53
    rank_number = ((source_id - 2) % 13) + 2
    source_suit = (source_id - 2) // 13
    if source_suit < 0 or source_suit > 3:
        raise SchemaError(f"invalid source suit block: {source_id}")
    rank_label = RANK_LABELS[rank_number]
    rank_index = BOTZONE_RANK_INDEX[rank_label]
    botzone_suit = SOURCE_SUIT_TO_BOTZONE_SUIT[source_suit]
    return rank_index * 4 + botzone_suit


def face_to_physical_ids(face: int) -> tuple[int, int]:
    """Return deterministic first/second-deck BotZone ids for one face."""

    if not isinstance(face, int) or not 0 <= face < CANONICAL_FACE_COUNT:
        raise SchemaError(f"invalid canonical face: {face!r}")
    return (face, face + 54)


def _physical_card_name(card_id: int) -> str:
    """Human-readable canonical card name used only in optional diagnostics."""

    if card_id in (52, 106):
        return "jo"
    if card_id in (53, 107):
        return "jO"
    face = card_id % 54
    rank = BOTZONE_NATURAL_RANKS[face // 4]
    suit = ("h", "d", "s", "c")[face % 4]
    return suit + rank


def _counter_as_list(counter: Mapping[int, int]) -> list[int]:
    return [int(counter.get(face, 0)) for face in range(CANONICAL_FACE_COUNT)]


def _local_physical_cards(faces: Sequence[int]) -> list[int]:
    """Represent a visible multiset without consulting any other seat's hand.

    Global replay entities are useful for ownership checks but their artificial
    deck index depends on initial seat allocation.  They must never become an
    observation feature.  Local IDs use ``face`` then ``face + 54`` solely from
    this player's currently visible cards; labels use the same convention.
    """

    cards: list[int] = []
    for face, count in sorted(Counter(faces).items()):
        if not 0 <= face < CANONICAL_FACE_COUNT or not 1 <= count <= 2:
            raise SchemaError(f"invalid visible face multiset: {face}:{count}")
        cards.extend(face + 54 * copy for copy in range(count))
    return sorted(cards)


def _source_card_counter(cards: Sequence[int]) -> Counter[int]:
    result: Counter[int] = Counter()
    for card in cards:
        result[source_id_to_face(card)] += 1
    return result


def _validate_tuple_shape(event: Any) -> tuple[Any, ...]:
    if not isinstance(event, tuple) or not event or not isinstance(event[0], str):
        raise SchemaError(f"event is not a tagged tuple: {event!r}")
    return event


def _validate_deal_hands(hands: Mapping[int, Sequence[int]]) -> None:
    if set(hands) != {0, 1, 2, 3}:
        raise SchemaError(f"deal must contain seats 0..3, got {sorted(hands)}")
    all_faces: list[int] = []
    for seat in range(4):
        cards = hands[seat]
        if len(cards) != 27:
            raise SchemaError(f"seat {seat} initial hand has {len(cards)} cards")
        for card in cards:
            all_faces.append(source_id_to_face(card))
    counts = Counter(all_faces)
    if set(counts) != set(range(CANONICAL_FACE_COUNT)) or set(counts.values()) != {2}:
        raise SchemaError("initial deal does not contain exactly two copies of each face")


def _assign_physical_hands(hands: Mapping[int, Sequence[int]]) -> tuple[dict[int, list[int]], dict[int, Counter[int]]]:
    """Assign duplicate faces to deterministic first/second deck entities."""

    available = {face: list(face_to_physical_ids(face)) for face in range(CANONICAL_FACE_COUNT)}
    physical: dict[int, list[int]] = {}
    faces: dict[int, Counter[int]] = {}
    for seat in range(4):
        ids: list[int] = []
        counter: Counter[int] = Counter()
        for source_card in hands[seat]:
            face = source_id_to_face(source_card)
            if not available[face]:
                raise SchemaError(f"physical card over-allocation for face {face}")
            ids.append(available[face].pop(0))
            counter[face] += 1
        physical[seat] = ids
        faces[seat] = counter
    if any(available[face] for face in available):
        raise SchemaError("physical assignment did not consume exactly two copies per face")
    return physical, faces


def _take_face_card(face_hands: Mapping[int, Counter[int]], physical_hands: Mapping[int, list[int]], seat: int, face: int, count: int = 1) -> list[int]:
    if face_hands[seat].get(face, 0) < count:
        raise SchemaError(f"seat {seat} does not own face {face}")
    candidates = sorted(card for card in physical_hands[seat] if card % 54 == face)
    if len(candidates) < count:
        raise SchemaError(f"seat {seat} does not own enough physical copies of face {face}")
    selected = candidates[:count]
    for card in selected:
        physical_hands[seat].remove(card)
    face_hands[seat][face] -= count
    return selected


def _give_physical_cards(face_hands: Mapping[int, Counter[int]], physical_hands: Mapping[int, list[int]], seat: int, face: int, cards: Sequence[int]) -> None:
    face_hands[seat][face] += len(cards)
    physical_hands[seat].extend(cards)
    physical_hands[seat].sort()


def _compact_public_event(event: Mapping[str, Any]) -> list[Any]:
    """Compact event used in bounded decision history and full game events."""

    tag = event["tag"]
    if tag == "P":
        return ["P", event["seat"], event.get("face_cards", []), bool(event.get("pass", False))]
    if tag in ("T", "B"):
        return [tag, event["from"], event["to"], event.get("face_cards", [])]
    if tag == "C":
        return ["C"]
    if tag == "R":
        return ["R", event["current_level"], event["team_levels"][0], event["team_levels"][1]]
    return [tag]


def _rank_value(face: int) -> int | None:
    if face >= 52:
        return None
    return face // 4  # A=0, 2=1, ..., K=12


def _public_claim(face_cards: Sequence[int], level: int) -> list[int] | None:
    """Printed non-wildcard faces are the only claim consistent with action."""

    return None if source_id_to_face(level + 13) in face_cards else list(face_cards)


def _candidate_claims(face_cards: Sequence[int], level: int) -> tuple[list[dict[str, Any]], str]:
    """Reconstruct natural claim candidates without inventing wildcard labels.

    The recorded action never contains a BotZone ``claim``.  For a natural
    action its printed faces determine a candidate; actions using a heart level
    card stay explicitly unresolved for the legal-candidate reconstruction
    stage.  These are candidates, not evidence of the source player's claim.
    """

    wildcard_face = source_id_to_face(level + 13)
    if wildcard_face in face_cards:
        return [], "pending_reconstruction"
    ranks = [_rank_value(face) for face in face_cards]
    if any(rank is None for rank in ranks):
        if len(face_cards) == 1:
            return [{"kind": "single", "claim": list(face_cards)}], "natural_reconstructed"
        if len(face_cards) == 2 and len(set(face_cards)) == 1:
            return [{"kind": "pair", "claim": list(face_cards)}], "natural_reconstructed"
        if Counter(face_cards) == Counter({52: 2, 53: 2}):
            return [{"kind": "rocket", "claim": list(face_cards)}], "natural_reconstructed"
        return [], "pending_reconstruction"
    n = len(ranks)
    counts = Counter(ranks)
    claims: list[dict[str, Any]] = []

    def add(kind: str, key: Any = None, secondary: Any = None) -> None:
        claims.append({"kind": kind, "key": key, "secondary": secondary, "claim": list(face_cards)})

    if n == 1:
        add("single", ranks[0])
    elif n == 2 and len(counts) == 1:
        add("pair", ranks[0])
    elif n == 3 and len(counts) == 1:
        add("three", ranks[0])
    elif n == 4 and len(counts) == 1:
        add("bomb", ranks[0])
    elif n == 5:
        if sorted(counts.values()) == [2, 3]:
            triple = next(rank for rank, count in counts.items() if count == 3)
            pair = next(rank for rank, count in counts.items() if count == 2)
            add("set", triple, pair)
        elif len(counts) == 1:
            add("bomb", ranks[0])
        elif len(set(ranks)) == 5 and (max(ranks) - min(ranks) == 4 or set(ranks) == {0, 9, 10, 11, 12}):
            key = 9 if set(ranks) == {0, 9, 10, 11, 12} else min(ranks)
            add("straight_flush" if len({face % 4 for face in face_cards}) == 1 else "straight", key)
    elif n == 6:
        if len(counts) == 1:
            add("bomb", ranks[0])
        elif sorted(counts.values()) == [2, 2, 2] and (max(ranks) - min(ranks) == 2 or set(ranks) == {0, 11, 12}):
            add("triple_pairs", 11 if set(ranks) == {0, 11, 12} else min(ranks))
        elif sorted(counts.values()) == [3, 3] and (max(ranks) - min(ranks) == 1 or set(ranks) == {0, 12}):
            add("three_straight", 12 if set(ranks) == {0, 12} else min(ranks))
    elif 7 <= n <= 10 and len(counts) == 1:
        add("bomb", ranks[0])
    return claims, ("natural_reconstructed" if claims else "pending_reconstruction")


@dataclass(frozen=True)
class MatchRecord:
    match_id: str
    archive_path: str
    archive_sha256: str | None
    extracted_path: Path


@dataclass
class ParsedGame:
    match: MatchRecord
    game_id: str
    data_path: Path
    data_sha256: str
    game_index: int
    events: list[dict[str, Any]]
    decisions: list[dict[str, Any]]
    result: dict[str, Any]
    quality_flags: list[str] = field(default_factory=list)
    trailing_events: list[Any] = field(default_factory=list)
    deal_count: int = 0


def _jsonable(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    raise TypeError(type(value).__name__)


def _match_records(source_root: Path, manifest_path: Path | None = None) -> tuple[list[MatchRecord], str | None]:
    manifest_path = manifest_path or (source_root / "清单" / "njupt_archives.json")
    manifest_sha = sha256_file(manifest_path) if manifest_path.exists() else None
    records: list[MatchRecord] = []
    if manifest_path.exists():
        raw = json.loads(manifest_path.read_text(encoding="utf-8"))
        for record in raw.get("records", []):
            file_name = str(record.get("fileName", ""))
            if " vs " not in file_name or not record.get("extractedTo"):
                continue
            extracted = (source_root / Path(str(record["extractedTo"]))).resolve()
            if not extracted.is_relative_to(source_root.resolve()):
                raise SchemaError(f"manifest extracted path is outside the source root: {extracted}")
            if not extracted.exists():
                continue
            records.append(
                MatchRecord(
                    match_id=Path(file_name).stem,
                    archive_path=str(record.get("archive", "")),
                    archive_sha256=record.get("sha256"),
                    extracted_path=extracted,
                )
            )
    else:
        extracted_root = source_root / "已解压"
        if extracted_root.exists():
            for extracted in sorted(path for path in extracted_root.iterdir() if path.is_dir() and " vs " in path.name):
                records.append(MatchRecord(extracted.name, "", None, extracted))
    records.sort(key=lambda item: (item.match_id, item.archive_path))
    return records, manifest_sha


def assign_splits(records: Sequence[MatchRecord]) -> dict[str, str]:
    """Deterministically assign exactly 32/7/7 match packages."""

    ordered = sorted(records, key=lambda item: (item.match_id, item.archive_path))
    if len(ordered) < 46:
        # Fixtures and partial local mirrors fill the same ordered partitions.
        n = len(ordered)
        train_n = min(32, n)
        valid_n = min(7, max(0, n - train_n))
    else:
        train_n, valid_n = 32, 7
    result: dict[str, str] = {}
    for index, record in enumerate(ordered):
        result[record.match_id] = "train" if index < train_n else "validation" if index < train_n + valid_n else "test"
    return result


def _expected_tribute(previous_done: Sequence[int]) -> int:
    """Infer the protocol's tribute count from the public finish order."""

    if len(previous_done) >= 2 and (previous_done[0] - previous_done[1]) % 2 == 0:
        return 2
    if len(previous_done) >= 3 and (previous_done[0] - previous_done[2]) % 2 == 0:
        return 1
    if previous_done:
        return 1
    return 0


def _parse_game_index(path: Path) -> int | None:
    match = re.search(r"_(\d+)\.data$", path.name)
    return int(match.group(1)) if match else None


def _select_game_files(match: MatchRecord, *, max_file_bytes: int = DEFAULT_MAX_FILE_BYTES, max_events: int = DEFAULT_MAX_EVENTS) -> tuple[list[tuple[int, Path]], list[dict[str, Any]]]:
    candidates = [path for path in match.extracted_path.rglob("*.data") if path.is_file() and path.stat().st_size > 0 and re.search(r"\.data$", path.name)]
    grouped: dict[int | None, list[Path]] = defaultdict(list)
    for path in candidates:
        grouped[_parse_game_index(path)].append(path)
    selected: list[tuple[int, Path]] = []
    quarantine: list[dict[str, Any]] = []
    for game_index in sorted(grouped, key=lambda value: (value is None, value if value is not None else -1)):
        paths = sorted(grouped[game_index], key=lambda path: path.as_posix())
        if game_index is None:
            for path in paths:
                quarantine.append({"reason": "unindexed_data_file", "path": path.relative_to(match.extracted_path).as_posix()})
            continue
        # Duplicate _0 files occur in the source once: the shorter one is a
        # truncated initial-deal artifact.  Prefer a candidate containing V.
        complete: list[Path] = []
        for path in paths:
            try:
                events = list(iter_safe_pickle(path, max_file_bytes=max_file_bytes, max_events=max_events))
            except (ETLError, OSError, pickle.PickleError) as exc:
                quarantine.append({"reason": "unsafe_or_malformed_pickle", "path": path.relative_to(match.extracted_path).as_posix(), "detail": str(exc)})
                continue
            if not any(isinstance(event, tuple) and event and event[0] == "V" for event in events):
                quarantine.append({"reason": "missing_terminal_v", "path": path.relative_to(match.extracted_path).as_posix(), "event_count": len(events), "event_tags": dict(Counter(event[0] for event in events if isinstance(event, tuple) and event))})
                continue
            complete.append(path)
        if complete:
            chosen = complete[0]
            selected.append((game_index, chosen))
            for path in complete[1:]:
                quarantine.append({"reason": "duplicate_game_index", "path": path.relative_to(match.extracted_path).as_posix(), "selected": chosen.relative_to(match.extracted_path).as_posix()})
    return selected, quarantine


def _find_result_marker(data_path: Path) -> tuple[int, int] | None:
    matches = []
    pattern = re.compile(re.escape(data_path.name) + r"_(\d+)_(\d+)$")
    for sibling in data_path.parent.iterdir():
        match = pattern.fullmatch(sibling.name)
        if match and sibling.is_file():
            matches.append((int(match.group(1)), int(match.group(2))))
    return sorted(matches)[0] if matches else None


def _find_series_result(match: MatchRecord) -> dict[str, Any] | None:
    ros_files = sorted(match.extracted_path.rglob("*.ros"))
    if not ros_files:
        return None
    name = ros_files[0].name
    match_result = re.search(r"^(.*?)_(\d+)_(\d+)_\d{8}_\d{6}\.ros$", name)
    if not match_result:
        return {"ros_file": ros_files[0].relative_to(match.extracted_path).as_posix()}
    return {
        "ros_file": ros_files[0].relative_to(match.extracted_path).as_posix(),
        "score": [int(match_result.group(2)), int(match_result.group(3))],
        "team_name_prefix": match_result.group(1),
    }


def _game_result(v_event: tuple[Any, ...], marker: tuple[int, int] | None, series: dict[str, Any] | None) -> dict[str, Any]:
    payload = v_event[1] if len(v_event) == 2 and isinstance(v_event[1], tuple) else ()
    result: dict[str, Any] = {"single_game": {"raw_v": _jsonable(v_event)}}
    if len(payload) >= 6:
        result["single_game"].update(
            {
                "winner_team_name": payload[0],
                "winner_seat": payload[1],
                "winner_partner_seat": payload[2],
                "winner_target_level": payload[3],
                "loser_level": payload[4],
                "target_level_repeat": payload[5],
            }
        )
    if marker is not None:
        result["single_game"]["marker_levels"] = list(marker)
    if series is not None:
        result["series"] = series
    return result


def _parse_game(match: MatchRecord, game_index: int, data_path: Path, *, max_file_bytes: int, max_events: int) -> ParsedGame:
    raw_events = list(iter_safe_pickle(data_path, max_file_bytes=max_file_bytes, max_events=max_events))
    data_sha = sha256_file(data_path)
    first_v = next((index for index, event in enumerate(raw_events) if isinstance(event, tuple) and event and event[0] == "V"), None)
    if first_v is None:
        raise SchemaError("data stream has no V result event")
    trailing = raw_events[first_v + 1 :]
    quality: list[str] = []
    if any(isinstance(event, tuple) and event and event[0] != "F" for event in trailing):
        quality.append("trailing_events_after_v")
    if any(isinstance(event, tuple) and event and event[0] == "F" for event in trailing):
        quality.append("has_series_final_score")
    marker = _find_result_marker(data_path)
    series = _find_series_result(match)
    result = _game_result(raw_events[first_v], marker, series)
    if marker is not None and len(result["single_game"].get("marker_levels", [])) == 2:
        if result["single_game"].get("loser_level") not in marker:
            quality.append("marker_v_level_order_mismatch")

    events_out: list[dict[str, Any]] = []
    decisions: list[dict[str, Any]] = []
    current_level: int | None = None
    team_levels: list[int] = [2, 2]
    hands_raw: dict[int, list[int]] | None = None
    physical_hands: dict[int, list[int]] | None = None
    face_hands: dict[int, Counter[int]] | None = None
    deal_id = -1
    public_history: list[list[Any]] = []
    play_history: list[dict[str, Any]] = []
    done_order: list[int] = []
    game_events_by_deal: dict[int, list[list[Any]]] = defaultdict(list)
    played_faces: Counter[int] = Counter()
    pending_r: list[tuple[Any, ...]] = []
    last_player = -1
    last_cards: list[int] = []
    last_claim: list[int] | None = None
    tribute_count = 0
    expected_tribute = 0
    exchange_context_known = False
    resist = False
    expected_player: int | None = None
    f_after_v = next((event for event in trailing if isinstance(event, tuple) and event and event[0] == "F"), None)
    if f_after_v is not None:
        result["series_game_score"] = _jsonable(f_after_v)

    def emit_decision(tag: str, seat: int, label: dict[str, Any], event_index: int) -> None:
        assert physical_hands is not None and face_hands is not None and current_level is not None
        stage = {"P": "play", "T": "tribute", "B": "return"}[tag]
        claims, claim_status = ([], "not_applicable")
        if tag == "P" and not label.get("pass", False):
            claims, claim_status = _candidate_claims(label["face_cards"], current_level)
        history_tail = public_history[-64:]
        own_faces = face_hands[seat]
        own_hand = _local_physical_cards(list(own_faces.elements()))
        local_label_cards = _local_physical_cards(label.get("face_cards", []))
        leading = last_player in (-1, seat)
        quality_flags = list(quality)
        if tag == "B" and label.get("face_cards"):
            # Current-level returns are accepted by the historical source but
            # rejected by the corrected BotZone oracle; mark, do not discard.
            level_faces = {
                source_id_to_face(current_level + 13 * block)
                for block in range(4)
            }
            if any(face in level_faces for face in label["face_cards"]):
                quality_flags.append("return_current_level_conflicts_oracle")
        record = {
            "schema": SCHEMA_VERSION,
            "match_id": match.match_id,
            "game_id": f"{match.match_id}__game{game_index}",
            "deal_id": deal_id,
            "event_index": event_index,
            "stage": stage,
            "features": {
                "seat": seat,
                "team": seat % 2,
                "level": current_level,
                "level_label": RANK_LABELS[current_level],
                "team_levels": list(team_levels),
                "own_hand": own_hand,
                "own_hand_faces": _counter_as_list(own_faces),
                "remaining_counts": [len(physical_hands[player]) for player in range(4)],
                "leading": leading,
                "last_player": -1 if leading else last_player,
                "last_cards": [] if leading else _local_physical_cards(last_cards),
                "last_claim": None if leading else last_claim,
                "tribute": expected_tribute,
                "resist": resist,
                "exchange_context_known": exchange_context_known,
                "history": play_history[-PLAY_HISTORY_LIMIT:],
                "history_total": len(play_history),
                "public_history": history_tail,
                "public_history_truncated": max(0, len(public_history) - len(history_tail)) > 0,
                "public_history_total": len(public_history),
                "played_face_counts": _counter_as_list(played_faces),
                "done_order": list(done_order),
            },
            "label": {
                "kind": "pass" if label.get("pass", False) else stage,
                "cards": local_label_cards,
                "face_cards": list(label.get("face_cards", [])),
                "source_cards": list(label.get("source_cards", [])),
                "claim": None,
                "claim_candidates": claims,
                "claim_status": claim_status,
            },
            "provenance": {
                "source_data": data_path.relative_to(match.extracted_path).as_posix(),
                "data_sha256": data_sha,
                "archive_sha256": match.archive_sha256,
                "raw_tag": tag,
            },
            "quality_flags": sorted(set(quality_flags)),
            "result": result,
        }
        decisions.append(record)

    for index, raw in enumerate(raw_events[: first_v + 1]):
        event = _validate_tuple_shape(raw)
        tag = event[0]
        if tag == "R":
            if len(event) != 3 or event[1] not in (-1, 0, 1) or type(event[2]) is not int or not 2 <= event[2] <= 14:
                raise SchemaError(f"invalid R event at {index}: {event!r}")
            if event[1] != (-1, 0, 1)[len(pending_r)]:
                raise SchemaError(f"R group is not (-1,0,1) at {index}")
            pending_r.append(event)
            if event[1] == 1:
                previous_done = list(done_order)
                expected_tribute = _expected_tribute(previous_done)
                exchange_context_known = bool(previous_done)
                current_level = pending_r[-3][2]
                team_levels = [pending_r[-2][2], pending_r[-1][2]]
                if done_order and current_level != team_levels[done_order[0] % 2]:
                    raise SchemaError(f"current level differs from previous first-finisher team at {index}")
                pending_r.clear()
                deal_id += 1
                public_history = []
                play_history = []
                done_order = []
                played_faces = Counter()
                hands_raw = {}
                physical_hands = None
                face_hands = None
                last_player = -1
                last_cards = []
                last_claim = None
                tribute_count = 0
                resist = False
                expected_player = None
                r_record = {"tag": "R", "current_level": current_level, "team_levels": list(team_levels), "event_index": index, "deal_id": deal_id}
                events_out.append(r_record)
                compact = _compact_public_event(r_record)
                public_history.append(compact)
                game_events_by_deal[deal_id].append(compact)
            continue
        if tag == "I":
            if len(event) != 3 or event[1] not in (0, 1, 2, 3) or not isinstance(event[2], list):
                raise SchemaError(f"invalid I event at {index}: {event!r}")
            if hands_raw is None or pending_r or event[1] != len(hands_raw):
                raise SchemaError(f"initial hands are not ordered I0..I3 after R at {index}")
            hands_raw[event[1]] = event[2]
            if event[1] == 3:
                _validate_deal_hands(hands_raw)
                physical_hands, face_hands = _assign_physical_hands(hands_raw)
                # Initial deals are private; retain only a compact audit event
                # in games.jsonl, never in decision features.
                events_out.append({"tag": "I", "seat": 3, "deal_id": deal_id, "event_index": index, "private": True})
            continue
        if tag == "P":
            if physical_hands is None or face_hands is None or deal_id < 0:
                raise SchemaError(f"P before initialized deal at {index}")
            if len(event) != 3 or event[1] not in (0, 1, 2, 3):
                raise SchemaError(f"invalid P event at {index}: {event!r}")
            seat = int(event[1])
            if expected_player is not None and seat != expected_player:
                raise SchemaError(f"P turn order mismatch at {index}: expected {expected_player}, got {seat}")
            if not physical_hands[seat]:
                raise SchemaError(f"finished seat acts at {index}: {seat}")
            is_pass = type(event[2]) is int and event[2] == 1
            if not is_pass and (not isinstance(event[2], list) or not event[2]):
                raise SchemaError(f"invalid P action at {index}: {event!r}")
            if is_pass and last_player in (-1, seat):
                raise SchemaError(f"leading P cannot pass at {index}")
            if not play_history:
                # At the first P the exchange stage has visibly ended. The
                # previous public finish order determines how many tributes
                # were due; no transfers at all therefore means resistance.
                # This uses only the event prefix before the current action.
                if exchange_context_known:
                    if tribute_count not in (0, expected_tribute):
                        raise SchemaError(f"partial tribute sequence before play at {index}")
                    resist = expected_tribute > 0 and tribute_count == 0
                elif tribute_count:
                    expected_tribute = tribute_count
                    exchange_context_known = True
            source_cards = [] if is_pass else list(event[2])
            face_cards = [source_id_to_face(card) for card in source_cards]
            label = {"pass": is_pass, "face_cards": face_cards, "source_cards": source_cards}
            emit_decision("P", seat, label, index)
            selected: list[int] = []
            for face in face_cards:
                selected.extend(_take_face_card(face_hands, physical_hands, seat, face))
            if not is_pass:
                played_faces.update(face_cards)
                last_player = seat
                last_cards = face_cards
                last_claim = _public_claim(face_cards, current_level)
            event_record = {"tag": "P", "seat": seat, **label, "cards": selected, "event_index": index, "deal_id": deal_id}
            # The public action is appended after the decision snapshot.
            public_history.append(_compact_public_event(event_record))
            play_history.append({"player": seat, "action": _local_physical_cards(face_cards), "claim": _public_claim(face_cards, current_level)})
            game_events_by_deal[deal_id].append(_compact_public_event(event_record))
            events_out.append(event_record)
            if not is_pass and not physical_hands[seat]:
                done_order.append(seat)
            expected_player = next(((seat + distance) % 4 for distance in range(1, 5) if physical_hands[(seat + distance) % 4]), None)
            continue
        if tag in ("T", "B"):
            if physical_hands is None or face_hands is None or deal_id < 0:
                raise SchemaError(f"{tag} before initialized deal at {index}")
            if len(event) != 4 or event[1] not in (0, 1, 2, 3) or event[2] not in (0, 1, 2, 3):
                raise SchemaError(f"invalid {tag} event at {index}: {event!r}")
            if play_history:
                raise SchemaError(f"{tag} transfer after play stage started at {index}")
            source_card = event[3]
            face = source_id_to_face(source_card)
            giver = int(event[1])
            recipient = int(event[2])
            if giver == recipient:
                raise SchemaError(f"{tag} transfer has identical giver and recipient at {index}")
            label = {"face_cards": [face], "source_cards": [source_card]}
            emit_decision(tag, giver, label, index)
            selected = _take_face_card(face_hands, physical_hands, giver, face)
            _give_physical_cards(face_hands, physical_hands, recipient, face, selected)
            if tag == "T":
                tribute_count += 1
            event_record = {"tag": tag, "from": giver, "to": recipient, **label, "cards": selected, "event_index": index, "deal_id": deal_id}
            public_history.append(_compact_public_event(event_record))
            game_events_by_deal[deal_id].append(_compact_public_event(event_record))
            events_out.append(event_record)
            continue
        if tag == "C":
            if len(event) != 1 or deal_id < 0 or last_player not in done_order or (last_player + 2) % 4 in done_order:
                raise SchemaError(f"invalid catch-wind C event at {index}")
            expected_player = (last_player + 2) % 4
            event_record = {"tag": "C", "player": expected_player, "event_index": index, "deal_id": deal_id}
            public_history.append(["C"])
            game_events_by_deal[deal_id].append(["C"])
            events_out.append(event_record)
            last_player = -1
            last_cards = []
            last_claim = None
            continue
        if tag == "V":
            events_out.append({"tag": "V", "event_index": index, "deal_id": deal_id, "result": result})
            continue
        # F only follows V in the source; retain it in full event output.
        if tag == "F":
            events_out.append({"tag": "F", "event_index": index, "score": _jsonable(event[1:]), "deal_id": deal_id})
            continue
        raise SchemaError(f"unknown event tag {tag!r} at {index}")

    # Keep event-level incompatibilities on the affected decision only.
    if any("return_current_level_conflicts_oracle" in record["quality_flags"] for record in decisions):
        quality.append("contains_return_current_level_conflicts_oracle")
    parsed = ParsedGame(
        match=match,
        game_id=f"{match.match_id}__game{game_index}",
        data_path=data_path,
        data_sha256=data_sha,
        game_index=game_index,
        events=events_out,
        decisions=decisions,
        result=result,
        quality_flags=sorted(set(quality)),
        trailing_events=[_jsonable(item) for item in trailing],
        deal_count=deal_id + 1,
    )
    # Full public history is stored once per game; initial hands are omitted.
    parsed.events.append({"tag": "PUBLIC_EVENTS", "events_by_deal": {str(key): value for key, value in sorted(game_events_by_deal.items())}})
    return parsed


def _quarantine_entry(match: MatchRecord, path: Path, reason: str, detail: str | None = None) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "match_id": match.match_id,
        "path": path.relative_to(match.extracted_path).as_posix(),
        "reason": reason,
        "data_sha256": sha256_file(path) if path.exists() and path.is_file() else None,
    }
    if detail:
        entry["detail"] = detail
    return entry


def run_etl(source_root: Path, output_root: Path, *, manifest_path: Path | None = None, max_file_bytes: int = DEFAULT_MAX_FILE_BYTES, max_events: int = DEFAULT_MAX_EVENTS) -> dict[str, Any]:
    source_root = source_root.resolve()
    output_root = output_root.resolve()
    if output_root == source_root or output_root.is_relative_to(source_root):
        raise ETLError("output directory must not be inside the read-only source root")
    if max_file_bytes <= 0 or max_events <= 0:
        raise ETLError("file and event limits must be positive")
    records, source_manifest_sha = _match_records(source_root, manifest_path)
    if not records:
        raise ETLError("no NUPT match packages found")
    split_by_match = assign_splits(records)
    output_root.mkdir(parents=True, exist_ok=True)
    paths = {split: output_root / f"{split}.jsonl" for split in ("train", "validation", "test")}
    games_path = output_root / "games.jsonl"
    quarantine_path = output_root / "quarantine.jsonl"
    manifest_out = output_root / "manifest.json"
    # All old outputs remain readable until their replacement is complete.
    # The manifest is replaced last and contains hashes of every data output.
    run_token = os.urandom(8).hex()
    final_paths = [*paths.values(), games_path, quarantine_path, manifest_out]
    temporary_paths = {path: path.with_name(f".{path.name}.{run_token}.tmp") for path in final_paths}
    handles = {split: temporary_paths[paths[split]].open("x", encoding="utf-8", newline="\n") for split in paths}
    games_handle = temporary_paths[games_path].open("x", encoding="utf-8", newline="\n")
    quarantine_handle = temporary_paths[quarantine_path].open("x", encoding="utf-8", newline="\n")
    counts: Counter[str] = Counter({key: 0 for key in ("games", "deals", "decisions", "quarantined_files", "stage_play", "stage_tribute", "stage_return")})
    quality_counts: Counter[str] = Counter()
    game_quality_counts: Counter[str] = Counter()
    trailing_event_counts: Counter[str] = Counter()
    claim_status_counts: Counter[str] = Counter()
    play_exchange_context_counts: Counter[str] = Counter()
    decision_split_counts: Counter[str] = Counter()
    match_counts: Counter[str] = Counter()
    source_files: list[dict[str, Any]] = []
    try:
        for match in records:
            split = split_by_match[match.match_id]
            selected, duplicate_quarantine = _select_game_files(match, max_file_bytes=max_file_bytes, max_events=max_events)
            for item in duplicate_quarantine:
                path = match.extracted_path / item["path"]
                quarantine_handle.write(json.dumps({**item, "match_id": match.match_id, "data_sha256": sha256_file(path) if path.exists() else None}, ensure_ascii=False, sort_keys=True) + "\n")
                counts["quarantined_files"] += 1
            match_counts[split] += 1
            for game_index, data_path in selected:
                try:
                    parsed = _parse_game(match, game_index, data_path, max_file_bytes=max_file_bytes, max_events=max_events)
                except (ETLError, OSError, pickle.PickleError) as exc:
                    quarantine = _quarantine_entry(match, data_path, "parse_failed", str(exc))
                    quarantine_handle.write(json.dumps(quarantine, ensure_ascii=False, sort_keys=True) + "\n")
                    counts["quarantined_files"] += 1
                    continue
                source_files.append(
                    {
                        "match_id": match.match_id,
                        "split": split,
                        "game_id": parsed.game_id,
                        "data_path": data_path.relative_to(match.extracted_path).as_posix(),
                        "source_relative_path": data_path.relative_to(source_root).as_posix(),
                        "data_sha256": parsed.data_sha256,
                        "archive_sha256": match.archive_sha256,
                        "deal_count": parsed.deal_count,
                    }
                )
                games_handle.write(
                    json.dumps(
                        {
                            "schema": "njupt-game-public-v1",
                            "match_id": match.match_id,
                            "game_id": parsed.game_id,
                            "game_index": game_index,
                            "result": parsed.result,
                            "quality_flags": parsed.quality_flags,
                            "events": parsed.events,
                            "trailing_events": parsed.trailing_events,
                        },
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    )
                    + "\n"
                )
                for decision in parsed.decisions:
                    handles[split].write(json.dumps(decision, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")
                counts["games"] += 1
                counts["deals"] += parsed.deal_count
                counts["decisions"] += len(parsed.decisions)
                decision_split_counts[split] += len(parsed.decisions)
                game_quality_counts.update(parsed.quality_flags)
                trailing_event_counts.update(event[0] for event in parsed.trailing_events if isinstance(event, list) and event)
                for decision in parsed.decisions:
                    counts[f"stage_{decision['stage']}"] += 1
                    claim_status_counts[decision["label"]["claim_status"]] += 1
                    if decision["stage"] == "play":
                        feature = decision["features"]
                        context_key = f"tribute={feature['tribute']},resist={str(feature['resist']).lower()},known={str(feature['exchange_context_known']).lower()}"
                        play_exchange_context_counts[context_key] += 1
                    for flag in decision.get("quality_flags", []):
                        quality_counts[flag] += 1
    finally:
        for handle in handles.values():
            handle.close()
        games_handle.close()
        quarantine_handle.close()
    manifest = {
        "schema": MANIFEST_SCHEMA,
        "decision_schema": SCHEMA_VERSION,
        "source_root_name": source_root.name,
        "source_manifest_sha256": source_manifest_sha,
        "source_manifest": "清单/njupt_archives.json" if source_manifest_sha else None,
        "split_policy": {"match_order": "match_id,archive_path ascending", "counts": {"train": 32, "validation": 7, "test": 7}, "actual": dict(match_counts)},
        "mapping": {
            "source_face_id": "source 2..53 rank-major by four suit blocks; 54 small joker; 55 big joker",
            "source_suit_to_botzone_suit": list(SOURCE_SUIT_TO_BOTZONE_SUIT),
            "botzone_face_ids": "canonical 0..53; physical duplicate is face+54",
            "features_private_hand": "own visible counts and local physical IDs only; local IDs do not depend on opponent allocation",
            "replay_physical_ids": "global physical entities are retained only in games.jsonl for ownership audit",
            "source_suit_certainty": "heart block 1 is verified; the permutation of the three non-heart suits is conventional and rules-invariant",
        },
        "limits": {"max_file_bytes": max_file_bytes, "max_events_per_pickle": max_events, "max_depth": DEFAULT_MAX_DEPTH, "max_sequence": DEFAULT_MAX_SEQUENCE, "max_string": DEFAULT_MAX_STRING, "max_nodes_per_event": DEFAULT_MAX_NODES},
        "counts": dict(counts),
        "quality_counts": dict(quality_counts),
        "game_quality_counts": dict(game_quality_counts),
        "trailing_event_counts": dict(trailing_event_counts),
        "claim_status_counts": dict(claim_status_counts),
        "play_exchange_context_counts": dict(play_exchange_context_counts),
        "decision_split_counts": dict(decision_split_counts),
        "feature_contract": {
            "play_history_limit": PLAY_HISTORY_LIMIT,
            "public_history_limit": 64,
            "played_face_counts": "all earlier P cards in current deal, excluding transfers",
            "tribute_resist": "prior public done_order gives 1/2 owed tributes; zero T before first P establishes resistance",
            "first_deal_exchange": "unknown source metadata is flagged exchange_context_known=false with neutral 0/false values",
        },
        "source_files": source_files,
        "quarantine_file": quarantine_path.name,
        "outputs": {**{split: paths[split].name for split in paths}, "games": games_path.name},
        "output_files": {
            path.name: {"sha256": sha256_file(temporary_paths[path]), "bytes": temporary_paths[path].stat().st_size}
            for path in final_paths if path != manifest_out
        },
    }
    temporary_paths[manifest_out].write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    for path in final_paths:
        os.replace(temporary_paths[path], path)
    return manifest


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path(r"D:\coding\RL\训练数据\南邮"), help="read-only NUPT data root")
    parser.add_argument("--output", type=Path, default=Path("data/processed/njupt"), help="processed JSONL output directory")
    parser.add_argument("--manifest", type=Path, default=None, help="optional archive manifest JSON path")
    parser.add_argument("--max-file-bytes", type=int, default=DEFAULT_MAX_FILE_BYTES)
    parser.add_argument("--max-events", type=int, default=DEFAULT_MAX_EVENTS)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    if not args.source.exists():
        print(f"source does not exist: {args.source}", file=sys.stderr)
        return 2
    manifest = run_etl(args.source, args.output, manifest_path=args.manifest, max_file_bytes=args.max_file_bytes, max_events=args.max_events)
    print(json.dumps({"output": str(args.output.resolve()), "counts": manifest["counts"], "quality_counts": manifest["quality_counts"], "split_counts": manifest["split_policy"]["actual"]}, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
