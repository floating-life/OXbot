"""Real-replay adaptation must preserve public information and split isolation."""
import copy
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import train_real as real
from fabledan.combos import classify_claim
from fabledan.encode import encode_decision


def record(*, game="g-train", match="match-train", digest="a" * 64, event=3,
           hand=None, label=None, leading=True, previous=None, claim=None):
    hand = [5, 6, 9] if hand is None else hand
    return {
        "schema": "njupt-decision-v3", "game_id": game, "match_id": match,
        "deal_id": 0, "event_index": event, "stage": "play",
        "provenance": {"data_sha256": digest},
        "features": {"own_hand": hand, "seat": 0, "level_label": "2",
                     "remaining_counts": [len(hand), 27, 27, 27],
                     "leading": leading, "last_cards": previous or [], "last_claim": claim,
                     "history": [], "opponent_hands": "must not be used"},
        "label": {"cards": [6] if label is None else label, "claim": "must not be used"},
        "result": {"winner": "must not be used"},
    }


def game_row(game="g-train", match="match-train", *, prior=None, future=None):
    prior = [{"tag": "P", "deal_id": 0, "event_index": 2, "seat": 1,
              "face_cards": [13], "cards": [67]}] if prior is None else prior
    events = [{"tag": "R", "deal_id": 0, "event_index": 0, "current_level": 2}, *prior,
              {"tag": "P", "deal_id": 0, "event_index": 3, "seat": 0, "face_cards": [6]},
              *([] if future is None else future)]
    compact = []
    for item in events:
        if item["tag"] == "R":
            compact.append(["R", 2, 2, 2])
        elif item["tag"] == "P":
            compact.append(["P", item["seat"], item["face_cards"], not item["face_cards"]])
        elif item["tag"] in ("T", "B"):
            compact.append([item["tag"], item["from"], item["to"], item["face_cards"]])
        else:
            compact.append([item["tag"]])
    events.append({"tag": "PUBLIC_EVENTS", "events_by_deal": {"0": compact}})
    return {"schema": "njupt-game-public-v1", "game_id": game, "match_id": match,
            "events": events, "result": "must not be used", "private_hands": "must not be used"}


def make_history(tmp_path, row=None):
    path = tmp_path / "games.jsonl"
    path.write_text(json.dumps(game_row() if row is None else row) + "\n", encoding="utf-8")
    return real.PublicHistoryIndex(path, {"g-train": {"match_id": "match-train"}})


def make_source(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    files = []
    games = []
    counts = {}
    for split, digest in (("train", "a" * 64), ("validation", "b" * 64), ("test", "c" * 64)):
        rows = [record(game=f"g-{split}", match=f"match-{split}", digest=digest)]
        if split == "train":
            exchange = copy.deepcopy(rows[0])
            exchange.update(event_index=1, stage="tribute")
            # An ambiguous heart-level pair can represent several distinct
            # comparison keys; the adapter must account for its rejection.
            uncertain = copy.deepcopy(rows[0])
            uncertain.update(event_index=4)
            uncertain["features"].update(leading=False, last_cards=[5, 9, 13, 17, 4], last_claim=None)
            rows = [exchange, rows[0], uncertain]
        (source / f"{split}.jsonl").write_text("".join(json.dumps(item) + "\n" for item in rows), encoding="utf-8")
        counts[split] = len(rows)
        row = game_row(f"g-{split}", f"match-{split}")
        if split == "train":
            row["events"].insert(1, {"tag": "T", "deal_id": 0, "event_index": 1,
                                      "from": 0, "to": 1, "face_cards": [53]})
            row["events"][-1]["events_by_deal"]["0"].insert(1, ["T", 0, 1, [53]])
            row["events"].insert(-1, {"tag": "P", "deal_id": 0, "event_index": 4,
                                       "seat": 0, "face_cards": [6]})
            row["events"][-1]["events_by_deal"]["0"].append(["P", 0, [6], False])
        games.append(row)
        files.append({"game_id": f"g-{split}", "match_id": f"match-{split}", "split": split,
                      "data_sha256": digest, "archive_sha256": digest})
    (source / "games.jsonl").write_text("".join(json.dumps(row) + "\n" for row in games), encoding="utf-8")
    manifest = {"decision_schema": "njupt-decision-v3", "decision_split_counts": counts,
                "source_files": files,
                "output_files": {name: {"sha256": real.sha256_file(source / name)}
                                 for name in ("train.jsonl", "validation.jsonl", "test.jsonl", "games.jsonl")}}
    (source / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return source


def test_adapter_matches_live_encoder_for_known_public_prefix(tmp_path):
    history = make_history(tmp_path)
    sample = record()
    encoded, _ = real.encode_record(sample, history)
    obs, _ = real.observation_from_record(sample)
    obs["legal"] = real.gen_moves(obs["hand"], obs["level"], obs["lead"])
    obs["events"] = [("play", 1, classify_claim([13], [13], obs["level"]))]
    tokens, actions = encode_decision(obs)
    assert encoded["tokens"].tolist() == tokens
    assert all(any(np.array_equal(candidate, row) for row in actions) for candidate in encoded["actions"])
    # The demo chose a different suit than the generator's canonical single.
    # Both have exactly the same competition features, so its row is positive.
    assert encoded["positives"].sum() == 1
    positive = encoded["actions"][encoded["positives"]][0]
    assert positive[49 + 1] == pytest.approx(0.25)


def test_hidden_results_and_future_actions_do_not_change_inputs(tmp_path):
    history = make_history(tmp_path)
    first, _ = real.encode_record(record(), history)
    sample = record()
    sample["result"] = {"winner": 3, "all_hands": list(range(108))}
    sample["features"]["opponent_hands"] = list(range(108))
    sample["label"]["claim"] = [53]  # unsupported alleged private claim ignored
    other_dir = tmp_path / "other"
    other_dir.mkdir()
    future = [{"tag": "P", "deal_id": 0, "event_index": 5, "seat": 2, "face_cards": [53]}]
    second, _ = real.encode_record(sample, make_history(other_dir, game_row(future=future)))
    for key in ("tokens", "actions", "positives"):
        np.testing.assert_array_equal(first[key], second[key])


def test_source_wildcard_substitution_maps_to_live_canonical_semantics(tmp_path):
    # The source deliberately spends a wildcard in a triple despite holding
    # three natural cards of the same rank. Live gen_moves emits the natural
    # representative; the shared type/key/claim semantics remain a BC target.
    sample = record(hand=[4, 5, 6, 7, 9], label=[4, 5, 6])
    encoded, _ = real.encode_record(sample, make_history(tmp_path))
    positives = encoded["actions"][encoded["positives"]]
    assert len(positives) == 1
    assert positives[0, 38 + 3] == 1.0  # TRIPLE action type
    assert positives[0, 49 + 1] == 0.75
    assert positives[0, 65] == 0.0  # canonical action spends no wildcard


def test_history_reconstructs_more_than_the_etl_snapshot_window(tmp_path):
    events = [{"tag": "R", "deal_id": 0, "event_index": 0, "current_level": 2},
              {"tag": "T", "deal_id": 0, "event_index": 1, "from": 2, "to": 0, "face_cards": [53]}]
    events.extend({"tag": "P", "deal_id": 0, "event_index": index + 2, "seat": index % 4,
                   "face_cards": []} for index in range(160))
    events.append({"tag": "P", "deal_id": 0, "event_index": 162, "seat": 0, "face_cards": [6]})
    compact = [["R", 2, 2, 2], ["T", 2, 0, [53]]]
    compact.extend(["P", index % 4, [], True] for index in range(160))
    compact.append(["P", 0, [6], False])
    events.append({"tag": "PUBLIC_EVENTS", "events_by_deal": {"0": compact}})
    row = {"game_id": "g-train", "match_id": "match-train", "events": events}
    encoded, stats = real.encode_record(record(event=162), make_history(tmp_path, row))
    assert len(encoded["tokens"]) == 2 + 3 + 160 * 2
    assert real.TRIBUTE_TOK in encoded["tokens"]
    assert stats["full_history_window_truncated"] == 0


def test_full_history_includes_transfers_and_marks_unknown_claim(tmp_path):
    prior = [{"tag": "T", "deal_id": 0, "event_index": 1, "from": 2, "to": 0, "face_cards": [53]},
             {"tag": "P", "deal_id": 0, "event_index": 2, "seat": 1, "face_cards": [4, 5]}]
    history = make_history(tmp_path, game_row(prior=prior))
    encoded, stats = real.encode_record(record(), history)
    assert real.TRIBUTE_TOK in encoded["tokens"]
    assert real.UNKNOWN_CLAIM_TOK in encoded["tokens"]
    assert stats["unknown_claim_history"] == 1


def test_pending_prior_claim_never_becomes_an_asserted_claim():
    sample = record(leading=False, previous=[5, 9, 13, 17, 4], claim=None)
    with pytest.raises(real.SkipRecord, match="multiple lead") as error:
        real.observation_from_record(sample)
    assert error.value.reason == "uncertain_previous"
    # A single printed wildcard has a unique legal comparison key. Recovering
    # that invariant is safe but is always counted explicitly.
    sample["features"]["last_cards"] = [4]
    _, stats = real.observation_from_record(sample)
    assert stats["unknown_previous_semantics_recovered"] == 1


def test_prepare_accounts_for_every_source_record_and_never_reads_test(tmp_path, monkeypatch):
    source = make_source(tmp_path)
    real_open = Path.open

    def guarded_open(path, *args, **kwargs):
        if path.name == "test.jsonl":
            raise AssertionError("development preparation opened held-out test")
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guarded_open)
    output = tmp_path / "prepared"
    manifest = real.prepare(source, output, shard_size=2)
    assert manifest["status"] == "complete"
    assert manifest["test_used"] is False
    counts = manifest["splits"]["train"]["counts"]
    assert counts["input_records"] == 3
    assert counts["accepted_records"] == 2
    assert counts["rule_handled_exchange"] == 1
    assert counts["ntp_only_uncertain_previous"] == 1
    assert counts["bc_supported_records"] == 1
    assert counts["input_records"] == counts["accepted_records"] + counts["rule_handled_exchange"]
    _, paths = real.load_manifest(output, ("train", "validation"))
    batches = list(real.batches(paths["train"], batch_size=1, candidate_budget=1, seed=4))
    assert len(batches) == 2
    assert sum(len(batch["ids"]) for batch in batches) == 2
    auxiliary = next(batch for batch in batches if not batch["bc_supported"][0])
    assert not auxiliary["positives"].any()


def test_reject_provenance_and_cross_split_match_changes(tmp_path):
    source = make_source(tmp_path)
    manifest_path = source / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["source_files"][1]["match_id"] = "match-train"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(real.DataError, match="crosses"):
        real.prepare(source, tmp_path / "prepared")


def test_batching_visits_all_rows_even_when_budget_is_exceeded(tmp_path):
    writer = real.ShardWriter(tmp_path, "train", size=2)
    for index, count in enumerate((3, 1, 4)):
        writer.add({"id": f"id-{index}", "tokens": np.array([1, 3], dtype=np.uint8),
                    "actions": np.zeros((count, real.FEAT_DIM), dtype=np.float32),
                    "positives": np.array([True] + [False] * (count - 1))})
    writer.flush()
    paths = [tmp_path / item["path"] for item in writer.files]
    output = list(real.batches(paths, batch_size=2, candidate_budget=2, seed=13))
    observed = [str(value) for batch in output for value in batch["ids"]]
    assert len(observed) == len(set(observed)) == 3
    assert set(observed) == {"id-0", "id-1", "id-2"}


def test_marginal_loss_retains_all_positive_claims():
    torch = pytest.importorskip("torch")
    scores = torch.tensor([0.0, 1.0, 2.0, 1.0], requires_grad=True)
    positives = torch.tensor([True, False, True, True])
    losses = real.marginal_loss(scores, positives, [3, 1])
    expected = torch.logsumexp(scores[:3], 0) - torch.logsumexp(scores[[0, 2]], 0)
    torch.testing.assert_close(losses, torch.stack((expected, scores.new_zeros(()))))
    losses.mean().backward()
    assert torch.isfinite(scores.grad).all()
    auxiliary_scores = torch.tensor([1.0], requires_grad=True)
    auxiliary = real.marginal_loss(auxiliary_scores, torch.tensor([False]), [1], torch.tensor([False]))
    torch.testing.assert_close(auxiliary, torch.zeros(1))
    auxiliary.sum().backward()
    torch.testing.assert_close(auxiliary_scores.grad, torch.zeros(1))


def test_training_never_opens_held_out_shards_and_visits_every_row(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    from argparse import Namespace
    source = make_source(tmp_path)
    data = tmp_path / "prepared"
    manifest = real.prepare(source, data, splits=("train", "validation", "test"), shard_size=2)
    # A tiny compatible config exercises a real forward/backward without
    # turning this split-isolation test into a GPU or throughput test.
    from fabledan.model_torch import FableDanNet, ModelConfig, save_ckpt
    initial = tmp_path / "initial.pt"
    cfg = ModelConfig(d_model=8, n_blocks=1, n_heads=1, qk_dim=4, v_dim=4,
                      ffn_hidden=16, hand_hidden=16, n_hand_layers=1,
                      q_hidden=16, n_q_layers=1)
    save_ckpt(FableDanNet(cfg), None, {}, str(initial))
    real_open = Path.open

    def guarded_open(path, *args, **kwargs):
        if path.name == "test.jsonl" or "test" in path.parts and path.suffix == ".npz":
            raise AssertionError("real training opened held-out data")
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guarded_open)
    args = Namespace(data=data, source=source, out=tmp_path / "trained", epochs=1,
                     device="cpu", seed=17, threads=1, fp32=True, n_blocks=1,
                     ntp_weight=0.02, lr=1e-3, resume=None, init=initial,
                     batch=2, candidate_budget=4, candidate_chunk=2, log_every=100)
    report = real.train(args)
    assert report["status"] == "complete"
    assert report["test_used"] is False
    assert report["epochs"][0]["train_samples"] == 2
    assert report["epochs"][0]["train_bc_samples"] == 1
    assert report["epochs"][0]["train_public_ntp_only_samples"] == 1
    assert report["epochs"][0]["validation"]["samples"] == 1
    assert report["expected_samples"]["train"] == manifest["splits"]["train"]["counts"]["accepted_records"]
    checkpoint = torch.load(args.out / "best.pt", weights_only=False)
    assert checkpoint["meta"]["training_kind"] == "real_bc"
    assert checkpoint["meta"]["cycle"] == 0
    assert (args.out / "best.npz").is_file()
    args.init, args.resume, args.epochs = None, args.out / "latest.pt", 2
    args.lr = args.ntp_weight = args.seed = None
    resumed = real.train(args)
    assert len(resumed["epochs"]) == 2
    assert resumed["training_args"]["lr"] == 1e-3
    assert resumed["training_args"]["ntp_weight"] == 0.02
    assert resumed["training_args"]["seed"] == 17
    assert resumed["best_epoch"] in (1, 2)
