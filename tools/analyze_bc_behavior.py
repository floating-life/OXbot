"""Validation-only behavioral diagnostics; never selects or updates weights."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "train")]
from model import CandidateModel, ModelConfig
from train_bc import batches, shard_paths, sha256_file
from features import KINDS


def main():
    checkpoint_path = ROOT / "models/bc-v1/best.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model = CandidateModel(ModelConfig(**checkpoint["config"])).cuda().eval()
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    payload = hashlib.sha256()
    for tensor in model.state_dict().values():
        payload.update(tensor.detach().cpu().numpy().astype("<f4").tobytes())
    payload_sha = payload.hexdigest()
    manifest = json.loads((ROOT / "models/oxbot-bc-v1.manifest.json").read_text(encoding="utf-8"))
    if payload_sha != manifest["payload_sha256"]:
        raise ValueError("checkpoint payload does not match the frozen deployed model")
    torch.set_num_threads(4)
    stats = Counter()
    predicted, demonstrated = Counter(), Counter()
    examples = []
    with torch.inference_mode():
        for batch in batches(shard_paths(ROOT / "data/processed/bc-v1", "validation"), 64, 0, False):
            batch = {key: value.cuda() for key, value in batch.items()}
            positives = batch.pop("positives")
            scores = model(**batch)
            winners = scores.argmax(-1)
            for row, winner in enumerate(winners.tolist()):
                candidate = batch["actions"][row]
                chosen = candidate[winner]
                target = candidate[positives[row].nonzero()[0, 0]]
                lead = bool(batch["state"][row, 125])
                where = "lead" if lead else "follow"
                predicted[where + ":" + KINDS[int(chosen[108:120].argmax())]] += 1
                demonstrated[where + ":" + KINDS[int(target[108:120].argmax())]] += 1
                finishing = candidate[:, 127].bool() & batch["mask"][row]
                if finishing.any():
                    stats["has_finishing_candidate"] += 1
                    stats["model_finishes"] += int(bool(chosen[127]))
                    stats["demo_finishes"] += int(bool(target[127]))
                    if not chosen[127] and len(examples) < 8:
                        examples.append({"leading": lead, "hand_face_counts": (batch["state"][row,:54]*2).int().tolist(),
                            "chosen_face_counts": (chosen[:54]*2).int().tolist(), "chosen_kind": KINDS[int(chosen[108:120].argmax())],
                            "chosen_score": float(scores[row,winner]), "best_finish_score": float(scores[row].masked_fill(~finishing, float("-inf")).max())})
    report = {"scope": "fixed v1 checkpoint; validation behavioral diagnostics only",
              "checkpoint_sha256": sha256_file(checkpoint_path), "payload_sha256": payload_sha,
              "weights_loaded": True, "test_data_opened": False,
              "supersedes_invalid_random_init": "The previous version of this diagnostic omitted load_state_dict; its statistics were from an untrained network and must not be attributed to BC v1.",
              "stats": dict(stats),
              "predicted": dict(predicted), "demonstrated": dict(demonstrated), "missed_finish_examples": examples}
    (ROOT / "reports/bc-v1-behavior.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "missed_finish_examples"}))


if __name__ == "__main__":
    main()
