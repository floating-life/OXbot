"""Produce a single UTF-8 C++17 submission from the independent local engine."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import re
import struct

ROOT = Path(__file__).resolve().parents[1]
ORDER = [
    "core/include/oxbot/mini_json.hpp", "core/include/oxbot/card.hpp", "core/include/oxbot/rules.hpp",
    "core/include/oxbot/state.hpp", "core/include/oxbot/network.hpp", "core/include/oxbot/fabledan_network.hpp", "core/include/oxbot/features.hpp", "core/include/oxbot/fabledan_features.hpp", "core/include/oxbot/policy.hpp", "core/include/oxbot/protocol.hpp",
    "core/include/oxbot/fabledan_candidates.hpp",
    "core/src/rules.cpp", "core/src/state.cpp", "core/src/network.cpp", "core/src/fabledan_network.cpp", "core/src/features.cpp", "core/src/fabledan_features.cpp", "core/src/fabledan_candidates.cpp", "core/src/policy.cpp", "core/src/protocol.cpp", "botzone/main.cpp",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "dist" / "oxbot.cpp")
    parser.add_argument("--embed-model", type=Path, help="include exact exported binary bytes inside the source")
    parser.add_argument("--model-path", help="verified BotZone storage path for an external binary")
    parser.add_argument("--strategy", choices=("raw", "raw-pass-bias", "group-logmeanexp"), default="raw",
                        help="selection strategy compiled into the package; raw is the release default")
    parser.add_argument("--candidate-version", default="",
                        help="explicit package diagnostic identity; empty derives it from the model header")
    args = parser.parse_args()
    if args.embed_model and args.model_path:
        raise ValueError("choose embedded model or external model path")
    if args.candidate_version and (len(args.candidate_version) > 96 or
                                   any(not (char.isascii() and (char.isalnum() or char in ".-_") )
                                       for char in args.candidate_version)):
        raise ValueError("invalid candidate version")
    chunks = ["// OXbot C++17 ordinary JSON. Candidate build; platform verification remains mandatory.\n",
              "#define OXBOT_POLICY_STRATEGY " + json.dumps(args.strategy) + "\n",
              "#define OXBOT_CANDIDATE_VERSION " + json.dumps(args.candidate_version) + "\n"]
    model_info = None
    if args.embed_model:
        raw_model = args.embed_model.read_bytes()
        if raw_model[:8] != b"OXGDQ001" or len(raw_model) < 12:
            raise ValueError("invalid model binary magic")
        length = struct.unpack("<I", raw_model[8:12])[0]
        if length > 65536 or 12 + length >= len(raw_model):
            raise ValueError("invalid model header")
        header = json.loads(raw_model[12:12+length])
        payload_sha = hashlib.sha256(raw_model[12+length:]).hexdigest()
        if header["payload_sha256"] != payload_sha:
            raise ValueError("model checksum mismatch")
        model_info = {"mode": "embedded", "file_sha256": hashlib.sha256(raw_model).hexdigest(),
                      "payload_sha256": payload_sha, "bytes": len(raw_model),
                      "provenance": header.get("provenance", {})}
        chunks.append('#define OXBOT_EMBEDDED_MODEL 1\n#define OXBOT_MODEL_PATH ":embedded:"\n')
        chunks.append("static const unsigned char oxbot_embedded_model[] = {\n")
        for i in range(0, len(raw_model), 32):
            chunks.append(",".join(str(value) for value in raw_model[i:i+32]) + ",\n")
        chunks.append("};\n")
    elif args.model_path:
        if any(ord(c) < 32 for c in args.model_path):
            raise ValueError("model path contains control characters")
        chunks.append("#define OXBOT_MODEL_PATH " + json.dumps(args.model_path, ensure_ascii=True) + "\n")
        model_info = {"mode": "external", "path": args.model_path}
        # When the paired file is present locally, record its identity in the
        # manifest.  BotZone still resolves the relative path from data/ at
        # runtime; the hash here is an audit binding, not a source embedding.
        external = Path(args.model_path)
        if external.is_file():
            raw_external = external.read_bytes()
            if raw_external[:8] not in (b"OXGDQ001", b"FBDN001\0", b"FBDN002\0"):
                raise ValueError("external model has unsupported magic")
            model_info.update({"file_sha256": hashlib.sha256(raw_external).hexdigest(),
                               "bytes": len(raw_external),
                               "magic": raw_external[:8].rstrip(b"\0").decode("ascii")})
    sources = {}
    for relative in ORDER:
        raw = (ROOT / relative).read_bytes()
        sources[relative] = hashlib.sha256(raw).hexdigest()
        source = raw.decode("utf-8-sig")
        source = re.sub(r'^\s*#pragma once\s*$', "", source, flags=re.MULTILINE)
        source = re.sub(r'^\s*#include "oxbot/[^"\n]+"\s*$', "", source, flags=re.MULTILINE)
        chunks.append(f"\n// BEGIN {relative}\n{source}\n// END {relative}\n")
    content = "".join(chunks).encode("utf-8")
    if len(content) >= 4_000_000:
        raise RuntimeError(f"source exceeds conservative 4,000,000 byte limit: {len(content)}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(content)
    manifest = {"file": args.output.name, "bytes": len(content), "sha256": hashlib.sha256(content).hexdigest(),
                "source_hashes": sources, "language": "C++17", "interaction": "ordinary_json",
                "model": model_info, "release_eligible": False,
                "reason": "this packager does not establish training, gameplay or platform acceptance",
                "selection_strategy": args.strategy,
                "selection_strategy_version": args.strategy + "-v1",
                "candidate_version": args.candidate_version or None}
    args.output.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    summary = {k: v for k, v in manifest.items() if k not in ("source_hashes", "model")}
    summary["model"] = {k: v for k, v in model_info.items() if k != "provenance"} if model_info else None
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
