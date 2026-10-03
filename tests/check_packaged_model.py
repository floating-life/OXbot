"""Compile an embedded-model single file and verify real process inference."""
from pathlib import Path
import json
import subprocess
import sys
import tempfile
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "train"), str(ROOT / "tools")]
from model import CandidateModel
from export_model import export


def main():
    torch.manual_seed(20261001)
    with tempfile.TemporaryDirectory(prefix="oxbot-package-test-") as directory:
        folder = Path(directory)
        model = folder / "synthetic.bin"
        exported = export(CandidateModel().eval(), model, {"purpose": "synthetic_package_test_only"})
        source, bot = folder / "oxbot.cpp", folder / "oxbot"
        subprocess.run([sys.executable, str(ROOT / "tools/amalgamate.py"), "--output", str(source),
                        "--embed-model", str(model)], check=True, capture_output=True, text=True)
        subprocess.run(["g++", "-std=c++17", "-O2", "-Wall", "-Wextra", "-Wpedantic", str(source), "-o", str(bot)],
                       check=True, timeout=90, capture_output=True, text=True)
        global_info = {"level": "2", "tribute": 0, "first": None, "last": None,
                       "resist": False, "tribute_cards": {}, "return_cards": {}}
        payload = {"requests": [{"stage": "deal", "your_id": 0, "deliver": list(range(27)), "global": global_info},
                                {"stage": "play", "history": [[], [], [], []], "done": [], "pass_on": -1,
                                 "global": global_info}], "responses": [[]]}
        # An unrelated cwd proves the model is included in the executable.
        result = subprocess.run([str(bot)], input=json.dumps(payload)+"\n", text=True, capture_output=True,
                                cwd=folder, timeout=2, check=True)
        assert result.stderr == "" and len(result.stdout.splitlines()) == 1
        response = json.loads(result.stdout)
        assert "policy=model;" in response["debug"], response
        assert exported["payload_sha256"][:12] in response["debug"], response
        manifest = json.loads(source.with_suffix(".manifest.json").read_text())
        assert manifest["bytes"] < 4_000_000 and not manifest["release_eligible"]
        report = {"status": "passed", "source_bytes": manifest["bytes"], "model_bytes": exported["bytes"],
                  "scope": "synthetic embedded-model build, standalone process and SHA; not a gameplay release"}
        (ROOT / "reports").mkdir(exist_ok=True)
        (ROOT / "reports/embedded_package.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps(report))


if __name__ == "__main__":
    main()
