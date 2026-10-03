"""Exercise policy selection and explicit fallback with temporary test weights."""
from pathlib import Path
import json
import sys
import tempfile
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "train"), str(ROOT / "tools")]
from model import CandidateModel
from export_model import export
from probe import Probe


def main():
    torch.manual_seed(20261001)
    deal = {"stage": "deal", "your_id": 0, "deliver": list(range(27)),
            "global": {"level": "2", "tribute": 0, "first": None, "last": None}}
    play = {"stage": "play", "history": [[], [], [], []], "done": [], "pass_on": -1,
            "global": {"level": "2", "tribute": 0, "first": None, "last": None,
                       "resist": False, "tribute_cards": {}, "return_cards": {}}}
    observation = {"requests": [deal, play], "responses": [[]]}
    with tempfile.TemporaryDirectory(prefix="oxbot-policy-test-") as directory, Probe(ROOT / "bin/core_probe") as probe:
        weights = Path(directory) / "synthetic-do-not-publish.bin"
        manifest = export(CandidateModel().eval(), weights, {"purpose": "synthetic_policy_integration_test_only"})
        output = probe.call(command="bot", input=observation, model=str(weights))
        assert "policy=model;" in output["debug"], output
        assert "selection_strategy=raw-v1;" in output["debug"], output
        assert manifest["payload_sha256"][:12] in output["debug"], output
        move = output["response"]
        validated = probe.call(command="validate", hand=deal["deliver"], level="2", leading=True, move=move)
        assert validated["ok"], (move, validated)
        explicit_raw = probe.call(command="bot", input=observation, model=str(weights), strategy="raw")
        assert explicit_raw["response"] == output["response"], (output, explicit_raw)
        group = probe.call(command="bot", input=observation, model=str(weights), strategy="group-logmeanexp")
        assert "selection_strategy=group-logmeanexp-v1;" in group["debug"], group
        assert not group.get("error"), group
        manifest_group = Path(directory) / "synthetic-manifest-group.bin"
        export(CandidateModel().eval(), manifest_group,
               {"purpose": "synthetic_manifest_selection_test_only"}, "group-logmeanexp")
        manifest_default_group = probe.call(command="bot", input=observation, model=str(manifest_group))
        assert "selection_strategy=group-logmeanexp-v1;" in manifest_default_group["debug"], manifest_default_group
        assert "selection_source=manifest;" in manifest_default_group["debug"], manifest_default_group
        future = Path(directory) / "synthetic-model-v2.bin"
        future_manifest = export(CandidateModel().eval(), future,
                                 {"purpose": "synthetic_model_version_test_only"}, model_version="oxbot-model-v2")
        future_output = probe.call(command="bot", input=observation, model=str(future))
        assert "version=oxbot-model-v2;" in future_output["debug"], future_output
        assert future_manifest["model_version"] == "oxbot-model-v2"
        explicit_output = probe.call(command="bot", input=observation, model=str(future),
                                     candidate_version="package-candidate-v9")
        assert "version=package-candidate-v9;" in explicit_output["debug"], explicit_output
        invalid_explicit = probe.call(command="bot", input=observation, model=str(future),
                                      candidate_version="bad;injection")
        assert "version=oxbot-model-v1;" in invalid_explicit["debug"], invalid_explicit
        assert "bad;injection" not in invalid_explicit["debug"], invalid_explicit
        frozen = ROOT / "models/oxbot-bc-v1.bin"
        if frozen.is_file():
            default_frozen = probe.call(command="bot", input=observation, model=str(frozen))
            raw_frozen = probe.call(command="bot", input=observation, model=str(frozen), strategy="raw")
            assert default_frozen["response"] == raw_frozen["response"], (default_frozen, raw_frozen)
            assert "selection_strategy=raw-v1;" in default_frozen["debug"], default_frozen
            assert "selection_source=default;" in default_frozen["debug"], default_frozen
        for path in ("", str(Path(directory) / "missing.bin")):
            output = probe.call(command="bot", input=observation, model=path)
            assert "policy=rule_fallback;" in output["debug"], output
        corrupt = Path(directory) / "corrupt.bin"
        corrupt.write_bytes(b"OXGDQ001" + b"\x00"*10)
        output = probe.call(command="bot", input=observation, model=str(corrupt))
        assert "policy=rule_fallback;" in output["debug"], output
    print(json.dumps({"status": "passed", "scope": "model selection, final legality, missing/corrupt fallback; synthetic weights only"}))


if __name__ == "__main__":
    main()
