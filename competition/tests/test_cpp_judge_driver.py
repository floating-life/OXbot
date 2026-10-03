"""The C++ process gate must not mistake a legal rule fallback for model use."""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import judge_runner as runner


def cpp_output(stage="play", policy="model", status="model_selected",
               sha="ace1339157c5", fallback="0"):
    return {"response": [[], []],
            "debug": "version=fabledan-cpp;policy=%s;stage=%s;model_status=%s;"
                     "model_sha=%s;legality_fallback=%s" %
                     (policy, stage, status, sha, fallback)}


def test_cpp_play_requires_actual_model_selection_and_sha():
    fields = runner.check_response_model(cpp_output(), True, "play", "cpp")
    assert fields["model_sha"] == "ace1339157c5"
    for out in (cpp_output(policy="rule_fallback"),
                cpp_output(status="model_fallback:missing_tensor"),
                cpp_output(sha=""), cpp_output(sha="not-a-sha"),
                cpp_output(fallback="1"), cpp_output(stage="return")):
        with pytest.raises(RuntimeError):
            runner.check_response_model(out, True, "play", "cpp")
    out = cpp_output()
    out["error"] = "state_rebuild_failed"
    with pytest.raises(RuntimeError):
        runner.check_response_model(out, True, "play", "cpp")


def test_cpp_deal_deferral_and_rule_exchange_are_valid():
    for stage, status in (("deal", "deferred"), ("tribute", "ready"),
                          ("return", "ready")):
        out = cpp_output(stage=stage, policy="stage_rules", status=status, sha="")
        runner.check_response_model(out, True, stage, "cpp")
    with pytest.raises(RuntimeError):
        runner.check_response_model(cpp_output(stage="deal", policy="none"),
                                    True, "deal", "cpp")


def test_python_model_checks_are_still_strict():
    runner.check_response_model({"debug": "OXbot/FableDan model=transformer"}, True)
    for debug in ("model=rule", "model=transformer inference failed", "model=mlp unhandled:"):
        with pytest.raises(RuntimeError):
            runner.check_response_model({"debug": debug}, True)
    with pytest.raises(RuntimeError):
        runner.check_response_model(cpp_output(), True)


def test_cpp_driver_does_not_append_python_flag(monkeypatch):
    calls = []

    def fake_bot(*args, **kwargs):
        calls.append((args, kwargs))
        return object()

    monkeypatch.setattr(runner, "KeepProcBot", fake_bot)
    runner.make_bot("cpp:/tmp/oxbot --model data/version.fbd", None,
                    require_model=True)
    assert calls[-1][0][0] == "/tmp/oxbot --model data/version.fbd"
    assert calls[-1][1]["model_protocol"] == "cpp"
    runner.make_bot("keep:python bot.py", None, require_model=True)
    assert calls[-1][0][0] == "python bot.py --keep-running"
    assert "model_protocol" not in calls[-1][1]

