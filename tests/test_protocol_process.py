"""Process-level contract: one JSON line, no EOF dependency, clean termination.

The release package is intended for BotZone's long-running mode: the referee
keeps stdin open, so a response must be produced before EOF and the process
must remain available for the next request.  The harness closes stdin only
after observing the response so the child can terminate cleanly.
"""
import argparse
import json
from pathlib import Path
import queue
import subprocess
import threading


KEEP_RUNNING_MARKER = ">>>BOTZONE_REQUEST_KEEP_RUNNING<<<\n"


def decide(executable, payload):
    process = subprocess.Popen([str(executable)], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE, text=True, encoding="utf-8")
    output = queue.Queue()
    def read():
        output.put(process.stdout.readline())
    reader = threading.Thread(target=read, daemon=True)
    reader.start()
    try:
        process.stdin.write(payload + "\n")
        process.stdin.flush()  # Deliberately keep stdin open.
        line = output.get(timeout=2)
        assert process.poll() is None, "bot exited even though stdin remained open"
        assert line.endswith("\n")
        process.stdin.close()
        process.wait(timeout=1)
        assert process.returncode == 0
        assert process.stdout.read() == KEEP_RUNNING_MARKER
        assert process.stderr.read() == ""
        return json.loads(line)
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()
        if not process.stdin.closed:
            process.stdin.close()
        process.stdout.close()
        process.stderr.close()


def decide_incremental_keep_running(executable):
    """Exercise the BotZone marker plus its raw-request follow-up format."""
    process = subprocess.Popen([str(executable)], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE, text=True, encoding="utf-8")
    lines = queue.Queue()
    def read_lines():
        for line in process.stdout:
            lines.put(line)
    reader = threading.Thread(target=read_lines, daemon=True)
    reader.start()
    deal = {"stage": "deal", "your_id": 0, "deliver": list(range(27)),
            "global": {"level": "2", "tribute": 0, "first": None, "last": None}}
    play = {"stage": "play", "history": [[], [], [], []], "done": [], "pass_on": -1,
            "global": {"level": "2", "tribute": 0, "first": None, "last": None,
                       "resist": False, "tribute_cards": {}, "return_cards": {}}}
    try:
        process.stdin.write(json.dumps({"requests": [deal], "responses": []}) + "\n")
        process.stdin.flush()
        first = json.loads(lines.get(timeout=2))
        assert first["response"] == [], first
        assert lines.get(timeout=2) == KEEP_RUNNING_MARKER
        assert process.poll() is None

        # Long-running BotZone sends only the next request, not the historical
        # requests/responses envelope.  The C++ process must rebuild that
        # envelope internally before handing it to StateMirror.
        process.stdin.write(json.dumps(play) + "\n")
        process.stdin.flush()
        second = json.loads(lines.get(timeout=2))
        assert len(second["response"]) == 2 and second["response"][0], second
        assert "policy=model;" in second["debug"], second
        assert lines.get(timeout=2) == KEEP_RUNNING_MARKER
        assert process.poll() is None

        process.stdin.close()
        process.wait(timeout=2)
        assert process.returncode == 0
        assert process.stderr.read() == ""
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()
        if not process.stdin.closed:
            process.stdin.close()
        process.stdout.close()
        process.stderr.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bot", type=Path, default=Path("bin/oxbot"))
    parser.add_argument("--require-model", action="store_true")
    args = parser.parse_args()
    deal = {"stage": "deal", "your_id": 0, "deliver": list(range(27)),
            "global": {"level": "2", "tribute": 0, "first": None, "last": None}}
    out = decide(args.bot, json.dumps({"requests": [deal], "responses": []}))
    assert out["response"] == [], out
    play = {"stage": "play", "history": [[], [], [], []], "done": [], "pass_on": -1,
            "global": {"level": "2", "tribute": 0, "first": None, "last": None,
                       "resist": False, "tribute_cards": {}, "return_cards": {}}}
    out = decide(args.bot, json.dumps({"requests": [deal, play], "responses": [[]]}))
    assert len(out["response"]) == 2 and out["response"][0], out
    assert set(out["response"][0]).issubset(deal["deliver"]), out
    if args.require_model:
        assert "policy=model;" in out["debug"] and "model_sha=" in out["debug"], out
    # Malformed requests have no provably legal game action; require valid JSON
    # and an explicit diagnostic, not the false promise of universal legality.
    for malformed in ("{", "[]", "{\"requests\": []}", "{\"requests\": [null]}"):
        out = decide(args.bot, malformed)
        assert "response" in out and "debug" in out
    decide_incremental_keep_running(args.bot)
    print("7 process protocol checks passed (marker and incremental request included)")


if __name__ == "__main__":
    main()
