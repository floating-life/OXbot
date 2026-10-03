"""Check C++ BotzoneCompat return eligibility against the retrieved referee.

This is a focused compatibility golden, not a replacement for a full online
BotZone match.  The supplied corrected referee intentionally remains a
separate offline profile; this script imports the authenticated-page source
copy and compares its isValidReturn decisions with the explicit C++ variant.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
BASE_POINTORDER = list("234567890JQKA")
LEVELS = list("A234567890JQK")


class OfficialRejected(Exception):
    pass


def load_judge(path: Path):
    # The full referee imports numpy and also contains the online loop.  Only
    # extract the exact return-related definitions and their literal constants
    # so this golden can run with the small Windows Python install as well as
    # the 5080/WSL training environment.
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    selected = []
    constants = {"cardscale", "suitset", "jokers", "pointorder"}
    functions = {"num2Poker", "set_level", "isValidReturn"}
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
                isinstance(target, ast.Name) and target.id in constants
                for target in node.targets):
            selected.append(node)
        elif isinstance(node, ast.FunctionDef) and node.name in functions:
            selected.append(node)
    if not selected:
        raise RuntimeError(f"official_return_api_not_found:{path}")
    namespace = {}
    ast.fix_missing_locations(ast.Module(body=selected, type_ignores=[]))
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), "exec"), namespace)

    def reject(_player, _reason):
        raise OfficialRejected

    namespace["setError"] = reject
    api = SimpleNamespace(**namespace)
    # Keep the function globals available so each case can reset the mutable
    # pointorder list used by set_level.
    api._namespace = namespace
    return api


def official_accepts(judge, hand, card, level):
    # set_level mutates the module-global pointorder, so reset it for each
    # case to make the table independent of iteration order.
    judge._namespace["pointorder"] = BASE_POINTORDER.copy()
    judge.set_level(level)
    try:
        judge.isValidReturn(hand, card, level, 0)
    except OfficialRejected:
        return False
    return True


def card(rank_index: int) -> int:
    return rank_index * 4


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--judge", type=Path,
                        default=ROOT / "reports" / "official_judge_botzone_2026-10-02.py")
    parser.add_argument("--probe", type=Path, default=ROOT / "bin" / "core_probe")
    parser.add_argument("--report", type=Path,
                        default=ROOT / "reports" / "official_return_boundaries.json")
    args = parser.parse_args()

    sys.path.insert(0, str(ROOT / "tools"))
    from probe import Probe  # pylint: disable=import-outside-toplevel

    judge = load_judge(args.judge)
    # All natural faces are present so every boundary candidate is in hand.
    hand = list(range(52))
    cases = []
    mismatches = []
    with Probe(args.probe) as probe:
        for level in LEVELS:
            cxx = probe.call(command="return", hand=hand, level=level,
                             variant="botzone")["cards"]
            cxx_set = set(cxx)
            official_set = {
                physical for physical in hand
                if official_accepts(judge, hand, physical, level)
            }
            case = {"level": level,
                    "official_faces": sorted({value % 54 for value in official_set}),
                    "cxx_cards": sorted(cxx_set),
                    "match": cxx_set == official_set}
            cases.append(case)
            if cxx_set != official_set:
                mismatches.append(case)

    result = {
        "schema": "oxbot-official-return-boundaries-v1",
        "status": "passed" if not mismatches else "failed",
        "judge": str(args.judge.resolve()),
        "judge_sha256": hashlib.sha256(args.judge.read_bytes()).hexdigest(),
        "judge_normalized_sha256": hashlib.sha256(args.judge.read_bytes().rstrip(b"\r\n")).hexdigest(),
        "probe": str(args.probe.resolve()),
        "levels": len(LEVELS),
        "cases": cases,
        "mismatches": mismatches,
        "notes": [
            "Official isValidReturn is compared after set_level reorders pointorder.",
            "The attachment corrected referee is intentionally not used by this golden.",
            "A passed golden does not establish BotZone compiler, timing, or upload acceptance.",
        ],
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n",
                           encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "cases"},
                     ensure_ascii=False))
    return 0 if not mismatches else 1


if __name__ == "__main__":
    raise SystemExit(main())
