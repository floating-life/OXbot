"""Load the supplied judge as an isolated, fresh oracle for each invocation.

The only loader adaptation removes its unused numpy import (verified by AST).
No rule, control flow, exception, or state-mutating function is rewritten.
The original source remains untouched and its exact SHA is recorded.
"""
from __future__ import annotations

import ast
import contextlib
import copy
import hashlib
import io
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_JUDGE = ROOT / "裁判代码-修正版.py"


class JudgeRejected(Exception):
    pass


class Oracle:
    def __init__(self, path: Path = DEFAULT_JUDGE):
        self.path = Path(path).resolve()
        source = self.path.read_bytes()
        self.sha256 = hashlib.sha256(source).hexdigest()
        tree = ast.parse(source.decode("utf-8-sig"), filename=str(self.path))
        numpy_names = set()
        for statement in tree.body:
            if isinstance(statement, ast.Import):
                for alias in statement.names:
                    if alias.name == "numpy":
                        numpy_names.add(alias.asname or alias.name)
        used = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
        if not numpy_names.intersection(used):
            tree.body = [stmt for stmt in tree.body if not (
                isinstance(stmt, ast.Import) and len(stmt.names) == 1 and stmt.names[0].name == "numpy"
            )]
        self.code = compile(tree, str(self.path), "exec")

    def module(self, level: str | None = None):
        scope = {"__name__": "oxbot_readonly_judge", "__file__": str(self.path)}
        exec(self.code, scope)
        if level is not None:
            scope["set_level"]("0" if level == "10" else level)
        return scope

    def step(self, full_input):
        scope = self.module()
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            try:
                scope["main"](copy.deepcopy(full_input))
            except SystemExit:
                pass
        lines = stdout.getvalue().splitlines()
        if len(lines) != 1:
            raise RuntimeError(f"judge output has {len(lines)} lines: {stdout.getvalue()[:400]}")
        return json.loads(lines[0])

    def rules(self, level="2"):
        scope = self.module(level)
        def reject(player, reason):
            raise JudgeRejected(f"{player}:{reason}")
        # Function-level rule checks raise a local exception rather than exiting
        # the test process. Game replays use step(), which does NOT substitute it.
        scope["setError"] = reject
        return scope
