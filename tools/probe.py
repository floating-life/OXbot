"""Bounded IPC to the offline C++ core probe (never a BotZone long-running bot)."""
from __future__ import annotations
import json
import os
from pathlib import Path
import queue
import subprocess
import threading


class Probe:
    def __init__(self, executable, timeout=5):
        self.timeout = timeout
        command = [str(executable)]
        self.wsl_bridge = False
        # The repository's checked-in probes are Linux ELF binaries because
        # they are built for the WSL2 training environment.  When a Windows
        # Python process runs the parity/ETL tests, bridge such a binary via
        # WSL instead of failing with WinError 193.  Native Windows probes
        # continue to launch directly.
        if os.name == "nt":
            path = Path(executable).resolve()
            try:
                with path.open("rb") as stream:
                    is_elf = stream.read(4) == b"\x7fELF"
            except OSError:
                is_elf = False
            if is_elf and len(path.drive) == 2:
                wsl_path = "/mnt/" + path.drive[0].lower() + path.as_posix()[2:]
                command = ["wsl.exe", "-e", wsl_path]
                self.wsl_bridge = True
        self.process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                        text=True, encoding="utf-8", bufsize=1)
        self.queue = queue.Queue()
        def reader():
            for line in self.process.stdout:
                self.queue.put(line)
            self.queue.put(None)
        self.reader = threading.Thread(target=reader, daemon=True)
        self.reader.start()

    def call(self, **request):
        if self.wsl_bridge and isinstance(request.get("path"), str):
            model_path = Path(request["path"])
            if len(model_path.drive) == 2:
                request = dict(request)
                request["path"] = "/mnt/" + model_path.drive[0].lower() + model_path.as_posix()[2:]
        self.process.stdin.write(json.dumps(request, ensure_ascii=False, separators=(",", ":")) + "\n")
        self.process.stdin.flush()
        try:
            line = self.queue.get(timeout=self.timeout)
        except queue.Empty:
            raise TimeoutError("C++ core probe did not respond") from None
        if line is None:
            raise RuntimeError(f"C++ core probe exited: {self.process.poll()}")
        return json.loads(line)

    def close(self):
        if self.process.poll() is None:
            self.process.stdin.close()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
        self.process.stdout.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
