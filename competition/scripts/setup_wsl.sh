#!/usr/bin/env bash
set -Eeuo pipefail

# One-time setup for WSL2 + RTX 5080.  Prefer the existing project runtime
# documented by the parent OXbot project; if it is absent, create a local
# .venv rather than silently using a system Python with the wrong CUDA wheel.

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="${OXBOT_VENV:-/home/ggcle/.venvs/oxbot}"
PYTHON="${OXBOT_PYTHON:-}"

die() { printf 'SETUP FAILED: %s\n' "$*" >&2; exit 1; }
trap 'printf "SETUP FAILED near line %s\n" "$LINENO" >&2' ERR

if [[ -z "$PYTHON" && -x "$VENV/bin/python" ]]; then
    PYTHON="$VENV/bin/python"
fi
if [[ -z "$PYTHON" && -x "$ROOT/.venv/bin/python" ]]; then
    PYTHON="$ROOT/.venv/bin/python"
fi
if [[ -z "$PYTHON" ]]; then
    BOOTSTRAP="$(command -v python3.12 || command -v python3 || true)"
    [[ -n "$BOOTSTRAP" ]] || die "python3.12/python3 not found"
    VENV="$ROOT/.venv"
    "$BOOTSTRAP" -m venv --system-site-packages "$VENV"
    PYTHON="$VENV/bin/python"
fi
[[ -x "$PYTHON" ]] || die "Python executable not found: $PYTHON"

if [[ "${OXBOT_UPGRADE_PIP:-0}" == 1 ]]; then
    "$PYTHON" -m pip install --upgrade pip
fi
if ! "$PYTHON" -c 'import numpy' >/dev/null 2>&1; then
    "$PYTHON" -m pip install 'numpy>=1.24'
fi

# requirements.txt intentionally keeps torch optional for Botzone packaging.
# Install the CUDA 12.8 wheel only when this runtime does not already expose
# a working CUDA build (the checked-in WSL env is torch 2.11 + cu128).
if ! "$PYTHON" -c 'import torch; assert torch.cuda.is_available()' >/dev/null 2>&1; then
    "$PYTHON" -m pip install torch --index-url https://download.pytorch.org/whl/cu128
fi

"$PYTHON" - <<'PY'
import sys
import torch
assert torch.cuda.is_available(), "CUDA is unavailable"
print("python", sys.version.split()[0])
print("torch", torch.__version__, "cuda", torch.version.cuda)
print("gpu", torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))
x = torch.randn(2048, 2048, device="cuda", dtype=torch.bfloat16)
print("bf16 matmul ok", bool((x @ x).abs().sum() > 0))
PY

cd "$ROOT"
"$PYTHON" tools/check_environment.py --micro-batch 256
"$PYTHON" tests/test_all.py
"$PYTHON" tests/test_judge_compat.py
printf 'SETUP OK using %s\n' "$PYTHON"
