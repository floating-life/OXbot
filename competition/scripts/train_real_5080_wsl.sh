#!/usr/bin/env bash
set -Eeuo pipefail

# Finite real-replay training; prepare --include-test before this launcher.
# Training itself opens train + validation only. Examples:
#   scripts/train_real_5080_wsl.sh
#   scripts/train_real_5080_wsl.sh --epochs 12

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="${OXBOT_VENV:-/home/ggcle/.venvs/oxbot}"
if [[ -n "${OXBOT_PYTHON:-}" ]]; then
    PYTHON="$OXBOT_PYTHON"
elif [[ -x "$VENV/bin/python" ]]; then
    PYTHON="$VENV/bin/python"
elif [[ -x "$ROOT/.venv/bin/python" ]]; then
    PYTHON="$ROOT/.venv/bin/python"
else
    PYTHON="$(command -v python3 || true)"
fi
[[ -n "$PYTHON" && -x "$PYTHON" ]] || {
    printf 'REAL TRAINING LAUNCH FAILED: Python runtime not found; run setup_wsl.sh.\n' >&2
    exit 1
}

OUT="${OXBOT_REAL_OUT:-ckpts/real-v2}"
args=( train_real.py train
       --out "$OUT"
       --epochs "${OXBOT_REAL_EPOCHS:-8}"
       --device "${OXBOT_REAL_DEVICE:-cuda:0}"
       --batch "${OXBOT_REAL_BATCH:-128}"
       --candidate-budget "${OXBOT_REAL_CANDIDATE_BUDGET:-8192}"
       --candidate-chunk "${OXBOT_REAL_CANDIDATE_CHUNK:-2048}"
       --threads "${OXBOT_REAL_THREADS:-2}" )
if [[ -n "${OXBOT_REAL_DATA:-}" ]]; then
    args+=( --data "$OXBOT_REAL_DATA" )
fi
if [[ -n "${OXBOT_REAL_SOURCE:-}" ]]; then
    args+=( --source "$OXBOT_REAL_SOURCE" )
fi
if [[ "$OUT" = /* ]]; then
    OUT_PATH="$OUT"
else
    OUT_PATH="$ROOT/$OUT"
fi
if [[ -f "$OUT_PATH/latest.pt" ]]; then
    args+=( --resume "$OUT_PATH/latest.pt" )
    printf 'Resuming real-replay training from %s\n' "$OUT_PATH/latest.pt"
fi
args+=( "$@" )
cd "$ROOT"
exec "$PYTHON" "${args[@]}"
