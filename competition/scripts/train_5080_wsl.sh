#!/usr/bin/env bash
set -Eeuo pipefail

# Single-GPU 9800X3D + RTX 5080 self-play launcher.  Use environment
# variables for persistent tuning and append any normal train_fast options:
#   scripts/train_5080_wsl.sh --cycles 1
#   OXBOT_OUT=ckpts/smoke scripts/train_5080_wsl.sh --cycles 2
# An existing latest.pt is resumed automatically; set OXBOT_NO_RESUME=1 for a
# fresh run.  Bash arrays preserve spaces in output paths and user options.

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
    printf 'TRAINING LAUNCH FAILED: Python runtime not found; run setup_wsl.sh.\n' >&2
    exit 1
}

OUT="${OXBOT_OUT:-ckpts/run1}"
ACTORS="${OXBOT_ACTORS:-8}"
RING="${OXBOT_RING:-32}"
LADDER="${OXBOT_LADDER_FRAC:-0.5}"
EVAL_GAMES="${OXBOT_EVAL_GAMES:-100}"
EVAL_CYCLES="${OXBOT_EVAL_CYCLES:-25}"
EXPORT_CYCLES="${OXBOT_EXPORT_CYCLES:-50}"
BATCH="${OXBOT_BATCH:-4096}"
MICRO_BATCH="${OXBOT_MICRO_BATCH:-256}"
DEVICE="${OXBOT_DEVICE:-cuda:0}"
INFER_DEVICE="${OXBOT_INFER_DEVICE:-cuda:0}"
if [[ "$OUT" = /* ]]; then
    OUT_PATH="$OUT"
else
    OUT_PATH="$ROOT/$OUT"
fi

args=( -m fabledan.train_fast
       --out "$OUT"
       --actors "$ACTORS"
       --ring "$RING"
       --ladder-frac "$LADDER"
       --batch "$BATCH"
       --micro-batch "$MICRO_BATCH"
       --eval-games "$EVAL_GAMES"
       --eval-cycles "$EVAL_CYCLES"
       --export-cycles "$EXPORT_CYCLES"
       --device "$DEVICE"
       --infer-device "$INFER_DEVICE" )

if [[ -n "${OXBOT_CYCLES:-}" ]]; then
    args+=( --cycles "$OXBOT_CYCLES" )
fi
if [[ -n "${OXBOT_MAX_HOURS:-}" ]]; then
    args+=( --max-hours "$OXBOT_MAX_HOURS" )
fi
if [[ -f "$OUT_PATH/latest.pt" && "${OXBOT_NO_RESUME:-0}" != 1 ]]; then
    args+=( --resume "$OUT_PATH/latest.pt" )
    printf 'Resuming %s\n' "$OUT_PATH/latest.pt"
else
    printf 'Starting fresh training in %s\n' "$OUT_PATH"
fi
args+=( "$@" )

cd "$ROOT"
exec "$PYTHON" "${args[@]}"
