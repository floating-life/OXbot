#!/usr/bin/env bash
# Long DMC self-play from the real-v2 BC checkpoint, then a statistically
# meaningful duplicate evaluation and an official-judge legality pass.
#
#   bash scripts/selfplay_eval_5080_wsl.sh            # train + eval (default 24 h)
#   OXBOT_HOURS=48 bash scripts/selfplay_eval_5080_wsl.sh
#   bash scripts/selfplay_eval_5080_wsl.sh eval       # only evaluate the current run
#   bash scripts/selfplay_eval_5080_wsl.sh train      # only train (resumes if possible)
#   OXBOT_SAFE_CUDA=0 ...                         # opt out of conservative inference
#   OXBOT_MICRO_BATCH=256 ...                     # override the memory-safe default
#   OXBOT_HOURS=0.1 OXBOT_DEALS=20 OXBOT_JUDGE_GAMES=8 \
#       OXBOT_OUT=ckpts/dmc-smoke bash scripts/selfplay_eval_5080_wsl.sh   # smoke test
#   # next round: continue from the champion, which the new candidate must beat
#   OXBOT_OUT=ckpts/dmc-r2 OXBOT_WARM_START=ckpts/dmc-realv2/champion/champion.pt \
#       OXBOT_CHAMPION_DIR=ckpts/dmc-realv2/champion bash scripts/selfplay_eval_5080_wsl.sh
#
# After a promotion the C++ release gate runs (scripts/cpp_release_5080_wsl.sh:
# local official-judge match vs the online cf8 bot); OXBOT_CPP_RELEASE=0 skips it.
#
# Re-running "train" resumes from $OUT/latest.pt; Ctrl+C is safe.
# Promotion rule (one rule, no hand tuning): the candidate becomes champion
# only if the 95% paired-bootstrap CI vs real-v2 AND vs the previous champion
# is entirely above 0 over at least 1000 deals, and the official judge
# completes the requested games with 0 errors and required model inference.
set -Eeuo pipefail

MODE="${1:-all}"
case "$MODE" in all|train|eval) ;; *)
    printf 'usage: %s [all|train|eval]\n' "$0" >&2; exit 2 ;;
esac

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
VENV="${OXBOT_VENV:-/home/ggcle/.venvs/oxbot}"
if [[ -n "${OXBOT_PYTHON:-}" ]]; then PY="$OXBOT_PYTHON"
elif [[ -x "$VENV/bin/python" ]]; then PY="$VENV/bin/python"
elif [[ -x "$ROOT/.venv/bin/python" ]]; then PY="$ROOT/.venv/bin/python"
else PY="$(command -v python3)"; fi
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

OUT="${OXBOT_OUT:-ckpts/dmc-realv2}"
WARM="${OXBOT_WARM_START:-ckpts/real-v2/best.pt}"
BASELINE="${OXBOT_BASELINE:-ckpts/real-v2/best.npz}"
HOURS="${OXBOT_HOURS:-24}"
ACTORS="${OXBOT_ACTORS:-10}"
RING="${OXBOT_RING:-32}"
LADDER="${OXBOT_LADDER_FRAC:-0.5}"
BATCH="${OXBOT_BATCH:-4096}"
DEALS="${OXBOT_DEALS:-1000}"            # 1000 deals = 2000 games per opponent
EVAL_SEED="${OXBOT_EVAL_SEED:-20261101}" # keep fixed across candidates
WORKERS="${OXBOT_EVAL_WORKERS:-14}"
JUDGE_GAMES="${OXBOT_JUDGE_GAMES:-200}"
JUDGE="${OXBOT_JUDGE:-judge/judge_official.py}"
SAFE_CUDA="${OXBOT_SAFE_CUDA:-1}"                # math SDPA + FP32 inference worker
MICRO_BATCH="${OXBOT_MICRO_BATCH:-128}"           # learner gradient-accumulation chunk
CHAMP_DIR="${OXBOT_CHAMPION_DIR:-$OUT/champion}"
STAMP="$(date +%Y%m%d-%H%M%S)"
RUN_ID="${OXBOT_RUN_ID:-$STAMP-$$}"
export OXBOT_RUN_ID="$RUN_ID"
EVAL_DIR=""

fail() { local code="$1"; shift; ERROR_MESSAGE="$*"; printf 'FAILED: %s\n' "$*" >&2; exit "$code"; }
die() { fail 1 "$@"; }
log() { printf '\n=== %s ===\n' "$*"; }
[[ "$BATCH" =~ ^[1-9][0-9]*$ ]] || die "OXBOT_BATCH must be a positive integer"
[[ "$MICRO_BATCH" =~ ^[1-9][0-9]*$ ]] || die "OXBOT_MICRO_BATCH must be a positive integer"
case "$SAFE_CUDA" in
    0|1) ;;
    *) die "OXBOT_SAFE_CUDA must be 0 or 1" ;;
esac

# Training and evaluation must not race over latest/champion files.
mkdir -p "$OUT"
exec 9>"$OUT/.selfplay.lock"
flock -n 9 || die "another self-play/evaluation run is using $OUT"

# Install lifecycle handling only after acquiring the lock. A rejected second
# launcher must never overwrite the live run's status, even on its EXIT path.
STARTED_AT="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
STAGE=starting PIPELINE_STATUS=running PIPELINE_EXIT="" CHILD_PID=""
SIGNAL_NAME="" ERROR_MESSAGE="" TRAINING_ACTIVE=0 TRAINING_STATUS=running
TRAINING_EXIT="" CHECKPOINT_SHA=""
BUDGET_FILE="$OUT/.remaining-hours-$$"
COMPLETION_FILE="$OUT/.completed-checkpoint-$$"

persist_state() {
    "$PY" - "$OUT" "$RUN_ID" "$$" "$CHILD_PID" "$MODE" "$STAGE" \
        "$PIPELINE_STATUS" "$PIPELINE_EXIT" "$STARTED_AT" "$EVAL_DIR" \
        "$SIGNAL_NAME" "$ERROR_MESSAGE" "$TRAINING_ACTIVE" "$TRAINING_STATUS" \
        "$TRAINING_EXIT" "$HOURS" "$CHECKPOINT_SHA" <<'EOF'
import json, os, sys
from datetime import datetime, timezone
from pathlib import Path
(out, run_id, pid, child, mode, stage, status, code, started, eval_dir,
 signal, error, training_active, training_status, training_code, hours,
 checkpoint_sha) = sys.argv[1:]
now = datetime.now(timezone.utc).isoformat()
record = dict(schema=1, run_id=run_id, pid=int(pid), child_pid=int(child) if child else None,
              mode=mode, stage=stage, status=status, exit_code=int(code) if code else None,
              started_at=started, updated_at=now,
              finished_at=now if status != "running" else None,
              eval_dir=os.path.abspath(eval_dir) if eval_dir else None,
              signal=signal or None, error=error or None, target_hours=hours)
def write(name, value):
    path = Path(out) / name
    temporary = path.with_name(path.name + ".%s.tmp" % os.getpid())
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
write("last_pipeline.json", record)
if training_active == "1":
    record.update(status=training_status,
                  training_ok=training_status == "completed" and training_code == "0",
                  exit_code=int(training_code) if training_code else None,
                  finished_at=now if training_status != "running" else None,
                  checkpoint_sha256=checkpoint_sha or None)
    write("last_training.json", record)
EOF
}

stop_child() {
    [[ -n "$CHILD_PID" ]] || return 0
    # Each stage has its own process group, including CUDA actors. Forward the
    # signal and bound cleanup so an unresponsive worker cannot hide termination.
    kill -s "$1" -- "-$CHILD_PID" 2>/dev/null || true
    local watchdog
    setsid bash -c 'sleep "$1"; kill -KILL -- "-$2" 2>/dev/null || true' \
        bash "${OXBOT_SIGNAL_GRACE_SECONDS:-30}" "$CHILD_PID" 9>&- &
    watchdog=$!
    wait "$CHILD_PID" 2>/dev/null || true
    kill -TERM -- "-$watchdog" 2>/dev/null || kill -TERM "$watchdog" 2>/dev/null || true
    wait "$watchdog" 2>/dev/null || true
    # A parent can exit before its descendants. Do not leave those workers
    # holding GPU memory or the inherited output-directory lock.
    kill -KILL -- "-$CHILD_PID" 2>/dev/null || true
    CHILD_PID=""
}

on_signal() {
    SIGNAL_NAME="$1" PIPELINE_STATUS=interrupted PIPELINE_EXIT="$2"
    ERROR_MESSAGE="received $1 during $STAGE"
    if [[ "$TRAINING_ACTIVE" == 1 ]]; then
        TRAINING_STATUS=interrupted TRAINING_EXIT="$2"
    fi
    persist_state || printf 'FAILED: cannot persist interruption status\n' >&2
    stop_child "$1"
    exit "$2"
}

on_exit() {
    local rc=$?
    trap - EXIT HUP INT TERM
    if [[ "$PIPELINE_STATUS" == running ]]; then
        if [[ $rc -eq 0 ]]; then PIPELINE_STATUS=completed
        else PIPELINE_STATUS=failed; fi
    fi
    PIPELINE_EXIT="$rc"
    if [[ "$TRAINING_ACTIVE" == 1 && "$TRAINING_STATUS" == running ]]; then
        TRAINING_STATUS=failed TRAINING_EXIT="$rc"
    fi
    if [[ $rc -ne 0 && -z "$ERROR_MESSAGE" ]]; then
        ERROR_MESSAGE="command exited with $rc during $STAGE"
    fi
    stop_child TERM
    persist_state || { printf 'FAILED: cannot persist final status\n' >&2; rc=1; }
    rm -f -- "$BUDGET_FILE" "$COMPLETION_FILE"
    exit "$rc"
}
trap on_exit EXIT
trap 'on_signal HUP 129' HUP
trap 'on_signal INT 130' INT
trap 'on_signal TERM 143' TERM
persist_state
[[ "$RUN_ID" =~ ^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$ ]] || die "invalid OXBOT_RUN_ID"
command -v setsid >/dev/null || die "setsid is required for supervised stages"

run_step() {
    STAGE="$1"; shift
    persist_state || return $?
    # Bash normally ignores SIGINT for asynchronous children. Reset it before
    # exec so Ctrl+C/HUP/TERM remain observable by the real Python process.
    env --default-signal=INT,TERM,HUP setsid "$@" <&0 &
    CHILD_PID=$!
    persist_state || return $?
    local rc=0
    wait "$CHILD_PID" || rc=$?
    if [[ $rc -ne 0 ]]; then
        # The trainer may crash before its actor shutdown path executes.
        # Its surviving descendants still own the GPU and inherit fd 9.
        stop_child TERM
    else
        CHILD_PID=""
    fi
    persist_state || return $?
    return "$rc"
}

preflight() {
    log "preflight"
    run_step preflight "$PY" - <<'EOF' || die "PyTorch/CUDA not usable; run scripts/setup_wsl.sh"
import torch, numpy
assert torch.cuda.is_available(), "CUDA not available"
print("torch", torch.__version__, "|", torch.cuda.get_device_name(0))
EOF
    [[ -f "$JUDGE" ]] || die "official judge missing: $JUDGE"
}

train() {
    log "DMC self-play for ${HOURS} h -> $OUT"
    local budget_hours="$HOURS"
    # --max-hours is per process. Missing/invalid accumulated time is not zero:
    # refuse to resume rather than silently granting a fresh 24-hour budget.
    run_step training_budget "$PY" - "$OUT/latest.pt" "$HOURS" "$BUDGET_FILE" <<'EOF' \
        || die "cannot determine remaining training budget from $OUT/latest.pt"
import math
import sys
from pathlib import Path
target = float(sys.argv[2]) * 3600.0
if not math.isfinite(target) or target < 0:
    raise ValueError("OXBOT_HOURS must be finite and nonnegative")
elapsed = 0.0
if Path(sys.argv[1]).exists():
    import torch
    checkpoint = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
    elapsed = checkpoint.get("meta", {}).get("elapsed_seconds")
    if isinstance(elapsed, bool) or not isinstance(elapsed, (int, float)) \
            or not math.isfinite(elapsed) or elapsed < 0:
        raise ValueError("missing or invalid accumulated elapsed_seconds; budget cannot be reset")
remaining = target - elapsed if target else 0.0
if target and remaining <= 0.0:
    raise SystemExit("total training budget exhausted; only a verified completed run may use eval mode")
Path(sys.argv[3]).write_text("%.12f\n" % (remaining / 3600.0), encoding="utf-8")
EOF
    read -r budget_hours < "$BUDGET_FILE"
    printf 'remaining total budget: %s h\n' "$budget_hours"
    local args=( -m fabledan.train_fast --out "$OUT"
                 --actors "$ACTORS" --ring "$RING" --ladder-frac "$LADDER"
                 --batch "$BATCH" --micro-batch "$MICRO_BATCH"
                 --max-hours "$budget_hours" --export-cycles 50 --eval-cycles 25
                 --eval-games 100 --device cuda:0 --infer-device cuda:0 )
    if [[ "$SAFE_CUDA" == "1" ]]; then
        args+=( --safe-cuda )
        printf 'safe CUDA inference: math SDPA, FP32 (learner AMP unchanged)\n'
    fi
    printf 'learner micro-batch: %s (effective batch remains 4096)\n' "$MICRO_BATCH"
    if [[ -f "$OUT/latest.pt" ]]; then
        printf 'resuming %s\n' "$OUT/latest.pt"
        args+=( --resume "$OUT/latest.pt" )
    else
        [[ -f "$WARM" ]] || die "warm-start checkpoint missing: $WARM"
        args+=( --warm-start "$WARM" )
        # Default keeps the source's 80-dim features (matches real-v2 and the
        # C++ FBDN path); set OXBOT_FEATURE_DIM=224 to migrate on warm start.
        [[ -n "${OXBOT_FEATURE_DIM:-}" ]] && args+=( --feature-dim "$OXBOT_FEATURE_DIM" )
    fi
    # A failed run may leave an older checkpoint behind; never evaluate it
    # as the result of that failed run.
    TRAINING_ACTIVE=1 TRAINING_STATUS=running TRAINING_EXIT=""
    local rc=0
    run_step training "$PY" "${args[@]}" || rc=$?
    [[ $rc -eq 0 ]] || fail "$rc" "train_fast exited with $rc; evaluation/promotion skipped"
    run_step training_validation "$PY" - "$OUT/latest.pt" "$HOURS" "$COMPLETION_FILE" <<'EOF' \
        || die "training did not produce a verified final checkpoint; evaluation/promotion skipped"
import hashlib, math, sys
from pathlib import Path
import torch
path = Path(sys.argv[1])
meta = torch.load(path, map_location="cpu", weights_only=False).get("meta", {})
elapsed = meta.get("elapsed_seconds")
if isinstance(elapsed, bool) or not isinstance(elapsed, (int, float)) \
        or not math.isfinite(elapsed) or elapsed < 0:
    raise ValueError("final checkpoint has invalid elapsed_seconds")
if meta.get("training_kind") != "dmc" or meta.get("optimizer_steps", 0) <= 0:
    raise ValueError("final checkpoint must contain completed DMC optimizer updates")
target = float(sys.argv[2]) * 3600.0
if target:
    if meta.get("stop_reason") != "time limit reached" or elapsed < target - 0.01:
        raise ValueError("training stopped before the total time budget completed")
elif meta.get("stop_reason") != "cycles completed" \
        or meta.get("cycle", -1) != meta.get("training_args", {}).get("cycles"):
    raise ValueError("training did not reach the configured cycle target")
with path.open("rb") as stream:
    digest = hashlib.file_digest(stream, "sha256").hexdigest()
Path(sys.argv[3]).write_text(digest + "\n", encoding="utf-8")
EOF
    read -r CHECKPOINT_SHA < "$COMPLETION_FILE"
    STAGE=training_complete TRAINING_STATUS=completed TRAINING_EXIT=0
    persist_state
    TRAINING_ACTIVE=0
}

dup() {   # dup <candidate> <opponent> <name>
    run_step "eval_$3" "$PY" tools/duplicate_eval.py --a "$1" --b "$2" --deals "$DEALS" \
        --seed "$EVAL_SEED" --ladder-frac "$LADDER" --workers "$WORKERS" \
        --report "$EVAL_DIR/vs_$3.json"
}

evaluate() {
    [[ -f "$OUT/latest.pt" ]] || die "no candidate $OUT/latest.pt (train first)"
    [[ -f "$BASELINE" ]] || die "baseline missing: $BASELINE"
    mkdir -p "$OUT/eval"
    EVAL_DIR="$(mktemp -d "$OUT/eval/$STAMP-XXXXXX")"
    local cand="$EVAL_DIR/candidate.npz"
    local baseline="$EVAL_DIR/baseline.npz"
    local previous_champion="$EVAL_DIR/previous_champion.npz"
    local verdict_args=()
    log "freeze candidate checkpoint and export its matching NumPy weights"
    run_step eval_freeze "$PY" - "$OUT" "$EVAL_DIR" "$BASELINE" "$CHAMP_DIR" <<'EOF' || die "candidate freeze/validation"
import hashlib, json, shutil, sys
from pathlib import Path
import torch
from fabledan.model_torch import export_npz, load_ckpt

torch.set_num_threads(1)
out, target, baseline, champion_dir = map(Path, sys.argv[1:])
status = json.loads((out / "last_training.json").read_text(encoding="utf-8"))
if status.get("training_ok") is not True or status.get("exit_code") != 0 \
        or status.get("status") != "completed":
    raise RuntimeError("the last training run is incomplete or failed")
shutil.copyfile(out / "latest.pt", target / "candidate.pt")
with (target / "candidate.pt").open("rb") as stream:
    checkpoint_sha = hashlib.file_digest(stream, "sha256").hexdigest()
if checkpoint_sha != status.get("checkpoint_sha256"):
    raise RuntimeError("candidate does not match the verified final training checkpoint")
model, checkpoint = load_ckpt(target / "candidate.pt", device="cpu")
meta = checkpoint.get("meta", {})
if meta.get("training_kind") != "dmc" or meta.get("optimizer_steps", 0) <= 0:
    raise RuntimeError("candidate must contain completed DMC optimizer updates")
if meta.get("stop_reason") not in ("cycles completed", "time limit reached"):
    raise RuntimeError("candidate has no successful training stop reason")
if not all(torch.isfinite(value).all().item() for value in model.state_dict().values()):
    raise RuntimeError("candidate has non-finite weights")
export_npz(model.eval(), target / "candidate.npz")
shutil.copyfile(baseline, target / "baseline.npz")
champion = champion_dir / "champion.npz"
if champion.exists():
    shutil.copyfile(champion, target / "previous_champion.npz")
def digest(path):
    with open(path, "rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()
report = {key: meta.get(key) for key in (
    "training_kind", "feature_dim", "feature_version", "cycle", "total_samples",
    "optimizer_steps", "partial_cycle_steps", "elapsed_seconds", "stop_reason")}
report.update(training_ok=True, exit_code=0, run_id=status.get("run_id"),
              candidate_sha256=digest(target / "candidate.npz"),
              checkpoint_sha256=digest(target / "candidate.pt"),
              baseline_sha256=digest(target / "baseline.npz"),
              champion_required=champion.exists())
(target / "training_status.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
print(json.dumps(report, indent=2))
EOF

    log "duplicate vs real-v2 BC ($DEALS deals)"; dup "$cand" "$baseline" realv2 || die "eval vs real-v2"
    log "duplicate vs rule ($DEALS deals)";       dup "$cand" rule rule || die "eval vs rule"
    if [[ -f "$previous_champion" ]]; then
        log "duplicate vs current champion"
        dup "$cand" "$previous_champion" champion || die "eval vs champion"
        verdict_args+=( --require-champion )
    else
        log "no previous champion; real-v2 is the initial benchmark"
    fi

    log "official judge legality ($JUDGE_GAMES games, mixed tribute scenarios)"
    local judge_rc=0
    run_step eval_judge "$PY" tools/judge_runner.py --judge "$JUDGE" --games "$JUDGE_GAMES" \
        --scenario mix --seed "$EVAL_SEED" --weights "$cand" --weights-b "$baseline" \
        --require-model --report "$EVAL_DIR/judge.json" || judge_rc=$?
    "$PY" - "$EVAL_DIR" "$EVAL_SEED" <<'EOF' || die "judge report identity"
import json, sys
from pathlib import Path
folder = Path(sys.argv[1])
path = folder / "judge.json"
if path.exists():
    report = json.loads(path.read_text(encoding="utf-8"))
    status = json.loads((folder / "training_status.json").read_text(encoding="utf-8"))
    report.update(candidate_sha256=status["candidate_sha256"],
                  baseline_sha256=status["baseline_sha256"], seed=int(sys.argv[2]))
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
EOF

    log "verdict"
    local rc=0
    run_step eval_verdict "$PY" tools/selfplay_verdict.py "$EVAL_DIR" --judge-exit-code "$judge_rc" \
        --deals "$DEALS" --judge-games "$JUDGE_GAMES" --seed "$EVAL_SEED" \
        "${verdict_args[@]}" || rc=$?
    [[ $rc -eq 0 || $rc -eq 3 ]] || die "verdict step"
    if [[ $rc -eq 0 ]]; then
        STAGE=promotion
        persist_state
        mkdir -p "$CHAMP_DIR"
        cp "$cand" "$CHAMP_DIR/champion.npz"
        cp "$EVAL_DIR/candidate.pt" "$CHAMP_DIR/champion.pt"
        printf '%s\n' "$EVAL_DIR" > "$CHAMP_DIR/promoted_from.txt"
        printf 'new champion: %s\n' "$CHAMP_DIR/champion.npz"
        if [[ "${OXBOT_CPP_RELEASE:-1}" == 1 ]]; then
            log "C++ release gate (local official-judge match vs online cf8)"
            local cpp_rc=0
            run_step cpp_release bash scripts/cpp_release_5080_wsl.sh \
                "$CHAMP_DIR/champion.npz" "$EVAL_DIR/cpp_release" || cpp_rc=$?
            [[ $cpp_rc -eq 0 || $cpp_rc -eq 3 ]] || die "C++ release gate failed ($cpp_rc)"
        fi
    fi
    printf 'reports: %s\n' "$EVAL_DIR"
}

[[ "$MODE" == eval ]] || { preflight; train; }
[[ "$MODE" == train ]] || evaluate   # "not promoted" is a normal outcome (exit 0)
STAGE=complete
