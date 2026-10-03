#!/usr/bin/env bash
# Long DMC self-play from the real-v2 BC checkpoint, then a statistically
# meaningful duplicate evaluation and an official-judge legality pass.
#
#   bash scripts/selfplay_eval_5080_wsl.sh            # train + eval (default 24 h)
#   OXBOT_HOURS=48 bash scripts/selfplay_eval_5080_wsl.sh
#   bash scripts/selfplay_eval_5080_wsl.sh eval       # only evaluate the current run
#   bash scripts/selfplay_eval_5080_wsl.sh train      # only train (resumes if possible)
#   OXBOT_HOURS=0.1 OXBOT_DEALS=20 OXBOT_JUDGE_GAMES=8 \
#       OXBOT_OUT=ckpts/dmc-smoke bash scripts/selfplay_eval_5080_wsl.sh   # smoke test
#
# Re-running "train" resumes from $OUT/latest.pt; Ctrl+C is safe.
# Promotion rule (one rule, no hand tuning): the candidate becomes champion
# only if the 95% paired-bootstrap CI vs real-v2 AND vs the previous champion
# is entirely above 0, and the official judge reports 0 errors.
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

OUT="${OXBOT_OUT:-ckpts/dmc-realv2}"
WARM="${OXBOT_WARM_START:-ckpts/real-v2/best.pt}"
BASELINE="${OXBOT_BASELINE:-ckpts/real-v2/best.npz}"
HOURS="${OXBOT_HOURS:-24}"
ACTORS="${OXBOT_ACTORS:-10}"
RING="${OXBOT_RING:-32}"
LADDER="${OXBOT_LADDER_FRAC:-0.5}"
DEALS="${OXBOT_DEALS:-1000}"            # 1000 deals = 2000 games per opponent
EVAL_SEED="${OXBOT_EVAL_SEED:-20261101}" # keep fixed across candidates
WORKERS="${OXBOT_EVAL_WORKERS:-14}"
JUDGE_GAMES="${OXBOT_JUDGE_GAMES:-200}"
JUDGE="${OXBOT_JUDGE:-judge/judge_official.py}"
CHAMP_DIR="$OUT/champion"
STAMP="$(date +%Y%m%d-%H%M%S)"
EVAL_DIR="$OUT/eval/$STAMP"

die() { printf 'FAILED: %s\n' "$*" >&2; exit 1; }
log() { printf '\n=== %s ===\n' "$*"; }

preflight() {
    log "preflight"
    "$PY" - <<'EOF' || die "PyTorch/CUDA not usable; run scripts/setup_wsl.sh"
import torch, numpy
assert torch.cuda.is_available(), "CUDA not available"
print("torch", torch.__version__, "|", torch.cuda.get_device_name(0))
EOF
    [[ -f "$JUDGE" ]] || die "official judge missing: $JUDGE"
}

train() {
    log "DMC self-play for ${HOURS} h -> $OUT"
    local args=( -m fabledan.train_fast --out "$OUT"
                 --actors "$ACTORS" --ring "$RING" --ladder-frac "$LADDER"
                 --max-hours "$HOURS" --export-cycles 50 --eval-cycles 25
                 --eval-games 100 --device cuda:0 --infer-device cuda:0 )
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
    # train_fast saves latest.pt/latest.npz on --max-hours or Ctrl+C.
    "$PY" "${args[@]}" || printf 'train_fast exited with %s; evaluating what was saved\n' "$?"
}

dup() {   # dup <candidate> <opponent> <name>
    "$PY" tools/duplicate_eval.py --a "$1" --b "$2" --deals "$DEALS" \
        --seed "$EVAL_SEED" --ladder-frac "$LADDER" --workers "$WORKERS" \
        --report "$EVAL_DIR/vs_$3.json"
}

evaluate() {
    local cand="$OUT/latest.npz"
    [[ -f "$cand" ]] || die "no candidate $cand (train first)"
    [[ -f "$BASELINE" ]] || die "baseline missing: $BASELINE"
    mkdir -p "$EVAL_DIR"
    cp "$cand" "$EVAL_DIR/candidate.npz"
    cand="$EVAL_DIR/candidate.npz"   # freeze: training may overwrite latest.npz

    log "duplicate vs real-v2 BC ($DEALS deals)"; dup "$cand" "$BASELINE" realv2 || die "eval vs real-v2"
    log "duplicate vs rule ($DEALS deals)";       dup "$cand" rule rule || die "eval vs rule"
    if [[ -f "$CHAMP_DIR/champion.npz" ]]; then
        log "duplicate vs current champion"
        dup "$cand" "$CHAMP_DIR/champion.npz" champion || die "eval vs champion"
    fi

    log "official judge legality ($JUDGE_GAMES games, all tribute scenarios)"
    local judge_rc=0
    "$PY" tools/judge_runner.py --judge "$JUDGE" --games "$JUDGE_GAMES" \
        --scenario mix --weights "$cand" --weights-b "$BASELINE" \
        --require-model --report "$EVAL_DIR/judge.json" || judge_rc=$?

    log "verdict"
    local rc=0
    "$PY" - "$EVAL_DIR" "$judge_rc" <<'EOF' || rc=$?
import json, os, sys
d, judge_rc = sys.argv[1], int(sys.argv[2])
rows, ok = [], judge_rc == 0
for name in ("realv2", "rule", "champion"):
    p = os.path.join(d, "vs_%s.json" % name)
    if not os.path.exists(p):
        continue
    r = json.load(open(p))
    lo, hi = r["ci95_per_game"]
    rows.append((name, r["avg_score_diff_per_game"], lo, hi, r["verdict"]))
    if name in ("realv2", "champion") and r["verdict"] != "A_STRONGER":
        ok = False
for name, m, lo, hi, v in rows:
    print("  vs %-9s %+.3f/game  95%% CI [%+.3f, %+.3f]  %s" % (name, m, lo, hi, v))
print("  official judge: %s" % ("0 errors" if judge_rc == 0 else "ERRORS (see judge.json)"))
summary = {"promote": ok, "judge_ok": judge_rc == 0,
           "results": [dict(zip(("opponent", "avg", "ci_lo", "ci_hi", "verdict"), r)) for r in rows]}
json.dump(summary, open(os.path.join(d, "summary.json"), "w"), indent=2)
print("PROMOTE" if ok else "KEEP CURRENT CHAMPION")
sys.exit(0 if ok else 3)
EOF
    [[ $rc -eq 0 || $rc -eq 3 ]] || die "verdict step"
    if [[ $rc -eq 0 ]]; then
        mkdir -p "$CHAMP_DIR"
        cp "$cand" "$CHAMP_DIR/champion.npz"
        cp "$OUT/latest.pt" "$CHAMP_DIR/champion.pt" 2>/dev/null || true
        printf '%s\n' "$EVAL_DIR" > "$CHAMP_DIR/promoted_from.txt"
        printf 'new champion: %s\n' "$CHAMP_DIR/champion.npz"
        printf 'next: export FBDN with tools/export_fabledan_cpp.py and run check_submission.py\n'
    fi
    printf 'reports: %s\n' "$EVAL_DIR"
}

[[ "$MODE" == eval ]] || { preflight; train; }
[[ "$MODE" == train ]] || evaluate   # "not promoted" is a normal outcome (exit 0)
