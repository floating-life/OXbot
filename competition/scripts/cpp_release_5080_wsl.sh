#!/usr/bin/env bash
# C++ release gate, run entirely on the local machine (no BotZone matches):
#   export FBDN weights -> build the C++ bot -> Python/C++ parity ->
#   duplicate match vs the online cf8 bot (and the previous C++ release)
#   through the official judge -> package the single-file BotZone source.
#
#   bash scripts/cpp_release_5080_wsl.sh [candidate.npz] [work_dir]
#   (default candidate: ckpts/dmc-realv2/champion/champion.npz)
#
# Env: OXBOT_CF8_MODEL (online cf8 weights, OXGDQ001 .bin), OXBOT_CPP_DEALS
# (default 500), OXBOT_CPP_WORKERS (14), OXBOT_EVAL_SEED, OXBOT_P99_LIMIT
# (seconds, default 0.5), OXBOT_RELEASE_DIR (default ../dist/release).
# Exit 0 = released, 3 = evaluated but not released, other = failure.
set -Eeuo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"   # competition/
REPO="$(cd -- "$ROOT/.." && pwd)"
cd "$ROOT"
VENV="${OXBOT_VENV:-/home/ggcle/.venvs/oxbot}"
if [[ -n "${OXBOT_PYTHON:-}" ]]; then PY="$OXBOT_PYTHON"
elif [[ -x "$VENV/bin/python" ]]; then PY="$VENV/bin/python"
elif [[ -x "$ROOT/.venv/bin/python" ]]; then PY="$ROOT/.venv/bin/python"
else PY="$(command -v python3)"; fi
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

CAND="${1:-ckpts/dmc-realv2/champion/champion.npz}"
WORK="${2:-$REPO/reports/cpp_release/$(date +%Y%m%d-%H%M%S)}"
CF8="${OXBOT_CF8_MODEL:-$REPO/models/oxbot-research-rollout-q-official256-cf8-listwise025-seed20261012.bin}"
DEALS="${OXBOT_CPP_DEALS:-500}"
WORKERS="${OXBOT_CPP_WORKERS:-14}"
SEED="${OXBOT_EVAL_SEED:-20261101}"
P99_LIMIT="${OXBOT_P99_LIMIT:-0.5}"
RELEASE_DIR="${OXBOT_RELEASE_DIR:-$REPO/dist/release}"

die() { printf 'FAILED: %s\n' "$*" >&2; exit 1; }
log() { printf '\n=== %s ===\n' "$*"; }
[[ -f "$CAND" ]] || die "candidate weights missing: $CAND"
[[ -f "$CF8" ]] || die "online cf8 model missing: $CF8 (set OXBOT_CF8_MODEL)"
mkdir -p "$WORK"
WORK="$(cd -- "$WORK" && pwd)"

log "build C++ bot and probes"
bash "$REPO/tools/build_wsl.sh" > "$WORK/build.log" 2>&1 || die "C++ build/tests failed (see $WORK/build.log)"
BOT="$REPO/bin/oxbot"

log "export FBDN weights"
"$PY" tools/export_fabledan_cpp.py "$CAND" "$WORK/candidate.fbdn" --dtype fp32 \
    --manifest "$WORK/candidate.fbdn.json" > /dev/null || die "FBDN export"
SHA8="$(sha256sum "$WORK/candidate.fbdn" | cut -c1-8)"
WEIGHTS_REL="data/fabledan_w_${SHA8}.fbd"
mkdir -p "$REPO/data"
cp "$WORK/candidate.fbdn" "$REPO/$WEIGHTS_REL"
printf 'weights: %s\n' "$WEIGHTS_REL"

log "Python/C++ parity"
"$PY" tools/check_fabledan_cpp.py --npz "$CAND" --weights "$REPO/$WEIGHTS_REL" \
    --probe "$REPO/bin/fabledan_probe" --report "$WORK/parity.json" > /dev/null \
    || die "C++ inference does not match Python (see $WORK/parity.json)"

duel() {   # duel <opponent model path> <name>
    "$PY" tools/cpp_duel.py --bot-a "$BOT --model $REPO/$WEIGHTS_REL" \
        --bot-b "$BOT --model $1" --deals "$DEALS" --workers "$WORKERS" \
        --seed "$SEED" --artifact-a "$REPO/$WEIGHTS_REL" --artifact-b "$1" \
        --report "$WORK/vs_$2.json"
}
log "duplicate vs online cf8 ($DEALS deals, official judge)"
duel "$CF8" cf8 || die "duel vs cf8 invalid (see $WORK/vs_cf8.json)"
PREV=""
if [[ -f "$RELEASE_DIR/current.json" ]]; then
    PREV="$("$PY" -c 'import json,sys; print(json.load(open(sys.argv[1]))["weights_path"])' "$RELEASE_DIR/current.json")"
    if [[ -f "$PREV" ]] && ! cmp -s "$PREV" "$REPO/$WEIGHTS_REL"; then
        log "duplicate vs previous C++ release"
        duel "$PREV" previous || die "duel vs previous release invalid"
    else
        PREV=""
    fi
fi

log "release verdict"
rc=0
"$PY" - "$WORK" "$P99_LIMIT" "$DEALS" <<'EOF' || rc=$?
import json, os, sys
work, limit, deals = sys.argv[1], float(sys.argv[2]), int(sys.argv[3])
gates, rows = [], {}
for name in ("cf8", "previous"):
    path = os.path.join(work, "vs_%s.json" % name)
    if not os.path.exists(path):
        continue
    r = json.load(open(path))
    rows[name] = {k: r.get(k) for k in ("avg_score_diff_per_game", "ci95_per_game", "deals",
                                        "verdict", "judge_errors", "turn_seconds")}
    if r.get("deals") != deals or not r.get("complete"):
        gates.append("%s: incomplete (%s/%s deals)" % (name, r.get("deals"), deals))
    if r.get("verdict") != "A_STRONGER":
        gates.append("%s: CI %s not entirely above 0" % (name, r.get("ci95_per_game")))
    p99 = r.get("turn_seconds", {}).get("a", {}).get("p99_max_shard")
    if p99 is None or p99 > limit:
        gates.append("%s: candidate p99 %.3fs over local limit %.3fs" % (name, p99 or -1, limit))
    lo, hi = r["ci95_per_game"]
    print("  vs %-8s %+.3f/game  95%% CI [%+.3f, %+.3f]  p99 %.0f ms  %s"
          % (name, r["avg_score_diff_per_game"], lo, hi, 1000 * (p99 or 0), r["verdict"]))
release = not gates
json.dump({"release": release, "gate_failures": gates, "results": rows,
           "p99_limit_seconds": limit, "deals": deals},
          open(os.path.join(work, "release_verdict.json"), "w"), indent=2)
for g in gates:
    print("  gate:", g)
print("RELEASE" if release else "NOT RELEASED")
sys.exit(0 if release else 3)
EOF
[[ $rc -eq 0 || $rc -eq 3 ]] || die "verdict step"
if [[ $rc -eq 3 ]]; then
    printf 'reports: %s\n' "$WORK"
    exit 3
fi

log "package BotZone single-file source"
mkdir -p "$RELEASE_DIR"
VERSION="fabledan-${SHA8}-cpp-fp32"
SRC="$RELEASE_DIR/oxbot-${VERSION}.cpp"
(cd "$REPO" && "$PY" tools/amalgamate.py --output "$SRC" --model-path "$WEIGHTS_REL" \
    --candidate-version "$VERSION") > "$WORK/amalgamate.log" 2>&1 || die "amalgamate (see $WORK/amalgamate.log)"
g++ -std=c++17 -O2 -Wall -Wextra -Werror "$SRC" -o "$WORK/oxbot-release" \
    || die "single-file source does not compile"
cp "$REPO/$WEIGHTS_REL" "$RELEASE_DIR/fabledan_w_${SHA8}.fbd"
"$PY" - "$RELEASE_DIR" "$SRC" "$REPO/$WEIGHTS_REL" "$WEIGHTS_REL" "$WORK" "$VERSION" <<'EOF'
import hashlib, json, os, sys
release_dir, src, weights, weights_rel, work, version = sys.argv[1:]
def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()
record = {"version": version, "source": os.path.abspath(src), "source_sha256": sha(src),
          "source_bytes": os.path.getsize(src), "weights_path": os.path.abspath(weights),
          "botzone_weights_name": os.path.basename(weights_rel),
          "weights_sha256": sha(weights), "evidence": os.path.abspath(work)}
json.dump(record, open(os.path.join(release_dir, "current.json"), "w"), indent=2)
print(json.dumps(record, indent=2))
EOF
cat <<EOF

RELEASED $VERSION
Upload to BotZone (C++17, ordinary JSON, keep-running on):
  source : $SRC
  storage: $RELEASE_DIR/fabledan_w_${SHA8}.fbd  (keep this exact file name)
First play debug must show policy=model and model_status=model_selected.
EOF
