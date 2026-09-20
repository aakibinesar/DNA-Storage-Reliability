#!/bin/bash
# scratch_rerun_fix_5_stages.sh
#
# Follow-up to scratch_rerun_stale_stages.sh: ablation, threshold_sensitivity,
# calibration_regimes, shap_stability, and channel_ablation all silently
# failed (0/28 configs, rc=1) during that run because scratch_parallel_stage.py
# had a Pool-pickling bug (fixed in commit e83f008). This reruns just those
# 5 stages with the fixed script, then regenerates figures again since some
# of them (fig3, fig_s4) were built from the empty/fallback results.
set -uo pipefail
cd "$(dirname "$0")"

export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
WORKERS=4
BRANCH="master-n267u8"
PROGRESS="STALE_RERUN_PROGRESS.md"
DRIVER_LOG="logs/_stale_rerun_driver.log"
TRAILER=$'\n\nCo-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>\nClaude-Session: https://claude.ai/code/session_01MD98rpFmYJMXYyvBk8QHQC'

mkdir -p logs
log() { echo "[$(date -u +%FT%TZ)] $*" | tee -a "$DRIVER_LOG"; }

commit_push() {
    local msg="$1"
    git add -A -- results/ "$PROGRESS"
    if ! git diff --cached --quiet; then
        git commit -m "${msg}${TRAILER}" >>"$DRIVER_LOG" 2>&1
        local ok=0
        for delay in 2 4 8 16; do
            if git push -u origin "$BRANCH" >>"$DRIVER_LOG" 2>&1; then ok=1; break; fi
            log "push failed, retrying in ${delay}s ..."
            sleep "$delay"
        done
        [ "$ok" = "1" ] || log "PUSH FAILED after retries for: $msg"
    else
        log "Nothing new to commit for: $msg"
    fi
}

mark() { echo "- [x] $1 -- $(date -u +%FT%TZ)" >> "$PROGRESS"; }

log "=== FIXUP: rerunning 5 stages that failed due to Pool-pickling bug ==="

for stage in ablation threshold_sensitivity calibration_regimes shap_stability channel_ablation; do
    log "=== PHASE $stage (fixed retry) ==="
    python3 -u scratch_parallel_stage.py "$stage" --workers "$WORKERS" --force \
        >> "logs/_phase_${stage}_fixed.log" 2>&1
    rc=$?
    n_out=$(ls "results/${stage}/" 2>/dev/null | wc -l)
    log "$stage rc=$rc, $n_out output files"
    mark "$stage rerun with fixed script at n=10000 (rc=$rc, $n_out files)"
    commit_push "Fix: $stage rerun with fixed scratch_parallel_stage.py"
done

log "=== PHASE figures (regenerate after fixup) ==="
python3 -u analysis/figures.py --config configs/experiment_config.yaml --out results/figures \
    >> logs/_phase_figures_fixed.log 2>&1
mark "figures regenerated after 5-stage fixup"
commit_push "Fix: figures regenerated from corrected n=10000 results"

log "=== FIXUP COMPLETE ==="
echo "FIXUP_DONE $(date -u +%FT%TZ)" >> "$PROGRESS"
commit_push "Fixup complete: all 9 stale stages now have real n=10000 output"
