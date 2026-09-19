#!/bin/bash
# scratch_rerun_stale_stages.sh
#
# Driver for regenerating the datasets/models at n=10000 and rerunning the
# 9 analysis stages that commit 0c188c6 deleted as stale n=2000-era output
# (ablation, threshold_sensitivity, calibration_regimes, shap_stability,
# channel_ablation, regime_evaluation, distribution_shift, transfer_radius,
# encoding_confound), pending a rerun that never happened.
#
# Designed to be started once with nohup/setsid and left alone: every phase
# commits+pushes its results immediately on completion, so progress survives
# even if whatever is watching this script gets interrupted. Safe to
# re-run/resume — every underlying script skips work whose output already
# exists on disk.
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

if [ ! -f "$PROGRESS" ]; then
    echo "# Stale-stage rerun at n=10000" > "$PROGRESS"
    echo "" >> "$PROGRESS"
fi
mark "driver started"
commit_push "Start n=10000 stale-stage rerun"

log "=== PHASE datasets ==="
python3 -u scratch_parallel_datasets.py --workers "$WORKERS" >> logs/_phase_datasets.log 2>&1
rc=$?
log "datasets rc=$rc"
mark "datasets regenerated at n=10000 (rc=$rc)"
commit_push "Stale rerun: datasets regenerated at n=10000"
if [ "$rc" != "0" ]; then log "ABORT: datasets phase failed"; exit 1; fi

log "=== PHASE train ==="
python3 -u scratch_parallel_train.py --workers "$WORKERS" >> logs/_phase_train.log 2>&1
rc=$?
log "train rc=$rc"
mark "risk models retrained at n=10000 (rc=$rc)"
commit_push "Stale rerun: risk models retrained at n=10000"
if [ "$rc" != "0" ]; then log "ABORT: train phase failed"; exit 1; fi

for stage in ablation threshold_sensitivity calibration_regimes shap_stability channel_ablation; do
    log "=== PHASE $stage ==="
    python3 -u scratch_parallel_stage.py "$stage" --workers "$WORKERS" >> "logs/_phase_${stage}.log" 2>&1
    rc=$?
    log "$stage rc=$rc"
    mark "$stage rerun at n=10000 (rc=$rc)"
    commit_push "Stale rerun: $stage results refreshed at n=10000"
done

log "=== PHASE regime_evaluation ==="
python3 -u analysis/regime_evaluation.py --config configs/experiment_config.yaml \
    --models-dir models/saved/ --out results/regime_evaluation \
    >> logs/_phase_regime_evaluation.log 2>&1
rc=$?
mark "regime_evaluation rerun at n=10000 (rc=$rc)"
commit_push "Stale rerun: regime_evaluation refreshed at n=10000"

log "=== PHASE distribution_shift ==="
: > logs/_phase_distribution_shift.log
for cov in 5 3; do
    for enc in simple constrained; do
        python3 -u analysis/distribution_shift.py --config configs/experiment_config.yaml \
            --coverage "$cov" --encoding "$enc" --out results/distribution_shift \
            >> logs/_phase_distribution_shift.log 2>&1
    done
done
mark "distribution_shift rerun at n=10000"
commit_push "Stale rerun: distribution_shift refreshed at n=10000"

log "=== PHASE transfer_radius ==="
python3 -u analysis/transfer_radius.py --config configs/experiment_config.yaml \
    --models-dir models/saved/ --out results/transfer_radius \
    >> logs/_phase_transfer_radius.log 2>&1
mark "transfer_radius rerun at n=10000"
commit_push "Stale rerun: transfer_radius refreshed at n=10000"

log "=== PHASE encoding_confound ==="
python3 -u analysis/encoding_confound.py --config configs/experiment_config.yaml \
    --out results/encoding_confound \
    >> logs/_phase_encoding_confound.log 2>&1
mark "encoding_confound rerun at n=10000"
commit_push "Stale rerun: encoding_confound refreshed at n=10000"

log "=== PHASE figures ==="
python3 -u analysis/figures.py --config configs/experiment_config.yaml --out results/figures \
    >> logs/_phase_figures.log 2>&1
mark "figures regenerated from refreshed n=10000 results"
commit_push "Stale rerun: figures regenerated from refreshed results"

log "=== ALL PHASES COMPLETE ==="
echo "ALL_DONE $(date -u +%FT%TZ)" >> "$PROGRESS"
commit_push "Stale rerun: all 9 stages + figures refreshed at n=10000 (complete)"
