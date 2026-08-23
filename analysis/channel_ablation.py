"""
analysis/channel_ablation.py
============================
R7 mitigation: test whether model predictions degrade when individual
reliability-skew mechanisms in the channel are disabled.

Background
----------
The DNAStorageChannel contains two hard-wired reliability-skew mechanisms:
  1. GC-content amplification bias  (controlled by pcr_bias)
  2. Homopolymer PCR-stutter        (controlled by hp_stutter)

Both mechanisms' inputs (gc_content, max_hp_run) are in the ML feature set,
so the model could achieve high AUROC simply by recovering the injected rules
rather than learning genuine sequence-error relationships.

Experiment
----------
Four channel variants are tested with a model trained on the FULL channel:

  full          — reference: original channel with all mechanisms active
  no_gc_skew    — pcr_bias=0, so effective coverage is uniform w.r.t. GC
  no_hp_stutter — hp_stutter=0, so HP runs no longer reduce coverage
  flat_skew     — both off; failures driven by substitution/indel rates only

For each variant:
  1. Re-simulate test sequences under the ablated channel (n_runs passes).
  2. Compute new failure labels (failure_freq under ablated channel).
  3. Evaluate model predictions (unchanged — trained on full channel) against
     the new labels using AUROC, Brier score, and ECE.

Interpretation
--------------
  - AUROC stays high across all variants  →  model learned genuine signal
    beyond the injected GC/HP rules.
  - AUROC drops on flat_skew but not on no_gc_skew / no_hp_stutter  →  model
    learned both mechanisms independently.
  - AUROC drops to ~0.5 on flat_skew  →  model primarily learned the injected
    rules; this is the pessimistic outcome and should be stated as a limitation.

Usage (CLI):
    python analysis/channel_ablation.py \\
        --config configs/experiment_config.yaml \\
        --key sub09_k5_simple \\
        --models-dir models/saved/ \\
        --out results/channel_ablation/
"""

import argparse
import os
import sys
from typing import Dict, List

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'models'))

ABLATION_VARIANTS = ['full', 'no_gc_skew', 'no_hp_stutter', 'flat_skew']


# ── Per-sequence failure-frequency estimation under an ablated channel ────────

def estimate_failure_freq(
    sequences:   List[str],
    channel,
    coverage:    int,
    l_rs:        int,
    n_runs:      int = 15,
    base_seed:   int = 9000,
) -> np.ndarray:
    """Run the channel n_runs times per sequence; return per-sequence failure_freq.

    Uses a separate seed range (base_seed + run_idx) that does not overlap
    with any dataset-generation or allocation seeds.

    Parameters
    ----------
    l_rs : RS parity bytes to use for decoding failure check
    """
    from consensus_voter import majority_vote_consensus, count_errors

    N = len(sequences)
    failures = np.zeros(N, dtype=np.float64)

    for run_idx in range(n_runs):
        run_ch = channel.clone(seed=base_seed + run_idx)
        for i, seq in enumerate(sequences):
            reads     = run_ch.simulate(seq, coverage)
            consensus = majority_vote_consensus(reads, len(seq))
            stats     = count_errors(seq, consensus)
            if stats['byte_errors'] > l_rs // 2:
                failures[i] += 1

    return failures / n_runs


# ── Main ablation experiment ──────────────────────────────────────────────────

def run_channel_ablation(
    sequences:   List[str],
    X_te:        np.ndarray,
    y_te:        np.ndarray,
    model,
    cfg:         dict,
    sub_rate:    float,
    coverage:    int,
    n_runs:      int = 15,
    base_seed:   int = 9000,
    verbose:     bool = True,
) -> pd.DataFrame:
    """Evaluate model AUROC/Brier/ECE on each ablated channel variant.

    Parameters
    ----------
    model     : fitted CalibratedModel (trained on the full channel)
    y_te      : original test labels (failure_freq under full channel)
    n_runs    : MC runs per sequence per ablation variant (15 is sufficient
                for ranking experiments; use 30 for production results)

    Returns
    -------
    DataFrame with columns:
        ablation, auroc, brier, ece, mean_failure_freq, n_sequences
    """
    from channel_model import build_ablated_channel
    from evaluate import expected_calibration_error, brier_score_decomposed

    try:
        from sklearn.metrics import roc_auc_score, average_precision_score
    except ImportError:
        raise ImportError("scikit-learn required")

    l_rs = cfg['sequence']['l_rs_default']
    model_proba = model.predict_proba(X_te)  # fixed — trained on full channel

    rows = []
    for ablation in ABLATION_VARIANTS:
        if verbose:
            print(f"  [channel_ablation] variant={ablation}")

        abl_channel = build_ablated_channel(cfg, sub_rate, ablation, seed=base_seed - 1)
        abl_ff = estimate_failure_freq(
            sequences, abl_channel, coverage, l_rs,
            n_runs=n_runs,
            # Offset by ablation index so variants get non-overlapping seeds
            base_seed=base_seed + ABLATION_VARIANTS.index(ablation) * 200,
        )

        y_bin = (abl_ff >= 0.5).astype(int)
        n_pos = int(y_bin.sum())
        n_neg = int(len(y_bin) - n_pos)

        if n_pos == 0 or n_neg == 0:
            auroc = float('nan')
            prauc = float('nan')
        else:
            auroc = float(roc_auc_score(y_bin, model_proba))
            prauc = float(average_precision_score(y_bin, model_proba))

        brier = float(np.mean((abl_ff - model_proba) ** 2))
        ece   = expected_calibration_error(abl_ff, model_proba, n_bins=10)

        rows.append({
            'ablation'          : ablation,
            'auroc'             : auroc,
            'pr_auc'            : prauc,
            'brier'             : brier,
            'ece'               : ece,
            'mean_failure_freq' : float(abl_ff.mean()),
            'frac_failed'       : float(y_bin.mean()),
            'n_pos'             : n_pos,
            'n_neg'             : n_neg,
            'n_sequences'       : len(sequences),
        })

        if verbose:
            status = f"AUROC={auroc:.3f}" if not np.isnan(auroc) else "AUROC=degenerate"
            print(f"    {status}  Brier={brier:.4f}  ECE={ece:.4f}"
                  f"  mean_ff={abl_ff.mean():.3f}  pos={n_pos}/{len(y_bin)}")

    return pd.DataFrame(rows)


# ── CLI entry point ───────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='R7: channel mechanism ablation experiment.'
    )
    parser.add_argument('--config',     default='configs/experiment_config.yaml')
    parser.add_argument('--key',        required=True,
                        help='Dataset key, e.g. sub09_k5_simple')
    parser.add_argument('--models-dir', default='models/saved/')
    parser.add_argument('--n-runs',     type=int, default=15,
                        help='MC runs per sequence per ablation variant (default 15)')
    parser.add_argument('--out',        default='results/channel_ablation/')
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    sys.path.insert(0, 'src')
    sys.path.insert(0, 'models')
    from dataset_assembler import load_dataset
    from train import load_models

    parts    = args.key.split('_')
    sub_rate = int(parts[0].replace('sub', '')) / 100
    coverage = int(parts[1].replace('k', ''))

    print(f"[channel_ablation] {args.key}  sub_rate={sub_rate}  coverage={coverage}")

    X_tr, X_val, X_te, y_tr, y_val, y_te, feat_names = load_dataset(args.key, cfg)
    models  = load_models(args.models_dir, args.key)
    xgb_cal = models.get('xgboost')

    if xgb_cal is None:
        print("[channel_ablation] No XGBoost model found — skipping.")
        return

    # Load test sequences
    data_path = os.path.join(cfg['paths']['datasets_dir'], f'{args.key}.parquet')
    splits    = pd.read_parquet(
        os.path.join(cfg['paths']['splits_dir'], f'{args.key}_splits.parquet')
    )
    df        = pd.read_parquet(data_path)
    test_idx  = splits[splits['split'] == 'test']['index'].values
    sequences = df['dna_sequence'].values[test_idx].tolist()

    results_df = run_channel_ablation(
        sequences, X_te, y_te, xgb_cal,
        cfg, sub_rate, coverage,
        n_runs=args.n_runs,
        base_seed=cfg['random_seed'] + 9000,
        verbose=True,
    )

    os.makedirs(args.out, exist_ok=True)
    out_path = os.path.join(args.out, f'{args.key}_channel_ablation.csv')
    results_df.to_csv(out_path, index=False)
    print(f"\n[channel_ablation] Saved -> {out_path}")
    print(results_df.to_string(index=False))


if __name__ == '__main__':
    main()
