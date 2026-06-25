"""
analysis/figures.py
===================
Generate all figures for the paper (paper-ready quality, matplotlib + seaborn).

Figure index:
  Figure 2 — Reliability diagrams for two representative conditions
  Figure 3 — SHAP feature importance bar chart (top 15 features)
  Figure 4 — FRR vs. delta for XGBoost (calibrated) and Oracle
  Figure 5 — Distribution shift: ECE degradation across transfer conditions
  Figure S1 — Feature distribution plots (R∞-P8 vs. simple mapping)
  Figure S2 — Brier score decomposition heatmap across all 16 configs
  Figure S3 — Calibration comparison: raw vs. Platt-scaled XGBoost
  Figure S4 — Cost-reliability trade-off curves (redundancy budget vs. FRR)

Usage:
    python analysis/figures.py --config configs/experiment_config.yaml \
        --out results/figures/
"""

import os
import sys
import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import yaml

warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'models'))

# ── Matplotlib setup (non-interactive backend) ──────────────────────────────
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

FONT_SIZE   = 11
TITLE_SIZE  = 13
FIG_DPI     = 150
PALETTE     = {
    'uniform'  : '#9e9e9e',
    'oracle'   : '#2196F3',
    'xgb_cal'  : '#4CAF50',
    'xgb_raw'  : '#FF5722',
    'simple'   : '#FF9800',
    'constrained': '#9C27B0',
}


def _setup_style():
    plt.rcParams.update({
        'font.size'         : FONT_SIZE,
        'axes.titlesize'    : TITLE_SIZE,
        'axes.labelsize'    : FONT_SIZE,
        'xtick.labelsize'   : FONT_SIZE - 1,
        'ytick.labelsize'   : FONT_SIZE - 1,
        'legend.fontsize'   : FONT_SIZE - 1,
        'figure.dpi'        : FIG_DPI,
        'axes.spines.top'   : False,
        'axes.spines.right' : False,
        'axes.grid'         : True,
        'grid.alpha'        : 0.3,
    })


# ── Figure 2: Reliability Diagrams ──────────────────────────────────────────

def figure2_reliability_diagrams(
    y_true_list:       List[np.ndarray],
    y_prob_cal_list:   List[np.ndarray],
    y_prob_raw_list:   List[np.ndarray],
    condition_labels:  List[str],
    out_path:          str,
    n_bins:            int = 10,
):
    """Reliability diagrams for two representative conditions side-by-side.

    Shows calibrated vs. uncalibrated XGBoost predictions.
    """
    _setup_style()
    from calibrate import calibration_curve_data
    from evaluate import expected_calibration_error

    n_cond = len(condition_labels)
    fig, axes = plt.subplots(1, n_cond, figsize=(6 * n_cond, 5), squeeze=False)

    for i, (y_t, y_cal, y_raw, label) in enumerate(
            zip(y_true_list, y_prob_cal_list, y_prob_raw_list, condition_labels)):
        ax = axes[0, i]

        for proba, name, colour, ls in [
            (y_cal, 'Calibrated (Platt)', PALETTE['xgb_cal'], '-'),
            (y_raw, 'Raw XGBoost',        PALETTE['xgb_raw'], '--'),
        ]:
            cd  = calibration_curve_data(y_t, proba, n_bins)
            ece = expected_calibration_error(y_t, proba, n_bins)
            ax.plot(cd['bin_means'], cd['bin_fracs'],
                    marker='o', color=colour, linestyle=ls,
                    label=f'{name} (ECE={ece:.3f})')

        # Perfect calibration diagonal
        ax.plot([0, 1], [0, 1], 'k:', linewidth=1, label='Perfect calibration')
        ax.fill_between([0, 1], [0, 1], alpha=0.05, color='k')

        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.set_xlabel('Mean predicted probability')
        ax.set_ylabel('Fraction of positives')
        ax.set_title(f'Reliability Diagram\n{label}')
        ax.legend(loc='upper left', framealpha=0.9)
        ax.set_aspect('equal')

    plt.tight_layout()
    _save_fig(fig, out_path)


# ── Figure 3: SHAP Feature Importance ───────────────────────────────────────

def figure3_shap_importance(
    importances:   np.ndarray,
    feature_names: List[str],
    out_path:      str,
    top_n:         int = 15,
):
    """Horizontal bar chart of mean |SHAP| values, top N features."""
    _setup_style()
    idx  = np.argsort(importances)[-top_n:]
    vals = importances[idx]
    names = [feature_names[i] for i in idx]

    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.barh(range(len(names)), vals, color=PALETTE['xgb_cal'], edgecolor='none')
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel('Mean |SHAP value|')
    ax.set_title(f'Feature Importance (Top {top_n}) — XGBoost')
    ax.invert_yaxis()

    # Add value labels
    for bar, val in zip(bars, vals):
        ax.text(bar.get_width() + max(vals) * 0.01, bar.get_y() + bar.get_height() / 2,
                f'{val:.4f}', va='center', fontsize=8)

    plt.tight_layout()
    _save_fig(fig, out_path)


# ── Figure 4: FRR vs. Delta ──────────────────────────────────────────────────

def figure4_frr_vs_delta(
    delta_results: Dict[int, Dict[str, np.ndarray]],
    out_path: str,
):
    """FRR mean ± std vs. delta for XGBoost-calibrated, Oracle, and Uniform."""
    _setup_style()
    fig, ax = plt.subplots(figsize=(7, 5))
    deltas = sorted(delta_results.keys())

    for condition, colour, label in [
        ('frr_uniform',  PALETTE['uniform'],  'Uniform (baseline)'),
        ('frr_oracle',   PALETTE['oracle'],   'Oracle (ceiling)'),
        ('frr_xgb_cal',  PALETTE['xgb_cal'],  'XGBoost Calibrated'),
    ]:
        means = [delta_results[d][condition].mean() for d in deltas]
        stds  = [delta_results[d][condition].std()  for d in deltas]
        ax.errorbar(
            deltas, means, yerr=stds,
            marker='o', color=colour, label=label,
            capsize=4, linewidth=2,
        )

    ax.set_xlabel('Delta (parity byte increment)')
    ax.set_ylabel('Failure Reduction Rate (FRR)')
    ax.set_title('FRR vs. Redundancy Reallocation Aggressiveness')
    ax.set_xticks(deltas)
    ax.legend()
    plt.tight_layout()
    _save_fig(fig, out_path)


# ── Figure 5: Distribution Shift ────────────────────────────────────────────

def figure5_distribution_shift(
    shift_results: Dict[str, dict],
    out_path:      str,
):
    """ECE and PR-AUC comparison across distribution shift conditions."""
    _setup_style()
    labels    = list(shift_results.keys())
    ece_indist = [shift_results[l]['indist_ece']      for l in labels]
    ece_trans  = [shift_results[l]['transfer_ece']    for l in labels]
    pra_indist = [shift_results[l]['indist_pr_auc']   for l in labels]
    pra_trans  = [shift_results[l]['transfer_pr_auc'] for l in labels]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, metric_in, metric_out, ylabel, title in [
        (axes[0], ece_indist, ece_trans, 'ECE', 'Expected Calibration Error'),
        (axes[1], pra_indist, pra_trans, 'PR-AUC', 'Precision-Recall AUC'),
    ]:
        x = np.arange(len(labels))
        w = 0.35
        ax.bar(x - w/2, metric_in, w, label='In-distribution', color=PALETTE['xgb_cal'], alpha=0.8)
        ax.bar(x + w/2, metric_out, w, label='Transfer',       color=PALETTE['xgb_raw'], alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels([l.replace('->', '→') for l in labels], rotation=20, ha='right')
        ax.set_ylabel(ylabel)
        ax.set_title(f'{title}\nIn-distribution vs. Transfer')
        ax.legend()

    plt.tight_layout()
    _save_fig(fig, out_path)


# ── Figure S1: Feature Distributions ────────────────────────────────────────

def figure_s1_feature_distributions(
    X_simple:      np.ndarray,
    X_constrained: np.ndarray,
    feature_names: List[str],
    out_path:      str,
    n_features:    int = 6,
):
    """Side-by-side feature distributions for simple vs. constrained encoding."""
    _setup_style()
    key_features = ['gc_global', 'homopolymer_max_run', 'approx_free_energy',
                    'shannon_entropy_global', 'repeat_fraction', 'palindrome_count']
    feat_idx = [feature_names.index(f) for f in key_features if f in feature_names][:n_features]

    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    axes = axes.flatten()

    for ax, idx in zip(axes, feat_idx):
        name = feature_names[idx]
        ax.hist(X_simple[:, idx],      bins=30, alpha=0.6,
                label='Simple',      color=PALETTE['simple'],      density=True)
        ax.hist(X_constrained[:, idx], bins=30, alpha=0.6,
                label='Constrained', color=PALETTE['constrained'], density=True)
        ax.set_title(name.replace('_', ' ').title())
        ax.set_ylabel('Density')
        ax.legend(fontsize=8)

    plt.suptitle('Feature Distributions: Simple vs. Constrained Encoding', y=1.02)
    plt.tight_layout()
    _save_fig(fig, out_path)


# ── Figure S4: Cost-Reliability Trade-off ───────────────────────────────────

def figure_s4_cost_reliability(
    delta_results: Dict[int, Dict[str, np.ndarray]],
    out_path:      str,
):
    """Cost-reliability trade-off: FRR reduction per redundancy unit."""
    _setup_style()
    fig, ax = plt.subplots(figsize=(7, 5))
    deltas = sorted(delta_results.keys())

    for condition, colour, label in [
        ('frr_xgb_cal',  PALETTE['xgb_cal'], 'XGBoost Calibrated'),
        ('frr_oracle',   PALETTE['oracle'],  'Oracle'),
        ('frr_uniform',  PALETTE['uniform'], 'Uniform baseline'),
    ]:
        frr_means = np.array([delta_results[d][condition].mean() for d in deltas])
        # Redundancy overhead: delta bytes × fraction of realloc sequences
        ax.plot(deltas, frr_means, marker='s', color=colour, label=label, linewidth=2)

    ax.set_xlabel('Reallocation delta Δ (parity bytes)')
    ax.set_ylabel('Mean Failure Rate')
    ax.set_title('Cost-Reliability Trade-off')
    ax.legend()
    plt.tight_layout()
    _save_fig(fig, out_path)


# ── Utility ──────────────────────────────────────────────────────────────────

def _save_fig(fig, out_path: str, dpi: int = FIG_DPI):
    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    print(f"  [figures] Saved {out_path}")


# ── CLI entry point ──────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Generate paper figures.')
    parser.add_argument('--config', default='configs/experiment_config.yaml')
    parser.add_argument('--out',    default='results/figures/')
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    print("[figures] Generating diagnostic feature-distribution figure (S1)...")
    from sequence_generator import generate_sequences
    from feature_extractor import extract_features

    seed    = cfg['random_seed']
    seq_cfg = cfg['sequence']
    n_diag  = 500

    seqs_simple = [s for s, _ in generate_sequences(
        n_diag, seq_cfg['seq_len_bases'], 'simple', seed=seed)]
    seqs_const  = [s for s, _ in generate_sequences(
        n_diag, seq_cfg['seq_len_bases'], 'constrained',
        seq_cfg['max_homopolymer'], seq_cfg['gc_min'], seq_cfg['gc_max'], seed=seed)]

    X_s, feat_names = extract_features(seqs_simple)
    X_c, _          = extract_features(seqs_const)

    figure_s1_feature_distributions(
        X_s, X_c, feat_names,
        out_path=os.path.join(args.out, 'fig_s1_feature_distributions.png')
    )
    print("[figures] Done — check", args.out)


if __name__ == '__main__':
    main()
