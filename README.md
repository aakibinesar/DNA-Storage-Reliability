# DNA Storage Reliability

Calibrated sequence-level failure prediction and adaptive redundancy allocation for DNA data storage.

---

## Overview

DNA data storage encodes digital information into synthetic DNA oligonucleotides. Errors introduced during synthesis, PCR amplification, storage decay, and sequencing can cause Reed-Solomon (RS) error correction to fail — resulting in data loss.

Current practice allocates the same number of RS parity bytes to every oligo, regardless of how error-prone its sequence is. This project proposes a machine learning approach: **predict which oligos are likely to fail, then adaptively reallocate parity bytes from safe sequences to risky ones**, keeping the total storage budget neutral.

The pipeline covers the full experimental workflow:
- Synthetic DNA channel simulation (DeSP-inspired Monte Carlo model)
- Feature extraction from raw sequence composition
- Calibrated ML model training (XGBoost, Random Forest, Logistic Regression)
- Adaptive redundancy allocation vs. uniform and oracle baselines
- Feature ablation and distribution shift robustness analysis
- Paper-ready figure generation

---

## Experimental Design

The pipeline runs across **28 dataset configurations** (7 × 2 × 2):

| Axis | Values |
|------|--------|
| Substitution rate | 1%, 5%, 9%, 12%, 15%, 18%, 20% |
| Coverage depth (K) | K=5, K=3 |
| Encoding scheme | Simple (2-bit), Constrained (R∞-P8) |

Each configuration generates 2,000 oligos, simulates 30 Monte Carlo channel runs per oligo, and records failure frequency as the ML target.

### Failure Regime Map

Across the 28 configurations, three distinct regimes emerge:

| Regime | Configs | Failure Rate | ML useful? |
|--------|---------|-------------|------------|
| Under-failure | sub01–05 K=3, K=5 below 15% sub | < 2% | No — uniform allocation sufficient |
| **Sweet spot** | **sub09–15 K=3, sub15–20 K=5** | **5–90%** | **Yes — informative-regime AUROC 0.57–1.00** |
| Over-failure | sub18–20 K=3 | ~100% | No — max parity still insufficient |

Only 17 of 28 configs have enough informative-regime sequences (n≥20) to support a reliable AUROC estimate; the rest are too close to under- or over-failure to say much beyond "not much room for ML here." See `results/regime_evaluation/` for the full per-config, per-regime breakdown.

---

## Project Structure

```
├── src/
│   ├── sequence_generator.py     # Synthetic oligo generation + channel simulation
│   ├── feature_extractor.py      # ~80 sequence features per oligo
│   └── dataset_assembler.py      # Train/val/test splits (70/15/15)
├── models/
│   ├── train.py                  # XGBoost / RF / LR training with grid search
│   ├── calibrate.py              # Platt scaling, isotonic regression, temperature scaling
│   └── evaluate.py               # ECE, Brier score, PR-AUC, AUROC
├── allocation/
│   ├── mechanism.py              # Budget-neutral parity reallocation + noise-aware oracle
│   ├── baselines.py              # Rule-based allocation baselines (GC-dev, HP, composite, random)
│   └── experiment.py             # Allocation experiments (XGBoost vs. Oracle vs. Uniform vs. baselines)
├── analysis/
│   ├── ablation.py               # Feature group ablation analysis
│   ├── threshold_sensitivity.py  # Decision-threshold sweep, near-threshold label noise
│   ├── calibration_regimes.py    # Regime-stratified calibration with bootstrap CIs
│   ├── regime_evaluation.py      # Full Layer 1 metrics per regime, all configs
│   ├── distribution_shift.py     # Cross-regime transfer experiments
│   ├── transfer_radius.py        # Formal all-pairs transfer radius (regime-aware AUROC)
│   ├── shap_stability.py         # Bootstrap SHAP feature-importance stability
│   ├── encoding_confound.py      # Deconfounded simple-vs-constrained encoding comparison
│   ├── channel_ablation.py       # Model robustness under ablated channel-noise variants
│   └── figures.py                # Paper figures (Figs 2–5, S1, S4)
├── configs/
│   └── experiment_config.yaml    # All hyperparameters and grid settings
└── run_pipeline.py               # End-to-end orchestrator with checkpointing
```

---

## Installation

```bash
pip install numpy pandas scipy scikit-learn xgboost shap matplotlib seaborn pyyaml pyarrow
```

Python 3.10+ recommended.

---

## Running the Pipeline

Run all stages end-to-end:

```bash
python run_pipeline.py
```

Resume from a specific stage after interruption:

```bash
python run_pipeline.py --from-stage train
```

Run a single stage:

```bash
python run_pipeline.py --only figures --force
```

Check pipeline status:

```bash
python run_pipeline.py --status
```

### Pipeline Stages

| Stage | Description |
|-------|-------------|
| `datasets` | Generate sequences, simulate channel, extract features |
| `train` | Train and calibrate all models across 28 configs |
| `gate_check` | Calibration sanity check on the held-out gate config (informational — currently fails; see Known Limitations) |
| `threshold_sensitivity` | Wilson-interval label-noise sweep across decision thresholds |
| `calibration_regimes` | Regime-stratified ECE/Brier with bootstrap 95% CIs |
| `regime_evaluation` | Full Layer 1 metric suite (AUROC/F1/ECE/Brier), stratified by regime |
| `allocation` | Adaptive vs. uniform vs. oracle vs. rule-based-baseline allocation (84 experiments) |
| `ablation` | Feature group ablation across all 28 configs |
| `distribution_shift` | Cross-substitution-regime transfer robustness tests (12 conditions per stratum) |
| `transfer_radius` | Formal all-pairs transfer radius using regime-aware AUROC |
| `shap_stability` | Bootstrap stability of SHAP feature importance (informative regime only) |
| `encoding_confound` | Deconfounds the simple-vs-constrained encoding comparison via GC/HP post-stratification |
| `channel_ablation` | Model robustness under ablated channel-noise variants |
| `figures` | Generate all paper figures |

---

## Results

After the full pipeline completes:

```
results/
├── allocation/             # 84 NPZ files — OFR arrays (30 MC runs each)
├── ablation/               # 28 CSVs — per-group metric deltas
├── distribution_shift/     # 8 CSVs  — 12-condition transfer sweep + per-stratum radius summary
├── transfer_radius/        # formal all-pairs transfer radius (regime-aware AUROC)
├── calibration_regimes/    # per-config regime-stratified ECE/Brier with bootstrap CIs
├── regime_evaluation/      # full Layer 1 metrics per config per regime
├── threshold_sensitivity/  # decision-threshold sweep + near-threshold label noise
├── shap_stability/         # bootstrap SHAP feature-importance stability
├── encoding_confound/      # deconfounded simple-vs-constrained encoding comparison
├── channel_ablation/       # model robustness under ablated channel-noise variants
└── figures/
    ├── fig2_reliability_diagrams.png
    ├── fig3_shap_importance.png
    ├── fig4_ofr_vs_delta.png
    ├── fig5_distribution_shift.png
    ├── fig_s1_feature_distributions.png
    └── fig_s4_cost_reliability.png
```

**Key metric — Oligo Failure Rate (OFR)**: fraction of oligos that fail RS decoding after allocation, at the same total parity budget across conditions (uniform / oracle / model / rule-based baselines).

### Allocation Results

Across all 84 configs (28 keys × 3 delta values), comparing the oracle allocation (privileged, marginal-benefit/harm-ranked, budget-neutral) against plain uniform allocation:

| Outcome | Configs |
|---|---|
| Oracle beats or ties uniform | 47 / 84 |
| Oracle within 0.5 percentage points of uniform | 27 / 84 |
| Oracle shows a small residual loss (≤ ~1pp) | 10 / 84 |

The 10 residual-loss configs are entirely at delta=1 (zero at delta=2 or delta=4) — the smallest parity-reallocation step tested, where the true effect size is closest to the noise floor of the 30-run Monte Carlo estimate. This is a known, bounded, and reported limitation, not a hidden failure — see `allocation/mechanism.py`'s `oracle_allocation_greedy_swap` docstring for the full derivation.

### Distribution Shift / Transfer Radius Results

Models are trained on one substitution regime and evaluated on another **without retraining**. The formal transfer radius — the largest substitution-rate step at which every tested pair still passes both an ECE-ratio and a regime-aware AUROC-drop threshold — is:

| Stratum | Direction | Transfer radius | First failing step |
|---|---|---|---|
| K=3, simple | up / down | 0.000 | 2% |
| K=3, constrained | up / down | 0.000 | 2% |
| K=5, simple | up / down | 0.000 | 2% |
| K=5, constrained | up / down | 0.000 | 2% |

**Models do not transfer across substitution rates without retraining** — the transfer radius is 0.000 in every stratum and direction tested, failing even at the smallest tested step (2%). This replaces an earlier, looser estimate that used a whole-test-set AUROC criterion instead of a regime-aware one; see `results/transfer_radius/` for the full all-pairs sweep and `results/distribution_shift/` for the 12-condition per-stratum breakdown.

### Known Limitations

- **`gate_check` currently fails** on its held-out gate config (whole-test-set ECE well above the 0.05 threshold). This reflects the general point above — aggregate calibration metrics are misleading outside the informative regime — rather than a specific bug; it's tracked as informational and does not block the pipeline.
- **Encoding comparison is largely confounded.** A raw comparison of simple vs. constrained encoding shows constrained encoding failing less often, but post-stratifying on GC content and homopolymer run length (the `encoding_confound` stage) shows most of that raw effect disappears or reverses once composition is controlled for — see `results/encoding_confound/`.

---

## Key Design Decisions

- **Calibration matters**: raw model scores are used directly as risk values in the allocation mechanism, so probability calibration (Platt scaling for XGBoost, isotonic for RF) is critical — not just discrimination
- **Single-class guard**: at extreme substitution rates (both too low and too high), all sequences either pass or fail. A `DummyClassifier` is used at the low end; ablation is skipped entirely at the high end (100% failure rate)
- **Budget neutrality**: the allocation mechanism strictly enforces that total parity bytes added equals total parity bytes removed — no free lunch
- **Monte Carlo evaluation**: OFR is estimated over 30 independent channel simulation runs per configuration to account for stochastic variation
- **Noise-aware oracle allocation**: the oracle doesn't just rank sequences by estimated marginal benefit — it only reallocates parity when the estimated benefit of promoting one sequence exceeds the estimated harm of demoting another by a statistically-derived margin (not just "benefit > harm"). Naively ranking on point estimates from 30-run Monte Carlo samples is a textbook winner's-curse setup — the top/bottom of a noisy ranking is disproportionately luck, not signal — and without this margin, the oracle can lose to plain uniform allocation despite having privileged information.

---

## Configuration

All experiment parameters are in [`configs/experiment_config.yaml`](configs/experiment_config.yaml), including substitution rates, coverage depths, model hyperparameter grids, RS thresholds, and allocation delta values.
