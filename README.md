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
| **Sweet spot** | **sub09–15 K=3, sub15–20 K=5** | **5–90%** | **Yes — AUROC 0.77–0.999** |
| Over-failure | sub18–20 K=3 | ~100% | No — max parity still insufficient |

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
│   ├── mechanism.py              # Budget-neutral parity reallocation logic
│   └── experiment.py             # Allocation experiments (XGBoost vs. Oracle vs. Uniform)
├── analysis/
│   ├── ablation.py               # Feature group ablation analysis
│   ├── distribution_shift.py     # Cross-regime transfer experiments
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
| `gate_check` | Validate model quality before allocation experiments |
| `allocation` | Run adaptive vs. uniform vs. oracle allocation (84 experiments) |
| `ablation` | Feature group ablation across all 28 configs |
| `distribution_shift` | Cross-substitution-regime transfer robustness tests |
| `figures` | Generate all paper figures |

---

## Results

After the full pipeline completes:

```
results/
├── allocation/          # 84 NPZ files — FRR arrays (30 MC runs each)
├── ablation/            # 28 CSVs — per-group metric deltas
├── distribution_shift/  # 4 CSVs  — ECE/Brier/PR-AUC under transfer
└── figures/
    ├── fig2_reliability_diagrams.png
    ├── fig3_shap_importance.png
    ├── fig4_ofr_vs_delta.png
    ├── fig_s1_feature_distributions.png
    └── fig_s4_cost_reliability.png
```

**Key metric — Failure Reduction Rate (FRR)**: how much the adaptive allocation reduces failures vs. the uniform baseline, at the same total parity budget.

### Distribution Shift Transfer Results

Models are trained on one substitution regime and evaluated on another without retraining:

| Transfer | AUROC | Robust? |
|----------|-------|---------|
| 9% → 12% (mild) | 0.997 | Yes |
| 9% → 15% (moderate) | 0.878 | No |
| 9% → 20% (severe) | 0.500 | No |
| 12% → 18% | 0.500 | No |

Models transfer robustly within one adjacent substitution step (~3%) but fail beyond that, defining a practical **transfer radius** for deployment without retraining.

---

## Key Design Decisions

- **Calibration matters**: raw model scores are used directly as risk values in the allocation mechanism, so probability calibration (Platt scaling for XGBoost, isotonic for RF) is critical — not just discrimination
- **Single-class guard**: at extreme substitution rates (both too low and too high), all sequences either pass or fail. A `DummyClassifier` is used at the low end; ablation is skipped entirely at the high end (100% failure rate)
- **Budget neutrality**: the allocation mechanism strictly enforces that total parity bytes added equals total parity bytes removed — no free lunch
- **Monte Carlo evaluation**: FRR is estimated over 30 independent channel simulation runs per configuration to account for stochastic variation

---

## Configuration

All experiment parameters are in [`configs/experiment_config.yaml`](configs/experiment_config.yaml), including substitution rates, coverage depths, model hyperparameter grids, RS thresholds, and allocation delta values.
