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
| **Sweet spot** | **sub09–15 K=3, sub15–20 K=5** | **5–90%** | **Yes — informative-regime AUROC 0.54–1.00** |
| Over-failure | sub18–20 K=3 | ~100% | No — max parity still insufficient |

12 of 28 configs have enough informative-regime sequences (n≥20) in their test set to support a reliable AUROC estimate. This count uses a leak-free canonical train/val/test split shared across every substitution rate and coverage depth for a given encoding scheme (see Fixed Issues below) — a genuinely unbiased split, unlike an earlier version that stratified independently per condition, which incidentally guaranteed better informative-regime representation in every test set but let the same physical sequence appear in one config's training data and another's "held-out" test data. The lower count is the honest cost of removing that leakage, not a regression in model quality. See `results/regime_evaluation/` for the full per-config, per-regime breakdown.

---

## Project Structure

```
├── src/
│   ├── sequence_generator.py     # Synthetic oligo generation + channel simulation
│   ├── feature_extractor.py      # ~80 sequence features per oligo
│   └── dataset_assembler.py      # Canonical (leak-free) train/val/test splits, one per encoding
├── models/
│   ├── train.py                  # XGBoost / RF / LR training with grid search
│   ├── train_benefit_model.py    # Part B: benefit-aware model (predicts marginal benefit, not failure risk)
│   ├── calibrate.py              # Platt scaling, isotonic regression, temperature scaling
│   └── evaluate.py               # ECE, Brier score, PR-AUC, AUROC
├── allocation/
│   ├── mechanism.py              # Budget-neutral parity reallocation + noise-aware oracle
│   ├── baselines.py              # Rule-based allocation baselines (GC-dev, HP, composite, random)
│   ├── experiment.py             # Allocation experiments; paired oracle estimation; deployable tier selection
│   └── significance.py           # Paired Wilcoxon significance tests with BH-FDR correction, all 84 combos
├── analysis/
│   ├── ablation.py               # Feature group ablation analysis
│   ├── threshold_sensitivity.py  # Decision-threshold sweep, near-threshold label noise
│   ├── calibration_regimes.py    # Regime-stratified calibration with bootstrap CIs
│   ├── regime_evaluation.py      # Full Layer 1 metrics per regime, all configs
│   ├── model_comparison.py       # XGBoost vs. Random Forest vs. Logistic Regression, same metric suite
│   ├── distribution_shift.py     # Cross-regime transfer experiments
│   ├── transfer_radius.py        # Formal all-pairs transfer radius (regime-aware AUROC)
│   ├── shap_stability.py         # Bootstrap SHAP feature-importance stability
│   ├── encoding_confound.py      # Deconfounded simple-vs-constrained encoding comparison
│   ├── channel_ablation.py       # Model robustness under ablated channel-noise variants
│   ├── risk_marginal_benefit_correlation.py  # Diagnostic: does model risk track true marginal benefit?
│   ├── validate_benefit_model.py # Part B: benefit model vs. failure-risk model, scored on true test-set benefit
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
| `gate_check` | Calibration sanity check on the held-out gate config (informational; scores the model against the same continuous target it was trained on) |
| `threshold_sensitivity` | Wilson-interval label-noise sweep across decision thresholds |
| `calibration_regimes` | Regime-stratified ECE/Brier with bootstrap 95% CIs |
| `regime_evaluation` | Full Layer 1 metric suite (AUROC/F1/ECE/Brier), stratified by regime |
| `model_comparison` | XGBoost vs. Random Forest vs. Logistic Regression, same metric suite |
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
├── model_comparison/       # XGBoost vs. RF vs. Logistic Regression, same metric suite
├── threshold_sensitivity/  # decision-threshold sweep + near-threshold label noise
├── shap_stability/         # bootstrap SHAP feature-importance stability
├── encoding_confound/      # deconfounded simple-vs-constrained encoding comparison
├── channel_ablation/       # model robustness under ablated channel-noise variants
└── figures/
    ├── fig2_reliability_diagrams.png
    ├── fig3_shap_importance.png
    ├── fig4_ofr_vs_delta.png
    ├── fig5_distribution_shift.png
    ├── fig6_benefit_vs_risk_deployable.png
    ├── fig_s1_feature_distributions.png
    └── fig_s4_cost_reliability.png
```

**Key metric — Oligo Failure Rate (OFR)**: fraction of oligos that fail RS decoding after allocation, at the same total parity budget across conditions (uniform / oracle / model / rule-based baselines).

### Allocation Results

Every allocation experiment reports **two distinct comparisons**, because they answer different questions:

- **Diagnostic (oracle-sized budget)**: the model and every rule-based baseline are given the *oracle's own* reallocation budget size, isolating ranking quality from budget size. Useful for research, but not a fair picture of real deployment — the oracle's budget comes from privileged marginal-benefit/harm information a real system doesn't have.
- **Deployable (validation-sized budget)**: the model chooses its own reallocation budget by testing a small grid (5%/10%/20%/30%) on validation data only and freezing the winner before touching test data — no oracle or test-label information anywhere in the choice. Every rule-based baseline gets the same validation-chosen budget for a fair comparison. This is the number that reflects what you'd actually get by deploying the model.

Across all 84 configs (28 keys × 3 delta values):

| Comparison | Beats/ties uniform | Small loss (≤0.5pp) | Meaningful loss |
|---|---|---|---|
| **Oracle** (diagnostic, marginal-benefit/harm-ranked) | 79 / 84 | 5 / 84 | 0 / 84 |
| **Deployed model** (fair, validation-sized budget) | 34 / 84 | 28 / 84 | 22 / 84 |

The oracle result reflects a paired Monte Carlo estimation design: the low/default/high-parity outcomes for a given run share one simulated channel-noise draw rather than three independently resimulated ones, which substantially reduces estimation noise for the same 30-run budget (see `allocation/experiment.py`'s `estimate_paired_marginal_effects`). Under that cleaner estimate, the theoretical ceiling is essentially always at least as good as uniform.

The deployed-model result is the more important number for anyone actually using this system, and it's honest about a real gap: the model reliably wins in about 40% of configs, is roughly break-even in another third, and meaningfully underperforms uniform in about a quarter. **Root cause (confirmed directly, see `analysis/risk_marginal_benefit_correlation.py`): the model is trained to predict raw failure probability, not marginal benefit of added parity, and these are essentially uncorrelated on average (mean Spearman ≈ -0.005, median ≈ -0.037 across 28 configs at Δ=2) even though the model discriminates raw failure risk well.** A failure-risk classifier is the wrong tool for a reallocation decision — see Benefit-Aware Model below for the direct fix, which closes most of this gap.

At Δ=4 the target mismatch is slightly worse (mean Spearman ≈ -0.026, median ≈ -0.087) — see Delta as a Leverage Dial below for why a larger reallocation step size makes the risk-vs-benefit mismatch, and its consequences, more pronounced rather than less.

### Benefit-Aware Model (Part B)

Rather than only diagnosing the target-mismatch problem above, the project includes a second model trained directly on the quantity the allocation decision needs: marginal benefit of added parity, not raw failure risk (`models/train_benefit_model.py`).

**First attempt used the same hard pass/fail threshold-crossing label the oracle uses internally, and it didn't help.** Diagnosis: at a representative delta, a sequence only registers *any* measurable benefit if one of its 30 simulated runs lands on a single specific byte-error-count integer (the exact boundary between two correction capacities) — for `sub20_k3_simple`, **62% of sequences got an exactly-zero label** purely from that narrow window rarely being hit in 30 samples, not because their true benefit was actually zero. A label that coarse gives a model almost nothing learnable to fit.

**Fix: a kernel-smoothed (sigmoid) label instead of a hard threshold**, replacing `1(byte_errors > capacity)` with a smooth function that extracts partial information from every simulated run based on how close it is to the boundary, not just runs landing exactly on it. This is a training-label device only — the oracle's actual allocation decisions and every reported OFR figure still use the true hard RS-decode pass/fail rule, since that's the real physical criterion. On `sub20_k3_simple` this eliminated the zero-inflation entirely (62% → 0% exactly-zero) and reversed the model's relationship with true benefit from strongly backwards to strongly correct (Spearman vs. true benefit: -0.36 → +0.38; vs. a comparably clean ground truth: -0.67 → +0.87).

**Full 28-config validation**, benefit model vs. the old failure-risk model, both scored against true test-set marginal benefit (`analysis/validate_benefit_model.py`), run at both reallocation step sizes:

| Δ | | Mean Spearman | Median Spearman | Configs improved |
|---|---|---|---|---|
| **2** | Old failure-risk model | -0.005 | -0.037 | — |
| **2** | **New benefit-aware model** | **+0.132** | **+0.112** | **17 / 28** |
| **4** | Old failure-risk model | -0.026 | -0.087 | — |
| **4** | **New benefit-aware model** | **+0.267** | **+0.201** | **20 / 28** |

At Δ=2, the improvement is concentrated exactly where it matters: the configs where the old model was most badly *backwards* (`sub20_k3_simple`: -0.36 → +0.38; `sub05_k3_simple`: -0.27 → +0.31; `sub01_k3_simple`: -0.20 → +0.30) see the largest gains, while configs where the old model was already reasonable see only small, bounded declines (worst case -0.08). Three `sub01` configs that had no usable signal at all under the old model (constant predictions) now get valid, small-positive correlations. The same pattern holds at Δ=4, more strongly — e.g. `sub20_k3_simple` goes from -0.61 to +0.74.

**Wired into the actual allocation OFR experiments** (`allocation/experiment.py`): the benefit model gets its own validation-chosen deployable budget (same grid, same no-oracle-information discipline as the risk model) and its own `ofr_benefit_model_deployable` condition, run across all 28 configs at both Δ=2 and Δ=4:

| Δ | Model | Beats/ties uniform | Small loss (≤0.5pp) | Meaningful loss | Mean OFR reduction (pp) |
|---|---|---|---|---|---|
| 2 | Risk model (deployable) | 17 / 28 | 6 / 28 | 5 / 28 | 0.21 ± 0.20 |
| 2 | **Benefit-aware model (deployable)** | 11 / 28 | 15 / 28 | **2 / 28** | **0.35 ± 0.23** |
| 4 | Risk model (deployable) | 12 / 28 | 9 / 28 | 7 / 28 | 0.26 ± 0.30 |
| 4 | **Benefit-aware model (deployable)** | 8 / 28 | 9 / 28 | **11 / 28** | **0.71 ± 0.50** |

(Mean ± SEM of the per-config `uniform − method` OFR difference across the 28 configs; see Figure 6, `results/figures/fig6_benefit_vs_risk_deployable.png`.)

At Δ=2 this is an unambiguous win: the benefit model has a *larger* average OFR reduction than the risk model **and** a *smaller* meaningful-loss count (2/28 vs. 5/28) — better on both the mean and the tail. At Δ=4 the picture is more mixed: the benefit model's average reduction more than doubles (0.71pp vs. 0.35pp at Δ=2), but its meaningful-loss count also roughly doubles (11/28 vs. 2/28) — a higher-mean, higher-variance trade-off rather than a clean win. This is not a contradiction with the correlation table above (which shows Δ=4 correlation is *better*, not worse) — see the next section for the mechanism that reconciles the two.

### Delta as a Leverage Dial

The Δ=4 result above is counter-intuitive on its face: the benefit model's *ranking quality* (Spearman correlation with true marginal benefit) is better at Δ=4 than at Δ=2, yet its *aggregate allocation outcome* has more meaningful-loss configs at Δ=4. The reconciliation is structural, not a modeling flaw.

RS correction capacity is `l_rs // 2` (integer floor division), and the default `l_rs=8` gives capacity 4. Reallocating parity by Δ changes capacity by:

| Δ | New `l_rs` (promote / demote) | New capacity | Capacity swing |
|---|---|---|---|
| 1 | 9 / 7 | 4 / 3 | promotion: **no change** (9 // 2 = 4) |
| 2 | 10 / 6 | 5 / 3 | ±1 |
| 4 | 12 / 4 | 6 / 2 | ±2 |

A larger Δ moves each reallocated sequence across a bigger capacity gap. That amplifies the payoff of a *correct* promote/demote decision — but by the same mechanism, it amplifies the cost of an *incorrect* one, independent of how good the underlying ranking is. Δ is a leverage dial on both sides of every decision, not just the good side.

**Controlled isolation test** (not yet part of the automated pipeline; ad hoc verification): to confirm this is a real mechanism and not a confound from the Δ=2 and Δ=4 benefit models selecting different sequences, four configs were tested by fixing the *exact same* promoted/demoted sequences (selected once, using only the Δ=2 model's ranking) and then applying a Δ=2-sized vs. a Δ=4-sized parity swap to that identical selection, against the same simulated channel noise:

| Config | Δ=2 swap, OFR vs. uniform | Δ=4 swap, OFR vs. uniform |
|---|---|---|
| `sub09_k5_simple` | +0.34% | +1.20% |
| `sub05_k3_simple` | +0.06% | +1.19% |
| `sub12_k5_constrained` | +0.21% | +0.84% |
| `sub18_k5_simple` | +0.10% | +0.22% |

With sequence selection held completely fixed, the Δ=4-sized swap costs more than the Δ=2-sized swap in all four cases — confirming the leverage effect directly, rather than inferring it from aggregate before/after numbers alone. **Practical takeaway**: Δ=2 is the safer operating point (smaller downside per wrong decision), and Δ=4 is only worth it if the ranking model is good enough, and consistently good enough across the deployed regime, to be trusted with the larger per-decision stakes it creates.

### Model Comparison: XGBoost vs. Random Forest vs. Logistic Regression

XGBoost is the primary model used everywhere above, but all three tiers are trained for every config (`models/train.py`). `analysis/model_comparison.py` scores all three on the same footing — the informative-regime Layer 1 suite from Regime Evaluation above (regime assignment and binary labels both derived from the same continuous `failure_freq`, so a linear discriminator and two calibrated regressors are compared on one common yardstick, not three different targets):

| Model | Usable configs | Mean AUROC | Median AUROC | Mean F1 | Mean ECE | Mean Brier |
|---|---|---|---|---|---|---|
| XGBoost (Platt-calibrated) | 12 / 28 | 0.858 | 0.914 | 0.655 | 0.040 | 0.0095 |
| **Random Forest (isotonic-calibrated)** | 12 / 28 | **0.866** | **0.948** | **0.713** | **0.034** | **0.0086** |
| Logistic Regression | 12 / 28 | 0.795 | 0.828 | 0.515 | 0.225 | 0.0739 |

All three reach a usable informative regime on the same 12 of 28 configs, confirming that count is a property of the data regime, not of any one model's failure to fit.

**Random Forest and XGBoost are not meaningfully different on discrimination**: a paired Wilcoxon signed-rank test on informative-regime AUROC across the 12 shared configs gives W=22, p=0.63 (mean difference +0.008 in RF's favor, 5 configs favoring each, 2 exact ties) — well short of significance. RF's edge in the table above is concentrated in F1/ECE/Brier, i.e. calibration and decision-threshold behavior, not ranking quality; the likely explanation is that isotonic calibration (RF) is a more flexible fit than Platt scaling (XGBoost) on this dataset size, not that the underlying random-forest model discriminates failure risk better than gradient boosting. **Logistic Regression is clearly worse on every metric** — expected, since it is a linear discriminator forced onto the majority-fail binary label while the other two regress directly on the continuous failure frequency, and the informative regime is exactly where a linear decision boundary is least likely to hold.

**Practical takeaway**: XGBoost remains a reasonable default (it's the model wired through calibration, allocation, and SHAP analysis throughout this project), but Random Forest is a legitimate, currently under-exploited alternative worth calibration attention in any follow-up work — the two are statistically tied on the metric that actually drives allocation quality (ranking), and RF's calibration numbers are better out of the box. See `results/model_comparison/` for the full per-config, per-regime breakdown.

### Distribution Shift / Transfer Radius Results

Models are trained on one substitution regime and evaluated on another **without retraining**. The formal transfer radius — the largest substitution-rate step at which every tested pair still passes both an ECE-ratio and a regime-aware AUROC-drop threshold — is:

| Stratum | Direction | Transfer radius | First failing step |
|---|---|---|---|
| K=3, simple | up / down | 0.000 | 2% |
| K=3, constrained | up / down | 0.000 | 2% |
| K=5, simple | up / down | 0.000 | 2% |
| K=5, constrained | up / down | 0.000 | 2% |

**Models do not transfer across substitution rates without retraining** — the transfer radius is 0.000 in every stratum and direction tested, failing even at the smallest tested step (2%). This is now measured on a leak-free canonical split (no sequence is ever shared between a source config's training set and a target config's "unseen" test set — see Fixed Issues below), so it's a clean test of generalization to genuinely unseen sequences, not just unseen substitution rates. See `results/transfer_radius/` for the full all-pairs sweep and `results/distribution_shift/` for the 12-condition per-stratum breakdown.

### Known Limitations

- **Encoding comparison is largely confounded.** A raw comparison of simple vs. constrained encoding shows constrained encoding failing less often, but post-stratifying on GC content and homopolymer run length (the `encoding_confound` stage) shows most of that raw effect disappears or reverses once composition is controlled for — see `results/encoding_confound/`.
- **The deployed (risk-model) allocation underperforms its theoretical ceiling** on roughly a quarter of configs (see Allocation Results) because it optimizes the wrong target (failure risk, not marginal benefit of added parity) — quantified, not hidden, and directly addressed by the Benefit-Aware Model above.
- **The benefit-aware model's improvement is delta-dependent, not a strict win at every step size.** At Δ=2 it beats the risk model on both mean OFR reduction and meaningful-loss count. At Δ=4 its mean improvement is larger but its meaningful-loss count is also larger — a genuine higher-mean/higher-variance trade-off, mechanistically explained by Δ acting as a leverage dial on both correct and incorrect ranking decisions (see Delta as a Leverage Dial above), not a sign the model itself got worse (its rank correlation with true benefit is *better* at Δ=4).

### Fixed Issues (resolved before manuscript writing)

An external code-level audit surfaced several real bugs, each verified against the code and fixed:

- **XGBoost/Random Forest were silently replaced by a trivial constant model in 15 of 28 configs.** The single-class training guard checked a crude binary (≥0.5) cutoff instead of the continuous target's actual variation, discarding real, learnable signal whenever every sequence happened to land on one side of that cutoff. Fixed in `models/train.py`; all 28 configs now train real regressors.
- **A related bug inverted the risk score for 4 near-100%-failure configs** (`sub18_k3_*`, `sub20_k3_*`), reporting 0% failure probability for sequences that fail almost every time. Fixed in `models/calibrate.py`.
- **The same physical sequence could appear in one config's training set and a different config's "held-out" test set**, since each of the 14 substitution-rate/coverage combinations per encoding drew its own independent split from the same underlying 2,000 sequences (up to 72% overlap measured in one pair before the fix). Fixed with one canonical split per encoding scheme in `src/dataset_assembler.py`, shared across every condition.
- **`gate_check` was comparing the model's continuous prediction against a coarsened binary target** — a semantic mismatch, not a real calibration failure. Once fixed to compare like-for-like, the default gate config passes cleanly (ECE=0.024, well under the 0.05 threshold).
- **The oracle's benefit/harm estimates used three independently resimulated channel-noise draws** instead of one shared draw evaluated at three parity thresholds, needlessly inflating the Monte Carlo noise floor. Fixed with a paired estimation design (see Allocation Results above).
- **The "deployable" model secretly used the oracle's own privileged budget size.** Fixed with a separate, validation-only budget selection (see Allocation Results above).

---

## Key Design Decisions

- **Calibration is a reliability-estimation contribution, not an allocation-quality one**: probability calibration (Platt scaling for XGBoost, isotonic for RF) matters for the model's probability estimates being trustworthy on their own terms — but since the allocation mechanism selects promotion/demotion tiers by *rank* alone, and Platt scaling is monotonic, calibrated and raw scores produce byte-identical allocations under a fixed budget (confirmed directly: `ofr_xgb_cal` and `ofr_xgb_raw` are identical whenever they share a budget). Calibration's value here is in the ECE/Brier/gate-check numbers, not in the allocation outcome itself.
- **Degeneracy guard uses the actual continuous target**: XGBoost/RF fall back to a trivial constant model only when `failure_freq` itself has no real variation (not when its binarized ≥0.5 version happens to be single-class) — the earlier version of this guard was the source of the 15/28-config dummy-model bug described in Fixed Issues.
- **Canonical, leak-free splits**: one train/val/test split per encoding scheme, built from sequence identity and a fixed seed, shared across every substitution-rate/coverage condition for that encoding — not stratified by any per-condition quantity like failure_freq, which is what let splits diverge across conditions in the first place.
- **Budget neutrality**: the allocation mechanism strictly enforces that total parity bytes added equals total parity bytes removed — no free lunch
- **Monte Carlo evaluation**: OFR is estimated over 30 independent channel simulation runs per configuration to account for stochastic variation
- **Noise-aware oracle allocation**: the oracle doesn't just rank sequences by estimated marginal benefit — it only reallocates parity when the estimated benefit of promoting one sequence exceeds the estimated harm of demoting another by a statistically-derived margin (not just "benefit > harm"). Naively ranking on point estimates from 30-run Monte Carlo samples is a textbook winner's-curse setup — the top/bottom of a noisy ranking is disproportionately luck, not signal.
- **Paired (common-random-numbers) oracle estimation**: the oracle's low/default/high-parity outcomes are estimated from one shared channel-noise draw per run rather than three independent ones, cancelling shared noise instead of compounding it.
- **Deployable vs. diagnostic allocation, reported separately**: the model's reallocation budget size is chosen on validation data only (a small grid, frozen before test evaluation) for the deployable claim; the oracle-sized-budget comparison is retained only as a ranking-quality diagnostic and is never conflated with the deployable result.

---

## Configuration

All experiment parameters are in [`configs/experiment_config.yaml`](configs/experiment_config.yaml), including substitution rates, coverage depths, model hyperparameter grids, RS thresholds, and allocation delta values.
