"""
label_generator.py
==================
Compute empirical sequence-level integrity-failure labels by running each
sequence through the channel model K independent times and measuring how
often RS correction fails.

Label definition (from the implementation plan):
    For each sequence, run K independent channel simulations.
    In each run: apply channel → consensus vote → count byte errors → check
    failure condition (byte_errors > floor(L_RS / 2)).
    Label = fraction of K runs that result in failure ∈ [0, 1].

These soft probabilistic labels serve as regression targets for model training
and also generate binary labels (threshold at 0.5) for classification metrics.

R8 note — label noise at M=30
------------------------------
With M=30 independent runs, a sequence whose true failure probability is p has
empirical failure_freq drawn from Binomial(30, p)/30.  At p=0.5 the standard
error is 0.5/√30 ≈ 0.091, so the 95% Wilson CI spans ±0.18 around the
threshold.  Sequences with true p ∈ (0.32, 0.68) may be mislabelled by the
binary 0.5 rule.

Mitigations implemented here:
  - `label_confidence_intervals`: Wilson score 95% CI per sequence.
  - `near_threshold_stats`: fraction of sequences with uncertain labels.
  - `save_labels`: stores `label_ci_lo`, `label_ci_hi`, `near_threshold` columns.
  - Use `failure_freq` (continuous) as the primary regression target; binary
    labels are secondary.  See `analysis/threshold_sensitivity.py` for a sweep
    of F1/Brier across thresholds 0.1–0.9.

Usage (CLI):
    python label_generator.py --config configs/experiment_config.yaml \
        --seqs data/sequences/seqs_simple_n2000.txt \
        --sub-rate 0.05 --coverage 30 --l-rs 8 \
        --out data/datasets/labels_simple_sub005_k30.parquet
"""

import argparse
import os
from typing import List, Optional, Tuple

import numpy as np
import yaml

# Lazy imports to avoid circular dependency when used as a library
_voter_module  = None
_channel_module = None


def _import_modules():
    global _voter_module, _channel_module
    if _voter_module is None:
        import sys; sys.path.insert(0, os.path.dirname(__file__))
        from consensus_voter import majority_vote_consensus, count_errors, is_decoding_failure
        from channel_model import DNAStorageChannel, build_channel_from_config
        _voter_module  = (majority_vote_consensus, count_errors, is_decoding_failure)
        _channel_module = (DNAStorageChannel, build_channel_from_config)


def compute_failure_labels(
    sequences: List[str],
    channel,
    coverage: int,
    l_rs: int,
    n_runs: int = 30,
    base_seed: int = 0,
    verbose: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute empirical failure frequency labels for a list of sequences.

    Parameters
    ----------
    sequences  : list of DNA sequences (strings)
    channel    : DNAStorageChannel instance
    coverage   : K reads per simulation run
    l_rs       : RS parity bytes per sequence (determines correction capacity)
    n_runs     : M independent runs per sequence (M=30 per implementation plan)
    base_seed  : seed offset; each run uses seed = base_seed + run_idx
    verbose    : print progress every 100 sequences

    Returns
    -------
    failure_freq : ndarray shape (N,) — empirical failure probability ∈ [0, 1]
    byte_errors_mean : ndarray shape (N,) — mean byte errors across runs
    """
    _import_modules()
    majority_vote_consensus, count_errors, is_decoding_failure = _voter_module

    N = len(sequences)
    failure_counts    = np.zeros(N, dtype=np.float64)
    byte_errors_total = np.zeros(N, dtype=np.float64)

    for run_idx in range(n_runs):
        # Clone channel with a unique seed for each run
        run_channel = channel.clone(seed=base_seed + run_idx)

        for seq_idx, orig in enumerate(sequences):
            reads     = run_channel.simulate(orig, coverage)
            consensus = majority_vote_consensus(reads, len(orig))
            stats     = count_errors(orig, consensus)
            byte_err  = stats['byte_errors']

            byte_errors_total[seq_idx] += byte_err
            if is_decoding_failure(byte_err, l_rs):
                failure_counts[seq_idx] += 1

        if verbose and (run_idx + 1) % 5 == 0:
            print(f"  [label_generator] Run {run_idx + 1}/{n_runs} complete")

    failure_freq      = failure_counts / n_runs
    byte_errors_mean  = byte_errors_total / n_runs
    return failure_freq, byte_errors_mean


def sanity_check(
    sequences: List[str],
    channel,
    coverage: int,
    l_rs: int,
    n_runs: int = 10,
) -> dict:
    """Run the go/no-go sanity checks from the implementation plan (Week 4).

    Expected results:
      - K=30, 1% sub  → consensus byte error rate < 0.1%
      - K=10, 12% sub → failure rate ≈ 3–5% (some failures expected)

    Returns a dict with 'pass', 'mean_byte_errors', 'failure_rate'.
    """
    failure_freq, byte_errors_mean = compute_failure_labels(
        sequences[:min(100, len(sequences))],
        channel, coverage, l_rs, n_runs
    )
    failure_rate      = float(failure_freq.mean())
    mean_byte_errors  = float(byte_errors_mean.mean())

    # Heuristic pass thresholds
    if coverage >= 30 and channel.sub_rate <= 0.01:
        passed = failure_rate < 0.01 and mean_byte_errors < 1.0
    else:
        passed = failure_rate > 0.01  # some failures expected at high error rates

    return {
        'pass'            : passed,
        'failure_rate'    : failure_rate,
        'mean_byte_errors': mean_byte_errors,
        'coverage'        : coverage,
        'sub_rate'        : channel.sub_rate,
        'l_rs'            : l_rs,
    }


def label_confidence_intervals(
    failure_freq: np.ndarray,
    n_runs: int,
    alpha: float = 0.05,
) -> Tuple[np.ndarray, np.ndarray]:
    """Wilson score confidence intervals for per-sequence failure_freq estimates.

    Each failure_freq is an empirical proportion k/n_runs from Bernoulli trials.
    At M=30 and p≈0.5 the 95% CI half-width is ≈ ±0.18, so sequences with
    true p ∈ (0.32, 0.68) have genuinely uncertain binary labels.

    Parameters
    ----------
    failure_freq : ndarray (N,) — empirical failure fraction ∈ [0, 1]
    n_runs       : M — number of independent MC trials
    alpha        : significance level (0.05 → 95% CI)

    Returns
    -------
    ci_lo, ci_hi : ndarrays (N,) — lower and upper Wilson score CI bounds
    """
    from scipy.stats import norm  # type: ignore
    z = float(norm.ppf(1.0 - alpha / 2.0))

    p = np.clip(np.asarray(failure_freq, dtype=float), 0.0, 1.0)
    n = int(n_runs)
    denom  = 1.0 + z ** 2 / n
    center = (p + z ** 2 / (2.0 * n)) / denom
    half   = (z / denom) * np.sqrt(p * (1.0 - p) / n + z ** 2 / (4.0 * n ** 2))

    ci_lo = np.clip(center - half, 0.0, 1.0)
    ci_hi = np.clip(center + half, 0.0, 1.0)
    return ci_lo, ci_hi


def near_threshold_stats(
    failure_freq: np.ndarray,
    n_runs: int,
    threshold: float = 0.5,
    alpha: float = 0.05,
) -> dict:
    """Report what fraction of sequences have genuinely uncertain binary labels.

    A sequence is "near-threshold" when its (1−alpha) Wilson CI spans the
    threshold, meaning the M Monte Carlo runs do not provide enough evidence to
    confidently assign a binary label.

    Parameters
    ----------
    failure_freq : ndarray (N,) — empirical failure fraction
    n_runs       : M — Monte Carlo runs used to produce failure_freq
    threshold    : binary classification threshold (typically 0.5)
    alpha        : significance level for the CI (0.05 → 95%)

    Returns
    -------
    dict with keys: n_total, n_near_threshold, frac_near, ci_width_mean,
    ci_width_near_threshold, threshold, n_runs, alpha
    """
    ci_lo, ci_hi = label_confidence_intervals(failure_freq, n_runs, alpha)
    spans_threshold = (ci_lo < threshold) & (ci_hi >= threshold)
    n_near  = int(spans_threshold.sum())
    n_total = len(failure_freq)

    ci_width = ci_hi - ci_lo
    near_mask = np.abs(failure_freq - threshold) <= 0.15
    ci_at_thresh = (float(ci_width[near_mask].mean())
                    if near_mask.any() else float('nan'))

    return {
        'n_total'                 : n_total,
        'n_near_threshold'        : n_near,
        'frac_near'               : float(n_near / n_total) if n_total > 0 else 0.0,
        'ci_width_mean'           : float(ci_width.mean()),
        'ci_width_near_threshold' : ci_at_thresh,
        'threshold'               : threshold,
        'n_runs'                  : n_runs,
        'alpha'                   : alpha,
    }


def load_sequences(path: str) -> List[str]:
    """Load sequences from a tab-separated file (first column = DNA string)."""
    seqs = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            seqs.append(line.split('\t')[0])
    return seqs


def save_labels(
    sequences: List[str],
    failure_freq: np.ndarray,
    byte_errors_mean: np.ndarray,
    out_path: str,
    metadata: Optional[dict] = None,
    n_runs: int = 30,
    label_threshold: float = 0.5,
):
    """Save labeled dataset to a Parquet file (pandas required).

    Stores Wilson score 95% CI columns so downstream analysis can quantify
    label uncertainty without re-running the channel (R8 mitigation).

    Parameters
    ----------
    n_runs          : MC runs used — required for CI computation
    label_threshold : threshold for `label_binary`; 0.5 is a working convention
                      with no operational grounding (acknowledged per R8)
    """
    import pandas as pd  # type: ignore
    ci_lo, ci_hi = label_confidence_intervals(failure_freq, n_runs)
    df = pd.DataFrame({
        'dna_sequence'    : sequences,
        'failure_freq'    : failure_freq,
        'byte_errors_mean': byte_errors_mean,
        'label_binary'    : (failure_freq >= label_threshold).astype(int),
        'label_ci_lo'     : ci_lo,
        'label_ci_hi'     : ci_hi,
        'label_ci_width'  : ci_hi - ci_lo,
        'near_threshold'  : ((ci_lo < label_threshold) & (ci_hi >= label_threshold)).astype(int),
    })
    if metadata:
        for k, v in metadata.items():
            df.attrs[k] = v

    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    if out_path.endswith('.parquet'):
        df.to_parquet(out_path, index=False)
    else:
        df.to_csv(out_path, index=False)
    return df


# ── CLI entry point ──────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Generate empirical failure-frequency labels for DNA sequences.'
    )
    parser.add_argument('--config',    default='configs/experiment_config.yaml')
    parser.add_argument('--seqs',      required=True, help='Path to sequences .txt file')
    parser.add_argument('--sub-rate',  type=float, required=True)
    parser.add_argument('--coverage',  type=int, required=True)
    parser.add_argument('--l-rs',      type=int, default=None)
    parser.add_argument('--n-runs',    type=int, default=None)
    parser.add_argument('--out',       required=True)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    _import_modules()
    _, build_channel_from_config = _channel_module

    l_rs   = args.l_rs   or cfg['sequence']['l_rs_default']
    n_runs = args.n_runs or cfg['allocation']['n_monte_carlo_runs']

    channel   = build_channel_from_config(cfg, args.sub_rate, seed=cfg['random_seed'])
    sequences = load_sequences(args.seqs)

    print(f"[label_generator] {len(sequences)} sequences | "
          f"sub={args.sub_rate} | K={args.coverage} | L_RS={l_rs} | M={n_runs}")

    # Sanity check on a small subset first
    check = sanity_check(sequences, channel, args.coverage, l_rs)
    print(f"  Sanity check → pass={check['pass']}, "
          f"failure_rate={check['failure_rate']:.4f}, "
          f"mean_byte_errors={check['mean_byte_errors']:.3f}")

    failure_freq, byte_errors_mean = compute_failure_labels(
        sequences, channel, args.coverage, l_rs, n_runs, verbose=True
    )

    meta = dict(sub_rate=args.sub_rate, coverage=args.coverage, l_rs=l_rs, n_runs=n_runs)
    df = save_labels(sequences, failure_freq, byte_errors_mean, args.out, meta,
                     n_runs=n_runs)

    nt = near_threshold_stats(failure_freq, n_runs)
    print(f"[label_generator] Saved {len(df)} rows to {args.out}")
    print(f"  Label stats: mean={failure_freq.mean():.4f}, "
          f"std={failure_freq.std():.4f}, "
          f"frac>0 = {(failure_freq > 0).mean():.3f}, "
          f"frac>=0.5 = {(failure_freq >= 0.5).mean():.3f}")
    print(f"  Label uncertainty (R8): near-threshold = {nt['n_near_threshold']}/{nt['n_total']} "
          f"({nt['frac_near']:.1%}), mean 95%-CI width = {nt['ci_width_mean']:.3f}")


if __name__ == '__main__':
    main()
