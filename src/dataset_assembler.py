"""
dataset_assembler.py
====================
Generates all 16 dataset configurations and writes them to Parquet files:
  4 substitution regimes × 2 coverage depths × 2 encoding schemes = 16 configs

Each Parquet file contains:
  - dna_sequence       : DNA string
  - failure_freq       : soft label in [0, 1] (empirical failure probability)
  - byte_errors_mean   : mean byte errors across M=30 runs
  - label_binary       : hard binary label (threshold at 0.5)
  - <feature columns>  : full feature matrix

Splits (70/15/15 stratified by failure_freq bins) are saved as separate index
files so that the exact same train/val/test partition is reusable across all
downstream experiments.

Usage (CLI):
    python dataset_assembler.py --config configs/experiment_config.yaml
"""

import argparse
import os
import sys
import hashlib
from typing import List, Tuple

import numpy as np
import yaml

# Adjust path for direct execution
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def _config_key(sub_rate: float, coverage: int, encoding: str) -> str:
    """Canonical identifier string for a dataset configuration."""
    return f'sub{int(sub_rate*100):02d}_k{coverage}_{encoding}'


def build_all_datasets(cfg: dict, verbose: bool = True):
    """Build all 16 dataset configurations.

    Generates sequences, runs the channel model, computes labels and features,
    and saves each configuration to a Parquet file.
    """
    from sequence_generator import generate_sequences
    from channel_model import build_channel_from_config
    from label_generator import compute_failure_labels
    from feature_extractor import extract_features

    try:
        import pandas as pd
    except ImportError:
        raise ImportError("pandas required: pip install pandas pyarrow")

    seq_cfg = cfg['sequence']
    n       = seq_cfg['n_sequences']
    seq_len = seq_cfg['seq_len_bases']
    l_rs    = seq_cfg['l_rs_default']
    seed    = cfg['random_seed']

    sub_rates  = cfg['channel']['substitution_rates']
    coverages  = cfg['coverage_depths']
    encodings  = seq_cfg['encoding_schemes']
    n_runs     = cfg['allocation']['n_monte_carlo_runs']

    os.makedirs(cfg['paths']['datasets_dir'], exist_ok=True)
    os.makedirs(cfg['paths']['splits_dir'],   exist_ok=True)
    os.makedirs(cfg['paths']['sequences_dir'], exist_ok=True)

    summary = []

    for encoding in encodings:
        if verbose:
            print(f"\n[dataset_assembler] Generating {n} sequences | encoding={encoding}")

        seqs_with_payload = generate_sequences(
            n, seq_len, encoding,
            max_homopolymer=seq_cfg['max_homopolymer'],
            gc_min=seq_cfg['gc_min'],
            gc_max=seq_cfg['gc_max'],
            seed=seed,
        )
        sequences = [s for s, _ in seqs_with_payload]

        # Extract features once per encoding scheme (features are channel-independent)
        if verbose:
            print(f"  Extracting features for {len(sequences)} sequences ...")
        X, feat_names = extract_features(sequences)

        # Save sequences
        seq_path = os.path.join(
            cfg['paths']['sequences_dir'], f'seqs_{encoding}_n{n}.txt'
        )
        with open(seq_path, 'w') as f:
            f.write('# dna_sequence\n')
            for seq in sequences:
                f.write(seq + '\n')

        for sub_rate in sub_rates:
            for coverage in coverages:
                key = _config_key(sub_rate, coverage, encoding)
                out_path = os.path.join(cfg['paths']['datasets_dir'], f'{key}.parquet')

                if os.path.exists(out_path):
                    if verbose:
                        print(f"  [SKIP] {key} already exists")
                    continue

                if verbose:
                    print(f"  Computing labels | sub={sub_rate:.2f} K={coverage} ...")

                channel = build_channel_from_config(cfg, sub_rate, seed=seed)
                failure_freq, byte_errors_mean = compute_failure_labels(
                    sequences, channel, coverage, l_rs,
                    n_runs=n_runs, base_seed=seed, verbose=False
                )

                # Assemble DataFrame
                df = pd.DataFrame(X, columns=feat_names)
                df.insert(0, 'dna_sequence',    sequences)
                df.insert(1, 'failure_freq',    failure_freq)
                df.insert(2, 'byte_errors_mean', byte_errors_mean)
                df.insert(3, 'label_binary',    (failure_freq >= 0.5).astype(int))

                df.to_parquet(out_path, index=False)

                # Generate and save stratified splits
                _save_splits(
                    df, key, failure_freq,
                    cfg['splits'],
                    cfg['paths']['splits_dir'],
                    seed
                )

                fail_rate = (failure_freq >= 0.5).mean()
                summary.append({
                    'config': key, 'n': n, 'failure_rate': fail_rate,
                    'mean_failure_freq': failure_freq.mean(),
                })
                if verbose:
                    print(f"  -> saved {key} | failure_rate={fail_rate:.4f} | "
                          f"mean_freq={failure_freq.mean():.4f}")

    # Print summary table
    if verbose and summary:
        print("\n[dataset_assembler] Summary:")
        print(f"  {'Config':<30} {'N':>6} {'FailRate':>10} {'MeanFreq':>10}")
        for row in summary:
            print(f"  {row['config']:<30} {row['n']:>6} "
                  f"{row['failure_rate']:>10.4f} {row['mean_failure_freq']:>10.4f}")

    return summary


def _save_splits(
    df,
    key: str,
    failure_freq: np.ndarray,
    split_cfg: dict,
    splits_dir: str,
    seed: int,
):
    """Save train/val/test index splits using stratified binning."""
    import pandas as pd

    train_f = split_cfg['train_frac']
    val_f   = split_cfg['val_frac']
    n_bins  = split_cfg['stratify_bins']

    N      = len(df)
    bins   = np.linspace(0, 1, n_bins + 1)
    bin_id = np.digitize(failure_freq, bins) - 1
    bin_id = np.clip(bin_id, 0, n_bins - 1)

    rng = np.random.default_rng(seed)
    train_idx, val_idx, test_idx = [], [], []

    for b in range(n_bins):
        idx = np.where(bin_id == b)[0]
        if len(idx) == 0:
            continue
        idx = rng.permutation(idx)
        n_train = max(1, int(len(idx) * train_f))
        n_val   = max(1, int(len(idx) * val_f))
        train_idx.extend(idx[:n_train])
        val_idx.extend(idx[n_train:n_train + n_val])
        test_idx.extend(idx[n_train + n_val:])

    # Verify: no overlap between splits
    assert len(set(train_idx) & set(test_idx)) == 0, "Train/test overlap!"
    assert len(set(val_idx)   & set(test_idx)) == 0, "Val/test overlap!"

    splits = pd.DataFrame({
        'index': list(range(N)),
        'split': ['train'] * N,
    })
    for idx in val_idx:
        splits.at[idx, 'split'] = 'val'
    for idx in test_idx:
        splits.at[idx, 'split'] = 'test'

    splits_path = os.path.join(splits_dir, f'{key}_splits.parquet')
    splits.to_parquet(splits_path, index=False)


def load_dataset(
    key: str,
    cfg: dict,
) -> Tuple:
    """Load a pre-built dataset and return (X_train, X_val, X_test, y_train, y_val, y_test, feature_names).

    Labels are soft (failure_freq) by default.  Use label_binary for hard labels.
    """
    import pandas as pd

    data_path   = os.path.join(cfg['paths']['datasets_dir'], f'{key}.parquet')
    splits_path = os.path.join(cfg['paths']['splits_dir'],   f'{key}_splits.parquet')

    df     = pd.read_parquet(data_path)
    splits = pd.read_parquet(splits_path)

    meta_cols  = ['dna_sequence', 'failure_freq', 'byte_errors_mean', 'label_binary']
    feat_names = [c for c in df.columns if c not in meta_cols]

    X = df[feat_names].values
    y = df['failure_freq'].values

    for split in ['train', 'val', 'test']:
        idx = splits[splits['split'] == split]['index'].values
        assert len(set(idx) & set(splits[splits['split'] != split]['index'].values)) == 0

    train_idx = splits[splits['split'] == 'train']['index'].values
    val_idx   = splits[splits['split'] == 'val'  ]['index'].values
    test_idx  = splits[splits['split'] == 'test' ]['index'].values

    return (
        X[train_idx], X[val_idx], X[test_idx],
        y[train_idx], y[val_idx], y[test_idx],
        feat_names,
    )


def get_all_config_keys(cfg: dict) -> List[str]:
    """Return all 16 configuration key strings."""
    sub_rates = cfg['channel']['substitution_rates']
    coverages = cfg['coverage_depths']
    encodings = cfg['sequence']['encoding_schemes']
    return [
        _config_key(s, k, e)
        for s in sub_rates for k in coverages for e in encodings
    ]


# -- CLI entry point ----------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Build all 16 dataset configurations for the DNA storage benchmark.'
    )
    parser.add_argument('--config', default='configs/experiment_config.yaml')
    parser.add_argument('--verbose', action='store_true', default=True)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    build_all_datasets(cfg, verbose=args.verbose)


if __name__ == '__main__':
    main()
