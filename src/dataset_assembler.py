"""
dataset_assembler.py
====================
Generates all dataset configurations and writes them to Parquet files:
  7 substitution regimes × 2 coverage depths × 2 encoding schemes = 28 configs

Each Parquet file contains:
  - dna_sequence       : DNA string
  - failure_freq       : soft label in [0, 1] (empirical failure probability)
  - byte_errors_mean   : mean byte errors across M=30 runs
  - label_binary       : hard binary label (threshold at 0.5)
  - <feature columns>  : full feature matrix

Splits (70/15/15) are computed ONCE per encoding scheme from sequence identity
and a fixed seed -- NOT stratified by failure_freq or any other per-condition
quantity -- and saved as separate index files so every substitution-rate/
coverage condition under a given encoding shares the exact same train/val/test
membership. (Stratifying per-condition was the original approach and caused
cross-config leakage: the same physical sequence could be "seen" in one
config's training set and "held out" in another's test set. See
_build_canonical_split() below.)

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
    """Build all dataset configurations.

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

        # Canonical split: computed ONCE per encoding, from sequence identity
        # (row position in `sequences`, which is fixed and shared across every
        # substitution-rate/coverage combo for this encoding) and a fixed seed
        # -- not from any per-condition failure_freq. Every config under this
        # encoding reuses the exact same train/val/test membership, so a
        # sequence can never be "seen" by one config's training set and
        # "held out" in another's test set (fix for cross-config leakage).
        canonical_split = _build_canonical_split(len(sequences), cfg['splits'], seed)

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

                # Save this config's copy of the encoding's canonical split
                # (identical membership across every config sharing this
                # encoding -- see _build_canonical_split above).
                _save_canonical_split_copy(
                    canonical_split, key, cfg['paths']['splits_dir']
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


def _build_canonical_split(
    n_sequences: int,
    split_cfg:   dict,
    seed:        int,
):
    """Build one train/val/test index split, shared across every
    substitution-rate/coverage config under a given encoding scheme.

    Deliberately NOT stratified by failure_freq (or any other per-condition
    quantity) -- stratifying by failure_freq is exactly what caused each
    config to draw a different split from the same underlying sequences,
    letting a sequence be "seen" in one config's training set and "held out"
    in another's test set. A plain fixed-seed random split over sequence
    identity (row position) has no such dependency, so it is identical no
    matter which condition asks for it.

    Returns
    -------
    pandas.DataFrame with columns 'index' (0..n_sequences-1) and 'split'
    ('train' / 'val' / 'test').
    """
    import pandas as pd

    train_f = split_cfg['train_frac']
    val_f   = split_cfg['val_frac']

    rng   = np.random.default_rng(seed)
    order = rng.permutation(n_sequences)
    n_train = max(1, int(n_sequences * train_f))
    n_val   = max(1, int(n_sequences * val_f))

    train_idx = order[:n_train]
    val_idx   = order[n_train:n_train + n_val]
    test_idx  = order[n_train + n_val:]

    assert len(set(train_idx) & set(test_idx)) == 0, "Train/test overlap!"
    assert len(set(val_idx)   & set(test_idx)) == 0, "Val/test overlap!"
    assert len(set(train_idx) & set(val_idx))  == 0, "Train/val overlap!"

    splits = pd.DataFrame({
        'index': list(range(n_sequences)),
        'split': ['train'] * n_sequences,
    })
    for idx in val_idx:
        splits.at[idx, 'split'] = 'val'
    for idx in test_idx:
        splits.at[idx, 'split'] = 'test'
    return splits


def _save_canonical_split_copy(canonical_split, key: str, splits_dir: str):
    """Persist this config's copy of the encoding's canonical split.

    Every config sharing an encoding writes an identical copy -- kept as a
    per-config file so load_dataset() and every downstream loader can stay
    unchanged, but the *content* is now guaranteed consistent across configs.
    """
    splits_path = os.path.join(splits_dir, f'{key}_splits.parquet')
    canonical_split.to_parquet(splits_path, index=False)


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


def load_dataset_sequences(key: str, cfg: dict) -> Tuple:
    """Load raw DNA sequences (not the engineered feature matrix) for one
    dataset config, split with the exact same train/val/test membership
    `load_dataset` uses -- for models (e.g. a CNN over one-hot sequence)
    that learn from the sequence directly instead of the ~80 hand-crafted
    features, while remaining directly comparable to the feature-based
    models via an identical split.

    Returns
    -------
    (seq_train, seq_val, seq_test, y_train, y_val, y_test) -- sequences as
    lists of strings, labels (failure_freq) as ndarrays.
    """
    import pandas as pd

    data_path   = os.path.join(cfg['paths']['datasets_dir'], f'{key}.parquet')
    splits_path = os.path.join(cfg['paths']['splits_dir'],   f'{key}_splits.parquet')

    df     = pd.read_parquet(data_path)
    splits = pd.read_parquet(splits_path)

    sequences = df['dna_sequence'].values
    y         = df['failure_freq'].values

    train_idx = splits[splits['split'] == 'train']['index'].values
    val_idx   = splits[splits['split'] == 'val'  ]['index'].values
    test_idx  = splits[splits['split'] == 'test' ]['index'].values

    return (
        list(sequences[train_idx]), list(sequences[val_idx]), list(sequences[test_idx]),
        y[train_idx], y[val_idx], y[test_idx],
    )


def get_all_config_keys(cfg: dict) -> List[str]:
    """Return all configuration key strings."""
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
        description='Build all dataset configurations for the DNA storage benchmark.'
    )
    parser.add_argument('--config', default='configs/experiment_config.yaml')
    parser.add_argument('--verbose', action='store_true', default=True)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    build_all_datasets(cfg, verbose=args.verbose)


if __name__ == '__main__':
    main()
