"""
Parallel launcher for the remaining dataset_assembler.py configs.

The sequential path (dataset_assembler.build_all_datasets) generates
sequences+features+split ONCE per encoding, then loops label computation
(the expensive part -- M=30 Monte Carlo channel-simulation runs per config)
over all 14 sub_rate x coverage combos for that encoding, one at a time.
That per-config label computation is the only part worth parallelizing --
sequence/feature/split generation is deterministic (fixed seed) and cheap
by comparison, so this script does it once per encoding (cached to disk),
then fans out label computation across worker processes.

Each worker calls the identical channel-model / compute_failure_labels code
path as the sequential version, with the same seed, so output is bit-for-bit
identical to what the sequential run would have produced -- this only
changes wall-clock time, not results.

Resumable: skips any (key) whose parquet already exists, same convention as
dataset_assembler.py and the other scratch_parallel_*.py launchers.

Usage:
    python scratch_parallel_datasets.py [--workers 5]
"""
import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))


def _config_key(sub_rate, coverage, encoding):
    return f'sub{int(sub_rate*100):02d}_k{coverage}_{encoding}'


def _ensure_encoding_assets(encoding, cfg):
    """Generate (if missing) and return cached sequences + features + split
    for one encoding scheme. Sequences are cached as the same .txt file
    dataset_assembler.py already uses; features are cached to a new .npy
    file (dataset_assembler.py never persisted these, it just recomputed
    them once per run -- caching them here just avoids paying that cost
    once per worker instead of once per run)."""
    from sequence_generator import generate_sequences
    from feature_extractor import extract_features

    seq_cfg = cfg['sequence']
    n = seq_cfg['n_sequences']
    seed = cfg['random_seed']

    seq_path = os.path.join(cfg['paths']['sequences_dir'], f'seqs_{encoding}_n{n}.txt')
    feat_path = os.path.join(cfg['paths']['sequences_dir'], f'feats_{encoding}_n{n}.npy')
    feat_names_path = os.path.join(cfg['paths']['sequences_dir'], f'feats_{encoding}_n{n}_names.json')

    if os.path.exists(seq_path):
        with open(seq_path) as f:
            lines = f.readlines()
        sequences = [l.strip() for l in lines if not l.startswith('#') and l.strip()]
    else:
        seqs_with_payload = generate_sequences(
            n, seq_cfg['seq_len_bases'], encoding,
            max_homopolymer=seq_cfg['max_homopolymer'],
            gc_min=seq_cfg['gc_min'], gc_max=seq_cfg['gc_max'],
            seed=seed,
        )
        sequences = [s for s, _ in seqs_with_payload]
        os.makedirs(cfg['paths']['sequences_dir'], exist_ok=True)
        with open(seq_path, 'w') as f:
            f.write('# dna_sequence\n')
            for seq in sequences:
                f.write(seq + '\n')

    if os.path.exists(feat_path) and os.path.exists(feat_names_path):
        X = np.load(feat_path)
        with open(feat_names_path) as f:
            feat_names = json.load(f)
    else:
        X, feat_names = extract_features(sequences)
        np.save(feat_path, X)
        with open(feat_names_path, 'w') as f:
            json.dump(feat_names, f)

    return sequences, X, feat_names


def _build_canonical_split(n_sequences, split_cfg, seed):
    """Identical logic to dataset_assembler._build_canonical_split -- kept
    duplicated here (not imported) so this file has no import-time
    dependency beyond src/, matching the other scratch_parallel_*.py files."""
    import pandas as pd

    train_f = split_cfg['train_frac']
    val_f = split_cfg['val_frac']

    rng = np.random.default_rng(seed)
    order = rng.permutation(n_sequences)
    n_train = max(1, int(n_sequences * train_f))
    n_val = max(1, int(n_sequences * val_f))

    train_idx = order[:n_train]
    val_idx = order[n_train:n_train + n_val]
    test_idx = order[n_train + n_val:]

    splits = pd.DataFrame({'index': list(range(n_sequences)), 'split': ['train'] * n_sequences})
    for idx in val_idx:
        splits.at[idx, 'split'] = 'val'
    for idx in test_idx:
        splits.at[idx, 'split'] = 'test'
    return splits


def _worker(sub_rate, coverage, encoding, cfg):
    """Runs in a separate process: compute labels for ONE config and save
    its parquet + split files. Re-derives sequences/features/split from the
    cached files written by _ensure_encoding_assets (called up front in the
    main process before workers are spawned, so no race on first-write)."""
    import pandas as pd
    from channel_model import build_channel_from_config
    from label_generator import compute_failure_labels

    key = _config_key(sub_rate, coverage, encoding)
    out_path = os.path.join(cfg['paths']['datasets_dir'], f'{key}.parquet')
    if os.path.exists(out_path):
        return key, 'skip', None, None

    sequences, X, feat_names = _ensure_encoding_assets(encoding, cfg)
    seed = cfg['random_seed']
    l_rs = cfg['sequence']['l_rs_default']
    n_runs = cfg['allocation']['n_monte_carlo_runs']

    canonical_split = _build_canonical_split(len(sequences), cfg['splits'], seed)

    t0 = time.time()
    channel = build_channel_from_config(cfg, sub_rate, seed=seed)
    failure_freq, byte_errors_mean = compute_failure_labels(
        sequences, channel, coverage, l_rs, n_runs=n_runs, base_seed=seed, verbose=False
    )

    df = pd.DataFrame(X, columns=feat_names)
    df.insert(0, 'dna_sequence', sequences)
    df.insert(1, 'failure_freq', failure_freq)
    df.insert(2, 'byte_errors_mean', byte_errors_mean)
    df.insert(3, 'label_binary', (failure_freq >= 0.5).astype(int))

    os.makedirs(cfg['paths']['datasets_dir'], exist_ok=True)
    df.to_parquet(out_path, index=False)

    splits_path = os.path.join(cfg['paths']['splits_dir'], f'{key}_splits.parquet')
    os.makedirs(cfg['paths']['splits_dir'], exist_ok=True)
    canonical_split.to_parquet(splits_path, index=False)

    fail_rate = (failure_freq >= 0.5).mean()
    elapsed = time.time() - t0
    return key, 'done', fail_rate, elapsed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/experiment_config.yaml')
    parser.add_argument('--workers', type=int, default=5)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    sub_rates = cfg['channel']['substitution_rates']
    coverages = cfg['coverage_depths']
    encodings = cfg['sequence']['encoding_schemes']

    all_combos = [(s, k, e) for e in encodings for s in sub_rates for k in coverages]

    remaining = []
    for s, k, e in all_combos:
        key = _config_key(s, k, e)
        out_path = os.path.join(cfg['paths']['datasets_dir'], f'{key}.parquet')
        if not os.path.exists(out_path):
            remaining.append((s, k, e))

    print(f"[parallel_datasets] {len(all_combos) - len(remaining)} already done, "
          f"{len(remaining)} remaining, {args.workers} workers")
    if not remaining:
        print("[parallel_datasets] Nothing to do.")
        return

    # Pre-build assets for every encoding that still has remaining work,
    # sequentially, in THIS process -- so workers never race on the same
    # sequences/features cache file.
    needed_encodings = sorted(set(e for _, _, e in remaining))
    for e in needed_encodings:
        print(f"[parallel_datasets] Ensuring cached sequences+features for encoding={e} ...")
        t0 = time.time()
        _ensure_encoding_assets(e, cfg)
        print(f"  done in {time.time()-t0:.1f}s")

    t_start = time.time()
    done_count = 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(_worker, s, k, e, cfg): (s, k, e) for s, k, e in remaining}
        for fut in as_completed(futures):
            s, k, e = futures[fut]
            key = _config_key(s, k, e)
            try:
                key2, status, fail_rate, elapsed = fut.result()
                done_count += 1
                if status == 'done':
                    print(f"  [{done_count}/{len(remaining)}] {key}: OK failure_rate={fail_rate:.4f} "
                          f"[{elapsed:.1f}s]")
                else:
                    print(f"  [{done_count}/{len(remaining)}] {key}: already existed, skipped")
            except Exception as exc:
                done_count += 1
                print(f"  [{done_count}/{len(remaining)}] {key}: FAILED - {exc}")

    print(f"[parallel_datasets] Done in {(time.time()-t_start)/60:.1f} min")


if __name__ == '__main__':
    main()
