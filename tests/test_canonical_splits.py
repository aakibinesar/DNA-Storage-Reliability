"""
Regression tests for the cross-config split-leakage bug fixed in
src/dataset_assembler.py:

Before the fix, each of the 14 substitution-rate/coverage combinations per
encoding scheme drew its OWN independent train/val/test split from the same
underlying 2,000 sequences (stratified by that condition's own failure_freq).
Because the underlying sequences are identical across every condition
sharing an encoding, the same physical sequence could be "seen" during
training in one config and "held out" as unseen test data in another --
quietly invalidating any experiment (e.g. transfer_radius,
distribution_shift) that claims to test generalization to unseen sequences.
Up to 72% train/test overlap was measured between config pairs before the fix.

The fix: one canonical split per encoding scheme, built from sequence
identity (row position) and a fixed seed, shared verbatim across every
substitution-rate/coverage condition under that encoding. These tests
verify that invariant holds on the actual on-disk data, not just in the
split-construction code path.

Requires data/datasets/*.parquet and data/splits/*.parquet to exist (run
`python src/dataset_assembler.py` first) -- skipped otherwise.
"""
import os

import numpy as np
import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASETS_DIR = os.path.join(ROOT, 'data', 'datasets')
SPLITS_DIR   = os.path.join(ROOT, 'data', 'splits')


def _all_config_keys(cfg):
    sub_rates = cfg['channel']['substitution_rates']
    coverages = cfg['coverage_depths']
    encodings = cfg['sequence']['encoding_schemes']
    return [
        f'sub{int(s * 100):02d}_k{k}_{e}'
        for s in sub_rates for k in coverages for e in encodings
    ]


def _require_data(cfg):
    keys = _all_config_keys(cfg)
    missing = [
        k for k in keys
        if not os.path.exists(os.path.join(DATASETS_DIR, f'{k}.parquet'))
        or not os.path.exists(os.path.join(SPLITS_DIR, f'{k}_splits.parquet'))
    ]
    if missing:
        pytest.skip(f"{len(missing)} dataset/split files missing (run dataset_assembler.py first): "
                    f"{missing[:3]}...")
    return keys


def _load_splits(key):
    return pd.read_parquet(os.path.join(SPLITS_DIR, f'{key}_splits.parquet'))


def _load_sequences(key):
    df = pd.read_parquet(os.path.join(DATASETS_DIR, f'{key}.parquet'))
    return df['dna_sequence'].values


def test_split_is_internally_disjoint_and_complete(cfg):
    """For every config: train/val/test partition every row exactly once."""
    keys = _require_data(cfg)
    for key in keys:
        splits = _load_splits(key)
        n = len(splits)
        train_idx = set(splits.loc[splits['split'] == 'train', 'index'])
        val_idx   = set(splits.loc[splits['split'] == 'val',   'index'])
        test_idx  = set(splits.loc[splits['split'] == 'test',  'index'])

        assert train_idx & val_idx == set(),  f"{key}: train/val overlap"
        assert train_idx & test_idx == set(), f"{key}: train/test overlap"
        assert val_idx & test_idx == set(),   f"{key}: val/test overlap"
        assert train_idx | val_idx | test_idx == set(range(n)), (
            f"{key}: split does not cover every row exactly once"
        )


def test_split_membership_identical_across_configs_sharing_an_encoding(cfg):
    """The core regression test: every config sharing an encoding scheme must
    assign the SAME row index to the SAME split (train/val/test) -- this is
    exactly the invariant the pre-fix code violated."""
    keys = _require_data(cfg)
    encodings = cfg['sequence']['encoding_schemes']

    for encoding in encodings:
        enc_keys = [k for k in keys if k.endswith(f'_{encoding}')]
        assert len(enc_keys) >= 2, f"expected multiple configs for encoding={encoding}"

        reference = _load_splits(enc_keys[0]).set_index('index')['split']
        for key in enc_keys[1:]:
            other = _load_splits(key).set_index('index')['split']
            mismatches = (reference != other.reindex(reference.index))
            assert not mismatches.any(), (
                f"{key} disagrees with {enc_keys[0]} on split membership for "
                f"{int(mismatches.sum())} sequences (encoding={encoding}) -- "
                f"this is the exact cross-config leakage bug that was fixed."
            )


def test_sequence_identity_stable_across_configs_sharing_an_encoding(cfg):
    """Row position is used as the sequence's stable identity for split
    purposes -- verify the underlying sequence content at each row index is
    actually identical across every config sharing an encoding, otherwise
    'same index = same sequence' (and therefore the split-sharing fix itself)
    would be meaningless."""
    keys = _require_data(cfg)
    encodings = cfg['sequence']['encoding_schemes']

    for encoding in encodings:
        enc_keys = [k for k in keys if k.endswith(f'_{encoding}')]
        reference_seqs = _load_sequences(enc_keys[0])
        for key in enc_keys[1:]:
            seqs = _load_sequences(key)
            assert len(seqs) == len(reference_seqs), (
                f"{key} has a different sequence count than {enc_keys[0]}"
            )
            assert np.array_equal(seqs, reference_seqs), (
                f"{key}'s sequences at fixed row positions differ from "
                f"{enc_keys[0]}'s -- row position is not a stable sequence "
                f"identity for encoding={encoding}, invalidating the shared-split fix."
            )


def test_no_config_has_an_empty_split(cfg):
    """Sanity guard: a genuinely leak-free split must still produce non-empty
    train/val/test partitions for every config (an empty val or test set
    would silently break downstream evaluation without raising an error)."""
    keys = _require_data(cfg)
    for key in keys:
        splits = _load_splits(key)
        counts = splits['split'].value_counts()
        for part in ('train', 'val', 'test'):
            assert counts.get(part, 0) > 0, f"{key}: empty '{part}' split"
