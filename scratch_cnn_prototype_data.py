"""
Phase 0 feasibility data: small (n=300) labeled datasets for the two target
configs (sub15_k3_constrained, sub15_k3_simple -- the near-chance-AUROC
configs the CNN is specifically meant to test), generated fast now that the
PCR fix is in. Not for real results -- just to validate the CNN training
loop end-to-end (data loading, architecture, GPU memory) before committing
to the full n=10000 run.

Usage:
    python scratch_cnn_prototype_data.py
"""
import os
import sys
import time

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, 'src')
from sequence_generator import generate_sequences
from channel_model import build_channel_from_config
from label_generator import compute_failure_labels

OUT_DIR = 'scratch_cnn_prototype_data'
N = 300
TARGETS = [
    ('sub15_k3_constrained', 0.15, 3, 'constrained'),
    ('sub15_k3_simple',      0.15, 3, 'simple'),
]

with open('configs/experiment_config.yaml') as f:
    cfg = yaml.safe_load(f)

seed = cfg['random_seed']
seq_cfg = cfg['sequence']
n_runs = cfg['allocation']['n_monte_carlo_runs']
l_rs = seq_cfg['l_rs_default']

os.makedirs(OUT_DIR, exist_ok=True)

for key, sub_rate, coverage, encoding in TARGETS:
    t0 = time.time()
    print(f"[{key}] generating {N} sequences (encoding={encoding}) ...")
    seqs_with_payload = generate_sequences(
        N, seq_cfg['seq_len_bases'], encoding,
        max_homopolymer=seq_cfg['max_homopolymer'],
        gc_min=seq_cfg['gc_min'], gc_max=seq_cfg['gc_max'], seed=seed,
    )
    sequences = [s for s, _ in seqs_with_payload]

    print(f"[{key}] running channel simulation (n_runs={n_runs}) ...")
    channel = build_channel_from_config(cfg, sub_rate, seed=seed)
    failure_freq, byte_errors_mean = compute_failure_labels(
        sequences, channel, coverage, l_rs, n_runs=n_runs, base_seed=seed, verbose=False,
    )

    df = pd.DataFrame({
        'dna_sequence': sequences,
        'failure_freq': failure_freq,
        'byte_errors_mean': byte_errors_mean,
        'label_binary': (failure_freq >= 0.5).astype(int),
    })
    out_path = os.path.join(OUT_DIR, f'{key}.parquet')
    df.to_parquet(out_path, index=False)
    elapsed = time.time() - t0
    print(f"[{key}] saved {len(df)} rows -> {out_path}  "
          f"(failure_rate={df['label_binary'].mean():.3f}, elapsed={elapsed:.1f}s)")

print("Done.")
