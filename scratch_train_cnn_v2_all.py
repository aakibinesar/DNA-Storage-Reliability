"""
Sequential launcher for models/train_cnn_v2.py (BigCNN) across all 28 configs.

Sequential, not parallel: same reasoning as scratch_train_cnn_all.py --
one GPU, so multiple processes would contend rather than speed anything up.

Usage:
    python scratch_train_cnn_v2_all.py [--force]
"""
import argparse
import os
import time

import yaml


def get_all_keys(cfg):
    sub_rates = cfg['channel']['substitution_rates']
    coverages = cfg['coverage_depths']
    encodings = cfg['sequence']['encoding_schemes']
    return [
        f'sub{int(s * 100):02d}_k{k}_{e}'
        for s in sub_rates for k in coverages for e in encodings
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--force', action='store_true')
    parser.add_argument('--config', default='configs/experiment_config.yaml')
    parser.add_argument('--out', default='models/saved_cnn_v2/')
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    all_keys = get_all_keys(cfg)

    def already_done(key):
        return not args.force and os.path.exists(os.path.join(args.out, f'{key}_cnn_v2.pkl'))

    keys = [k for k in all_keys if not already_done(k)]
    print(f"[train_cnn_v2] {len(all_keys) - len(keys)} already done, {len(keys)} remaining")

    import sys
    sys.path.insert(0, 'models')
    sys.path.insert(0, 'src')
    from train_cnn_v2 import train_cnn_v2_model

    t_start = time.time()
    for i, key in enumerate(keys):
        t0 = time.time()
        try:
            train_cnn_v2_model(key, cfg, cfg['random_seed'], args.out, verbose=False)
            print(f"  [{i+1}/{len(keys)}] {key}: OK  [{time.time()-t0:.1f}s]")
        except Exception as e:
            print(f"  [{i+1}/{len(keys)}] {key}: FAIL - {e}")
    print(f"[train_cnn_v2] Done in {(time.time() - t_start) / 60:.1f} min")


if __name__ == '__main__':
    main()
