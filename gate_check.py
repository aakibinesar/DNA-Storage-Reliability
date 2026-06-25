"""
gate_check.py
=============
Thin CLI wrapper around evaluate.go_no_go_check() (already implemented in
models/evaluate.py). Loads the trained+calibrated XGBoost model and test
split for one dataset configuration and reports the Week 4 go/no-go gate:

    PASS  iff  ECE < 0.05  AND  Brier < no-skill baseline

Usage:
    python gate_check.py --config configs/experiment_config.yaml \
        --key sub12_k5_constrained --models-dir models/saved/

Exit code 0 = PASS, 1 = FAIL, 2 = error (e.g. models not trained yet).
"""

import argparse
import os
import sys

import yaml

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, 'src'))
sys.path.insert(0, os.path.join(ROOT, 'models'))


def main():
    parser = argparse.ArgumentParser(description='Week 4 go/no-go gate check.')
    parser.add_argument('--config', default='configs/experiment_config.yaml')
    parser.add_argument('--key', default=None,
                         help='Dataset config key. Defaults to the highest '
                              'substitution rate + primary coverage + constrained '
                              'encoding (the R∞-P8 12%% condition from the plan).')
    parser.add_argument('--models-dir', default='models/saved/')
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    from dataset_assembler import load_dataset, get_all_config_keys
    from train import load_models
    from evaluate import go_no_go_check

    if args.key is None:
        sub_max  = max(cfg['channel']['substitution_rates'])
        cov_prim = cfg['coverage_depths'][0]
        args.key = f'sub{int(sub_max*100):02d}_k{cov_prim}_constrained'
        if args.key not in get_all_config_keys(cfg):
            print(f"[gate_check] ERROR: derived key '{args.key}' is not one of the "
                  f"16 configured datasets. Pass --key explicitly.")
            sys.exit(2)

    print(f"[gate_check] Gate dataset: {args.key}")

    try:
        X_tr, X_val, X_te, y_tr, y_val, y_te, feat_names = load_dataset(args.key, cfg)
        models = load_models(args.models_dir, args.key)
    except Exception as e:
        print(f"[gate_check] ERROR loading dataset/models for {args.key}: {e}")
        print("  Has the 'datasets' and 'train' stages completed for this key yet?")
        sys.exit(2)

    xgb_cal = models.get('xgboost')
    if xgb_cal is None:
        print(f"[gate_check] ERROR: no calibrated xgboost model found for {args.key}")
        sys.exit(2)

    y_prob = xgb_cal.predict_proba(X_te)
    y_bin  = (y_te >= 0.5).astype(int)

    passed, msg = go_no_go_check(y_bin, y_prob)

    print(f"\n[gate_check] {args.key}  (n_test={len(y_te)})")
    print(msg)
    print(f"\n[gate_check] RESULT: {'PASS' if passed else 'FAIL'}")

    if not passed:
        print(
            "\n  Diagnosis order per the implementation plan:\n"
            "  1. Class balance — failure rate below 5%% -> use weighted loss.\n"
            "  2. Feature variance — check XGBoost feature_importances_ for dead features.\n"
            "  3. Label quality — inspect high-predicted sequences against raw simulator output."
        )

    sys.exit(0 if passed else 1)


if __name__ == '__main__':
    main()
