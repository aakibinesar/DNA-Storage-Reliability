"""
models/train_cnn_v2.py
=======================
A second, meaningfully different CNN architecture over raw one-hot DNA
sequence, trained with the same methodology as train_cnn.py.

Motivation: models/train_cnn.py's SmallCNN (2 conv layers, plain average
pooling, ~tens of thousands of parameters) underperformed the classical
feature-based models on every usable config (see README "Sequence-Level
Model: 1D CNN vs. Hand-Crafted Features"). That's a real, honestly-reported
negative result -- but a reviewer's reflexive objection to any single
CNN-loses result is "you just didn't try hard enough." This trains a
BigCNN -- deeper (4 conv layers vs 2), wider, with batch norm, dilated
convolutions for a larger effective receptive field (captures longer-range
positional interactions plain stacked small kernels can't), and learned
attention pooling instead of plain averaging (lets the model weight
informative positions instead of treating every position equally) -- to
check whether the negative result survives a meaningfully different,
higher-capacity architecture, not just a re-run of the same one.

Same discipline as train_cnn.py: identical split, small HP grid selected
on validation Brier, early stopping, independent Platt-calibration split.

Usage (CLI):
    python models/train_cnn_v2.py --config configs/experiment_config.yaml \
        --key sub15_k3_simple --out models/saved_cnn_v2/
"""

import argparse
import os
import pickle
import sys
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.dirname(__file__))

from train_cnn import one_hot_encode, CNNSklearnAdapter  # noqa: E402 -- reuse shared plumbing

# Small HP grid -- same discipline as train_cnn.py, sized to this model's
# larger capacity (more filters, an explicit dropout sweep since a deeper
# net is more prone to overfitting the same small train sets).
_HP_GRID = [
    {'n_filters': 64,  'lr': 1e-3, 'weight_decay': 1e-4, 'dropout': 0.3},
    {'n_filters': 64,  'lr': 3e-4, 'weight_decay': 1e-4, 'dropout': 0.4},
    {'n_filters': 128, 'lr': 1e-3, 'weight_decay': 1e-4, 'dropout': 0.4},
]
_MAX_EPOCHS = 100
_PATIENCE = 10


class BigCNN(nn.Module):
    """Deeper, higher-capacity CNN with dilated convolutions and attention
    pooling. Module-level (not local) so instances can be pickled -- see
    SmallCNN's docstring in train_cnn.py for why that matters."""

    def __init__(self, n_filters: int = 64, dropout: float = 0.3):
        super().__init__()
        self.conv1 = nn.Conv1d(4, n_filters, kernel_size=7, padding=3)
        self.bn1 = nn.BatchNorm1d(n_filters)
        self.conv2 = nn.Conv1d(n_filters, n_filters, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(n_filters)
        # Dilated convolutions widen the effective receptive field without
        # adding more parameters than a plain deeper stack would -- lets
        # the model see longer-range positional interactions than
        # SmallCNN's two undilated layers (effective receptive field ~13bp)
        # can, up to roughly the full 100bp sequence length here.
        self.conv3 = nn.Conv1d(n_filters, n_filters * 2, kernel_size=5, padding=4, dilation=2)
        self.bn3 = nn.BatchNorm1d(n_filters * 2)
        self.conv4 = nn.Conv1d(n_filters * 2, n_filters * 2, kernel_size=3, padding=4, dilation=4)
        self.bn4 = nn.BatchNorm1d(n_filters * 2)
        # Attention pooling: a learned per-position importance score,
        # softmax-normalized and used as pooling weights -- instead of
        # SmallCNN's plain average over all positions, this lets the model
        # learn which positions matter rather than treating all equally.
        self.attn = nn.Conv1d(n_filters * 2, 1, kernel_size=1)
        self.fc1 = nn.Linear(n_filters * 2, n_filters)
        self.fc2 = nn.Linear(n_filters, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.relu(self.bn4(self.conv4(x)))          # (B, C, L)
        attn_weights = torch.softmax(self.attn(x), dim=-1)  # (B, 1, L)
        x = (x * attn_weights).sum(dim=-1)                # (B, C)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x).squeeze(-1)   # raw logit; sigmoid applied at inference


def _build_model(hp: dict):
    return BigCNN(n_filters=hp['n_filters'], dropout=hp['dropout'])


def _train_one(hp, X_tr, y_tr, X_tune, y_tune, device, seed):
    from torch.utils.data import DataLoader, TensorDataset

    torch.manual_seed(seed)
    model = _build_model(hp).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=hp['lr'], weight_decay=hp['weight_decay'])
    loss_fn = nn.BCEWithLogitsLoss()

    train_ds = TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(y_tr))
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True,
                               generator=torch.Generator().manual_seed(seed))

    X_tune_t = torch.from_numpy(X_tune).to(device)
    y_tune_np = y_tune

    best_val_brier = np.inf
    best_state = None
    epochs_no_improve = 0

    for epoch in range(_MAX_EPOCHS):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            val_proba = torch.sigmoid(model(X_tune_t)).cpu().numpy()
        val_brier = float(np.mean((val_proba - y_tune_np) ** 2))

        if val_brier < best_val_brier - 1e-5:
            best_val_brier = val_brier
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= _PATIENCE:
                break

    model.load_state_dict(best_state)
    return model, best_val_brier


def train_cnn_v2_model(key: str, cfg: dict, seed: int, out_dir: str, verbose: bool = True):
    from dataset_assembler import load_dataset_sequences
    from calibrate import CalibratedModel

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if verbose:
        print(f"[train_cnn_v2] {key}: device={device}")

    seq_tr, seq_val, seq_te, y_tr, y_val, y_te = load_dataset_sequences(key, cfg)
    seq_len = cfg['sequence']['seq_len_bases']

    X_tr = one_hot_encode(seq_tr, seq_len)
    X_val = one_hot_encode(seq_val, seq_len)
    X_te = one_hot_encode(seq_te, seq_len)

    y_tr = y_tr.astype(np.float32)
    y_val = y_val.astype(np.float32)

    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(X_val))
    mid = len(perm) // 2
    tune_idx, cal_idx = perm[:mid], perm[mid:]
    X_tune, y_tune = X_val[tune_idx], y_val[tune_idx]
    X_cal, y_cal = X_val[cal_idx], y_val[cal_idx]

    if verbose:
        print(f"  train={len(y_tr)}, tune={len(y_tune)}, cal={len(y_cal)}, test={len(y_te)}")

    best_model, best_brier, best_hp = None, np.inf, None
    for hp in _HP_GRID:
        model, val_brier = _train_one(hp, X_tr, y_tr, X_tune, y_tune, device, seed)
        if verbose:
            print(f"  hp={hp} -> tune_brier={val_brier:.4f}")
        if val_brier < best_brier:
            best_brier, best_model, best_hp = val_brier, model, hp

    if verbose:
        print(f"  best hp={best_hp} (tune_brier={best_brier:.4f})")

    adapter = CNNSklearnAdapter(best_model, device)
    calibrated = CalibratedModel(adapter, method='platt', seed=seed)
    calibrated.fit(X_cal, y_cal)

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f'{key}_cnn_v2.pkl')
    with open(out_path, 'wb') as f:
        pickle.dump(calibrated, f)
    if verbose:
        print(f"[train_cnn_v2] Saved -> {out_path}")

    return calibrated, X_te, y_te.astype(np.float32)


def main():
    parser = argparse.ArgumentParser(description='Train the deeper BigCNN variant for one dataset config.')
    parser.add_argument('--config', default='configs/experiment_config.yaml')
    parser.add_argument('--key', required=True)
    parser.add_argument('--out', default='models/saved_cnn_v2/')
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    train_cnn_v2_model(args.key, cfg, cfg['random_seed'], args.out, verbose=True)


if __name__ == '__main__':
    main()
