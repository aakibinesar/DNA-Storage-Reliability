"""
models/train_cnn.py
====================
Trains a small 1D CNN directly on raw one-hot-encoded DNA sequence to
predict failure_freq, as a 4th model type alongside XGBoost / Random
Forest / Logistic Regression in analysis/model_comparison.py.

Motivation (see README "Model Comparison" / "Benefit-Aware Model"): three
different classical model families trained on the same ~76 hand-crafted
features all agree there is no learnable signal in sub15_k3_constrained
and sub15_k3_simple (near-chance AUROC despite well-supported sample
sizes). A model that learns directly from sequence, rather than from those
engineered features, is a genuinely different hypothesis about where the
ceiling is -- feature representation vs. real absence of signal.

To be a fair, meaningful comparison against the classical models (not just
a runnable one), this follows the same discipline train.py uses:
  - identical train/val/test split (by row index) via
    dataset_assembler.load_dataset_sequences -- guaranteed same membership
    as load_dataset uses for the feature-based models.
  - a small hyperparameter search, selected on validation Brier score
    (MSE against continuous failure_freq), matching XGBoost/RF's grid
    search discipline -- just a much smaller grid, appropriate to how much
    smaller this model is.
  - early stopping on a held-out slice, not a fixed epoch count.
  - post-hoc Platt calibration via calibrate.CalibratedModel, fit on an
    INDEPENDENT calibration split (R6 fix: never the same data used for
    HP selection).
  - explicit seeding (numpy AND torch) for reproducibility.

Usage (CLI):
    python models/train_cnn.py --config configs/experiment_config.yaml \
        --key sub15_k3_simple --out models/saved_cnn/
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

BASE_TO_IDX = {'A': 0, 'C': 1, 'G': 2, 'T': 3}

# Small HP grid -- appropriately scoped to a model orders of magnitude
# smaller than the XGBoost 48-combo grid.
_HP_GRID = [
    {'n_filters': 32, 'lr': 1e-3, 'weight_decay': 0.0},
    {'n_filters': 32, 'lr': 3e-4, 'weight_decay': 1e-4},
    {'n_filters': 64, 'lr': 1e-3, 'weight_decay': 1e-4},
]
_MAX_EPOCHS = 100
_PATIENCE = 10


class SmallCNN(nn.Module):
    """Small 1D CNN over one-hot DNA sequence, sized to comfortably fit a
    2GB GPU. Module-level (not a local class) so instances can be pickled --
    a local/nested class breaks pickle.dump silently at save time."""

    def __init__(self, n_filters: int = 32):
        super().__init__()
        self.conv1 = nn.Conv1d(4, n_filters, kernel_size=9, padding=4)
        self.conv2 = nn.Conv1d(n_filters, n_filters * 2, kernel_size=5, padding=2)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(n_filters * 2, n_filters)
        self.fc2 = nn.Linear(n_filters, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.pool(x).squeeze(-1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x).squeeze(-1)   # raw logit; sigmoid applied at inference


def one_hot_encode(sequences: List[str], seq_len: int = 100) -> np.ndarray:
    """(N, 4, L) one-hot array, channels-first for Conv1d."""
    arr = np.zeros((len(sequences), 4, seq_len), dtype=np.float32)
    for i, seq in enumerate(sequences):
        for j, base in enumerate(seq[:seq_len]):
            idx = BASE_TO_IDX.get(base)
            if idx is not None:
                arr[i, idx, j] = 1.0
    return arr


def _build_model(n_filters: int, seq_len: int = 100):
    return SmallCNN(n_filters)


def _train_one(hp, X_tr, y_tr, X_tune, y_tune, device, seed):
    from torch.utils.data import DataLoader, TensorDataset

    torch.manual_seed(seed)
    model = _build_model(hp['n_filters']).to(device)
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


class CNNSklearnAdapter:
    """Minimal sklearn-style wrapper so CalibratedModel (calibrate.py) can
    treat the trained CNN identically to the classical models: exposes
    `predict_proba(X)` -> 1D array of P(failure), where X is the
    (N, 4, L) one-hot-encoded sequence array produced by one_hot_encode().
    """

    def __init__(self, torch_model, device):
        self._model = torch_model
        self._device = device

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        self._model.eval()
        with torch.no_grad():
            X_t = torch.from_numpy(X.astype(np.float32)).to(self._device)
            proba = torch.sigmoid(self._model(X_t)).cpu().numpy()
        return np.clip(proba, 0.0, 1.0)

    def __getstate__(self):
        # Move to CPU for portable pickling; predict_proba re-homes to
        # self._device (still CPU unless the caller mutates it after load).
        state = self.__dict__.copy()
        state['_model'] = self._model.to('cpu')
        state['_device'] = 'cpu'
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)


def train_cnn_model(key: str, cfg: dict, seed: int, out_dir: str, verbose: bool = True):
    from dataset_assembler import load_dataset_sequences
    from calibrate import CalibratedModel

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if verbose:
        print(f"[train_cnn] {key}: device={device}")

    seq_tr, seq_val, seq_te, y_tr, y_val, y_te = load_dataset_sequences(key, cfg)
    seq_len = cfg['sequence']['seq_len_bases']

    X_tr = one_hot_encode(seq_tr, seq_len)
    X_val = one_hot_encode(seq_val, seq_len)
    X_te = one_hot_encode(seq_te, seq_len)

    y_tr = y_tr.astype(np.float32)
    y_val = y_val.astype(np.float32)

    # Independent tune/cal split from val, matching train.py's R6 discipline:
    # HP selection never touches the data later used to fit calibration.
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
    out_path = os.path.join(out_dir, f'{key}_cnn.pkl')
    with open(out_path, 'wb') as f:
        pickle.dump(calibrated, f)
    if verbose:
        print(f"[train_cnn] Saved -> {out_path}")

    return calibrated, X_te, y_te.astype(np.float32)


def main():
    parser = argparse.ArgumentParser(description='Train a small 1D CNN model for one dataset config.')
    parser.add_argument('--config', default='configs/experiment_config.yaml')
    parser.add_argument('--key', required=True)
    parser.add_argument('--out', default='models/saved_cnn/')
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    train_cnn_model(args.key, cfg, cfg['random_seed'], args.out, verbose=True)


if __name__ == '__main__':
    main()
