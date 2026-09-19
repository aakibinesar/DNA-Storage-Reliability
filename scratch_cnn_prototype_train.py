"""
Phase 0 feasibility test: train a small 1D CNN on raw DNA sequence to
predict failure_freq, on the two target near-chance configs. Validates the
full loop (data -> tensors -> GPU -> training -> eval) before committing to
the full n=10000 run. Not a real result -- n=300, no HP tuning, no held-out
comparison against XGBoost yet.

Usage:
    python scratch_cnn_prototype_train.py
"""
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

BASE_TO_IDX = {'A': 0, 'C': 1, 'G': 2, 'T': 3}


def one_hot_encode(sequences, seq_len=100):
    """(N, 4, L) one-hot tensor -- channels-first for Conv1d."""
    arr = np.zeros((len(sequences), 4, seq_len), dtype=np.float32)
    for i, seq in enumerate(sequences):
        for j, base in enumerate(seq[:seq_len]):
            idx = BASE_TO_IDX.get(base)
            if idx is not None:
                arr[i, idx, j] = 1.0
    return arr


class SmallCNN(nn.Module):
    """Small 1D CNN sized to comfortably fit a 2GB GPU."""

    def __init__(self, seq_len=100):
        super().__init__()
        self.conv1 = nn.Conv1d(4, 32, kernel_size=9, padding=4)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, padding=2)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(64, 32)
        self.fc2 = nn.Linear(32, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.pool(x).squeeze(-1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.sigmoid(self.fc2(x))
        return x.squeeze(-1)


def run(key, device):
    df = pd.read_parquet(f'scratch_cnn_prototype_data/{key}.parquet')
    seqs = df['dna_sequence'].tolist()
    y = df['failure_freq'].values.astype(np.float32)

    n = len(seqs)
    rng = np.random.default_rng(42)
    perm = rng.permutation(n)
    n_train = int(n * 0.7)
    n_val = int(n * 0.15)
    train_idx = perm[:n_train]
    val_idx = perm[n_train:n_train + n_val]
    test_idx = perm[n_train + n_val:]

    X = one_hot_encode(seqs)
    X_t = torch.from_numpy(X)
    y_t = torch.from_numpy(y)

    train_ds = TensorDataset(X_t[train_idx], y_t[train_idx])
    val_ds = TensorDataset(X_t[val_idx], y_t[val_idx])
    test_ds = TensorDataset(X_t[test_idx], y_t[test_idx])

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=64)
    test_loader = DataLoader(test_ds, batch_size=64)

    model = SmallCNN().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    print(f"[{key}] n_train={len(train_idx)} n_val={len(val_idx)} n_test={len(test_idx)} "
          f"gpu_mem_allocated={torch.cuda.memory_allocated(device)/1e6:.1f}MB")

    t0 = time.time()
    for epoch in range(30):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()

        if (epoch + 1) % 10 == 0:
            model.eval()
            with torch.no_grad():
                val_losses = []
                for xb, yb in val_loader:
                    xb, yb = xb.to(device), yb.to(device)
                    val_losses.append(loss_fn(model(xb), yb).item())
            print(f"  epoch {epoch+1}: val_mse={np.mean(val_losses):.4f}")

    elapsed = time.time() - t0
    peak_mem = torch.cuda.max_memory_allocated(device) / 1e6
    print(f"[{key}] training done in {elapsed:.1f}s, peak GPU mem={peak_mem:.1f}MB")

    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(device)
            preds.append(model(xb).cpu().numpy())
            trues.append(yb.numpy())
    preds = np.concatenate(preds)
    trues = np.concatenate(trues)

    from sklearn.metrics import roc_auc_score
    y_bin = (trues >= 0.5).astype(int)
    if len(np.unique(y_bin)) == 2:
        auroc = roc_auc_score(y_bin, preds)
        print(f"[{key}] test AUROC (n={len(trues)}, tiny/no-HP-tuning prototype): {auroc:.4f}")
    else:
        print(f"[{key}] test set is single-class at n={len(trues)} -- AUROC undefined (expected at this tiny scale)")


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    torch.cuda.reset_peak_memory_stats(device) if device.type == 'cuda' else None
    for key in ['sub15_k3_constrained', 'sub15_k3_simple']:
        run(key, device)
