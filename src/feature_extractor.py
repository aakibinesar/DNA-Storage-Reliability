"""
feature_extractor.py
====================
Extracts a 51-dimensional feature vector for each DNA sequence.

Feature groups (inspired by REFORM — Reliable Per-Base Error Prediction):
  1. Global composition    — GC ratio, AT ratio, sequence length, molecular weight
                             proxy, dinucleotide bias, Shannon entropy
  2. Windowed composition  — GC content in sliding windows (mean, std, min, max)
  3. Homopolymer statistics— max run length, mean run length, centroid position,
                             homopolymer fraction per base type
  4. k-mer frequencies     — all 16 dinucleotide and 64 trinucleotide counts
                             (normalised), selected 9 via redundancy-penalised selection
  5. Structural proxies    — palindrome count, hairpin propensity score,
                             free-energy approximation, repeat fraction
  6. Positional/constraint — distance to start, distance to end (centroid-based),
                             GC-constraint violation score, run-length encoding stats

The full pool has ~100 features; the REFORM paper selects k=9 for the lightweight
framework.  Here we expose the full pool plus a ``select_features`` helper that
replicates the SHAP-based / PR-AUC-drop selection described in the paper.

Usage:
    from feature_extractor import extract_features, FEATURE_NAMES
    feat_matrix = extract_features(list_of_dna_strings)
"""

import math
from collections import Counter
from itertools import product
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


# -- Constants ----------------------------------------------------------------

BASES     = 'ACGT'
COMPLEMENT = {'A': 'T', 'T': 'A', 'G': 'C', 'C': 'G'}

# All 16 dinucleotides, all 64 trinucleotides
DINUCS  = [''.join(p) for p in product(BASES, repeat=2)]
TRINUCS = [''.join(p) for p in product(BASES, repeat=3)]


# -- Top-level API ------------------------------------------------------------

def extract_features(
    sequences: List[str],
    gc_windows: Tuple[int, ...] = (10, 20, 40),
    entropy_window: int = 10,
) -> Tuple[np.ndarray, List[str]]:
    """Extract feature matrix from a list of DNA sequences.

    Parameters
    ----------
    sequences    : list of DNA strings (all ACGT)
    gc_windows   : window sizes for windowed GC computation
    entropy_window : window for Shannon entropy

    Returns
    -------
    X     : float64 array of shape (N, n_features)
    names : list of feature name strings (length n_features)
    """
    if not sequences:
        raise ValueError("sequences is empty")

    rows = []
    for seq in sequences:
        feat, names = _extract_single(seq, gc_windows, entropy_window)
        rows.append(feat)

    return np.array(rows, dtype=np.float64), names


def _extract_single(
    seq: str,
    gc_windows: Tuple[int, ...],
    entropy_window: int,
) -> Tuple[List[float], List[str]]:
    """Extract all features for a single DNA sequence."""
    seq = seq.upper()
    L   = len(seq) if seq else 1

    features: List[float] = []
    names:    List[str]   = []

    def add(value, name):
        features.append(float(value))
        names.append(name)

    # --- Group 1: Global composition -----------------------------------------
    gc   = sum(1 for b in seq if b in 'GC') / L
    at   = sum(1 for b in seq if b in 'AT') / L
    frac = {b: seq.count(b) / L for b in BASES}

    add(gc,   'gc_global')
    add(at,   'at_global')
    add(L,    'seq_length')
    add(frac['A'], 'freq_A')
    add(frac['C'], 'freq_C')
    add(frac['G'], 'freq_G')
    add(frac['T'], 'freq_T')

    # Shannon entropy over full sequence
    add(_shannon_entropy([frac[b] for b in BASES]), 'shannon_entropy_global')

    # Dinucleotide bias: GC-skew (G-C)/(G+C), AT-skew (A-T)/(A+T)
    g, c = seq.count('G'), seq.count('C')
    a, t = seq.count('A'), seq.count('T')
    add((g - c) / max(g + c, 1), 'gc_skew')
    add((a - t) / max(a + t, 1), 'at_skew')

    # --- Group 2: Windowed GC content ----------------------------------------
    for w in gc_windows:
        gc_win = _windowed_gc(seq, w)
        add(np.mean(gc_win),  f'gc_win{w}_mean')
        add(np.std(gc_win),   f'gc_win{w}_std')
        add(np.min(gc_win),   f'gc_win{w}_min')
        add(np.max(gc_win),   f'gc_win{w}_max')

    # Windowed Shannon entropy
    ent_win = _windowed_entropy(seq, entropy_window)
    add(np.mean(ent_win), 'entropy_windowed_mean')
    add(np.std(ent_win),  'entropy_windowed_std')

    # --- Group 3: Homopolymer statistics -------------------------------------
    hp_runs = _homopolymer_runs(seq)
    all_runs = [r for runs in hp_runs.values() for r in runs]
    max_hp   = max(all_runs, key=lambda r: r[1])[1] if all_runs else 0
    mean_hp  = np.mean([r[1] for r in all_runs]) if all_runs else 1.0
    hp_frac  = sum(r[1] for r in all_runs if r[1] > 1) / L

    add(max_hp,   'homopolymer_max_run')
    add(mean_hp,  'homopolymer_mean_run')
    add(hp_frac,  'homopolymer_fraction')

    # Per-base max homopolymer
    for b in BASES:
        runs = hp_runs.get(b, [])
        add(max((r[1] for r in runs), default=0), f'hp_max_{b}')

    # Centroid position of longest homopolymer run
    if all_runs:
        longest = max(all_runs, key=lambda r: r[1])
        centroid = (longest[0] + longest[1] / 2) / L   # normalised 0–1
    else:
        centroid = 0.5
    add(centroid, 'homopolymer_centroid_pos')

    # --- Group 4: k-mer frequencies ------------------------------------------
    di_counts = Counter(seq[i:i+2] for i in range(L - 1))
    for dn in DINUCS:
        add(di_counts.get(dn, 0) / max(L - 1, 1), f'dinuc_{dn}')

    # Selected trinucleotides (first 16 to keep feature count manageable)
    tri_counts = Counter(seq[i:i+3] for i in range(L - 2))
    for tn in TRINUCS[:16]:
        add(tri_counts.get(tn, 0) / max(L - 2, 1), f'trinuc_{tn}')

    # --- Group 5: Structural proxies -----------------------------------------
    add(_palindrome_count(seq, min_len=4),     'palindrome_count')
    add(_hairpin_score(seq),                   'hairpin_score')
    add(_approximate_free_energy(seq),         'approx_free_energy')
    add(_repeat_fraction(seq, repeat_len=4),   'repeat_fraction')
    add(_inverted_repeat_count(seq, min_len=4),'inverted_repeat_count')

    # --- Group 6: Positional & constraint ------------------------------------
    add(0.0 / L, 'dist_to_start')   # 0 for sequence-level features (not per-base)
    add(1.0,     'dist_to_end')     # 1.0 = full sequence

    # Constraint violation score: GC out of [0.40, 0.60]
    gc_violation = max(0.0, 0.40 - gc) + max(0.0, gc - 0.60)
    add(gc_violation, 'gc_constraint_violation')

    # Run-length encoding statistics
    rle_runs = _run_length_encoding(seq)
    add(len(rle_runs),                                         'rle_n_runs')
    add(np.mean([r[1] for r in rle_runs]) if rle_runs else 0, 'rle_mean_run_len')
    add(np.std([r[1] for r in rle_runs]) if rle_runs else 0,  'rle_std_run_len')

    # GC content of first/last quartile (positional GC variation)
    q = max(1, L // 4)
    add(_gc(seq[:q]),      'gc_first_quarter')
    add(_gc(seq[-q:]),     'gc_last_quarter')
    add(_gc(seq[q:3*q]),   'gc_middle_half')

    return features, names


# -- Feature selection helper (REFORM-style) ----------------------------------

def select_features(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: List[str],
    k: int = 9,
    beta: float = 0.1,
    random_state: int = 42,
) -> Tuple[np.ndarray, List[str], List[int]]:
    """Select the top *k* features using SHAP importance + PR-AUC drop analysis
    with redundancy penalisation (as described in REFORM Section 3.1).

    Parameters
    ----------
    X            : feature matrix (N, n_features)
    y            : binary labels (N,)
    feature_names: list of feature name strings
    k            : target number of features to select
    beta         : correlation penalty coefficient
    random_state : random seed

    Returns
    -------
    X_selected    : (N, k) selected feature matrix
    selected_names: list of k selected feature names
    selected_idx  : list of k feature column indices
    """
    try:
        import shap                         # type: ignore
        from sklearn.linear_model import LogisticRegression  # type: ignore
        from sklearn.metrics import average_precision_score  # type: ignore
        from sklearn.preprocessing import StandardScaler     # type: ignore
    except ImportError:
        raise ImportError("shap and scikit-learn required for feature selection")

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    # Train a lightweight logistic probe
    probe = LogisticRegression(max_iter=500, random_state=random_state)
    probe.fit(Xs, (y >= 0.5).astype(int))

    # SHAP importance
    explainer = shap.LinearExplainer(probe, Xs, feature_perturbation='correlation_dependent')
    shap_vals = explainer.shap_values(Xs)  # shape (N, n_features)
    shap_importance = np.abs(shap_vals).mean(axis=0)

    # PR-AUC drop importance
    y_bin  = (y >= 0.5).astype(int)
    y_pred = probe.predict_proba(Xs)[:, 1]
    base_prauc = average_precision_score(y_bin, y_pred)

    pr_drop = np.zeros(X.shape[1])
    for j in range(X.shape[1]):
        Xp         = Xs.copy()
        Xp[:, j]   = np.random.RandomState(random_state).permutation(Xp[:, j])
        yp         = probe.predict_proba(Xp)[:, 1]
        pr_drop[j] = base_prauc - average_precision_score(y_bin, yp)

    # Normalise and aggregate rankings
    norm = lambda v: (v - v.min()) / (v.max() - v.min() + 1e-12)
    score = norm(shap_importance) + norm(np.maximum(pr_drop, 0))
    score = -score  # we'll minimise (lower score = more important)

    # Correlation matrix for redundancy penalty
    C = np.corrcoef(X.T)

    # Greedy selection minimising score + beta * correlation penalty
    selected = []
    remaining = list(range(X.shape[1]))

    for _ in range(k):
        best_j, best_val = None, np.inf
        for j in remaining:
            corr_penalty = sum(abs(C[j, s]) for s in selected)
            val = score[j] + beta * corr_penalty
            if val < best_val:
                best_val, best_j = val, j
        selected.append(best_j)
        remaining.remove(best_j)

    selected_names = [feature_names[i] for i in selected]
    return X[:, selected], selected_names, selected


# -- Internal computation helpers ---------------------------------------------

def _gc(seq: str) -> float:
    if not seq:
        return 0.0
    return sum(1 for b in seq if b in 'GC') / len(seq)


def _windowed_gc(seq: str, window: int) -> np.ndarray:
    L = len(seq)
    if L < window:
        return np.array([_gc(seq)])
    return np.array([_gc(seq[i:i+window]) for i in range(L - window + 1)])


def _windowed_entropy(seq: str, window: int) -> np.ndarray:
    L = len(seq)
    if L < window:
        return np.array([0.0])
    results = []
    for i in range(L - window + 1):
        sub = seq[i:i+window]
        freqs = [sub.count(b) / window for b in BASES]
        results.append(_shannon_entropy(freqs))
    return np.array(results)


def _shannon_entropy(probs: List[float]) -> float:
    h = 0.0
    for p in probs:
        if p > 0:
            h -= p * math.log2(p)
    return h


def _homopolymer_runs(seq: str) -> Dict[str, List[Tuple[int, int]]]:
    """Returns {base: [(start_pos, run_length), ...]} for all runs."""
    result = {b: [] for b in BASES}
    if not seq:
        return result
    start = 0
    for i in range(1, len(seq)):
        if seq[i] != seq[i-1]:
            if seq[i-1] in BASES:
                result[seq[i-1]].append((start, i - start))
            start = i
    if seq[-1] in BASES:
        result[seq[-1]].append((start, len(seq) - start))
    return result


def _run_length_encoding(seq: str) -> List[Tuple[str, int]]:
    if not seq:
        return []
    runs = []
    start = 0
    for i in range(1, len(seq)):
        if seq[i] != seq[i-1]:
            runs.append((seq[i-1], i - start))
            start = i
    runs.append((seq[-1], len(seq) - start))
    return runs


def _palindrome_count(seq: str, min_len: int = 4) -> int:
    """Count palindromic subsequences (reverse complement palindromes)."""
    L = len(seq)
    count = 0
    for length in range(min_len, min(L + 1, 20), 2):
        for i in range(L - length + 1):
            sub = seq[i:i+length]
            rc  = ''.join(COMPLEMENT.get(b, 'N') for b in reversed(sub))
            if sub == rc:
                count += 1
    return count


def _inverted_repeat_count(seq: str, min_len: int = 4) -> int:
    """Count occurrences of inverted repeats (hairpin precursors)."""
    L = len(seq)
    count = 0
    for length in range(min_len, 12):
        for i in range(L - length):
            sub = seq[i:i+length]
            rc  = ''.join(COMPLEMENT.get(b, 'N') for b in reversed(sub))
            if rc in seq[i+length:i+length*3]:
                count += 1
    return count


def _hairpin_score(seq: str, min_stem: int = 4, max_loop: int = 10) -> float:
    """Approximate hairpin propensity: count possible stem-loop formations."""
    L = len(seq)
    score = 0.0
    for stem_len in range(min_stem, min(12, L // 2)):
        for i in range(L - 2 * stem_len - 1):
            stem = seq[i:i+stem_len]
            for loop_len in range(3, max_loop + 1):
                j = i + stem_len + loop_len
                if j + stem_len > L:
                    break
                partner = seq[j:j+stem_len]
                # Reverse complement match
                rc_stem = ''.join(COMPLEMENT.get(b, 'N') for b in reversed(stem))
                match_score = sum(1 for a, b in zip(rc_stem, partner) if a == b)
                if match_score >= stem_len - 1:  # allow 1 mismatch
                    score += match_score / stem_len
    return min(score, 50.0)   # cap to prevent outliers


def _approximate_free_energy(seq: str) -> float:
    """Approximate secondary structure free energy using nearest-neighbour
    stacking parameters (simplified dG in kcal/mol at 37°C, no salt correction).

    Based on known thermodynamic stabilities:
      - GC/GC stacks are most stable (negative dG)
      - AT/AT stacks are least stable
    """
    # Simplified nearest-neighbour energies for DNA (SantaLucia 1998 values, kcal/mol)
    nn_energy = {
        'AA': -1.0, 'AC': -1.44, 'AG': -1.28, 'AT': -0.88,
        'CA': -1.45, 'CC': -1.84, 'CG': -2.17, 'CT': -1.28,
        'GA': -1.30, 'GC': -2.24, 'GG': -1.84, 'GT': -1.44,
        'TA': -0.58, 'TC': -1.30, 'TG': -1.45, 'TT': -1.0,
    }
    energy = 0.0
    for i in range(len(seq) - 1):
        dinuc = seq[i:i+2]
        energy += nn_energy.get(dinuc, -1.0)
    # Normalise by length
    return energy / max(len(seq) - 1, 1)


def _repeat_fraction(seq: str, repeat_len: int = 4) -> float:
    """Fraction of bases covered by repeated k-mers."""
    L = len(seq)
    if L < repeat_len:
        return 0.0
    kmers = Counter(seq[i:i+repeat_len] for i in range(L - repeat_len + 1))
    repeated_bases = sum(
        repeat_len * (count - 1) for count in kmers.values() if count > 1
    )
    return min(repeated_bases / L, 1.0)


# -- Feature names (for export) ------------------------------------------------

def get_feature_names(
    gc_windows: Tuple[int, ...] = (10, 20, 40),
    entropy_window: int = 10,
) -> List[str]:
    """Return feature name list without computing on actual data."""
    _, names = _extract_single('A' * 100, gc_windows, entropy_window)
    return names


# -- Convenience wrapper ------------------------------------------------------

FEATURE_NAMES = get_feature_names()   # module-level constant


if __name__ == '__main__':
    # Quick self-test
    test_seqs = [
        'ACGTACGTACGTACGTACGTACGTACGTACGT' * 3 + 'ACGT',
        'GCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCG',
        'AAAAACCCCCGGGGGTTTTTTAAAAACCCCCGGGGGTTTTTTAAAAACCCCCGGGGGTTTTTTAAAAACCCC',
        'ATATATATATATATATATATATATATATATATATATATATATATATATATATATATATATATATATATATATATATATATATAT',
    ]
    X, names = extract_features(test_seqs[:3])
    print(f"Feature matrix shape: {X.shape}")
    print(f"Feature names ({len(names)}): {names[:10]} ...")
    print(f"GC global: {X[:, names.index('gc_global')]}")
    print(f"Max HP run: {X[:, names.index('homopolymer_max_run')]}")
