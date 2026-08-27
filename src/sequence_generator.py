"""
sequence_generator.py
=====================
Generates DNA sequences from random binary payloads using two encoding schemes:

  1. ``simple``      — direct 2-bit mapping (00->A, 01->C, 10->G, 11->T).
                       Unconstrained: high feature variance, representative of naive
                       DNA-storage pipelines.

  2. ``constrained`` — GC-balanced, homopolymer-limited encoding that approximates
                       the class of constrained codes (e.g. Rinfinity-P8 from Seo et al.).
                       Lower feature variance; tests allocation under realistic
                       sequence design.

Both schemes operate at 2 bits/base (maximum information density).  The constrained
encoder uses a local re-mapping table to enforce GC in [gc_min, gc_max] and
homopolymer run <= max_homopolymer.

Usage (CLI):
    python sequence_generator.py --config configs/experiment_config.yaml \
                                  --n 2000 --scheme simple --out data/sequences/
"""

import os
import random
import hashlib
import argparse
from typing import List, Tuple, Optional

import numpy as np
import yaml


# -- 2-bit encoding alphabet --------------------------------------------------
SIMPLE_TABLE = {0b00: 'A', 0b01: 'C', 0b10: 'G', 0b11: 'T'}
SIMPLE_INV   = {v: k for k, v in SIMPLE_TABLE.items()}

COMPLEMENT = {'A': 'T', 'T': 'A', 'G': 'C', 'C': 'G', 'N': 'N'}

# Constraint re-mapping: for each (current_base, prev_base, prev_prev_base, gc_excess)
# context, provide an ordered list of candidate replacements that would be preferred.
# Simplified: AT <-> GC swaps to balance GC; duplicate-base swaps to break homopolymers.
_AT_BASES = {'A', 'T'}
_GC_BASES = {'G', 'C'}


# -- Public API ---------------------------------------------------------------

def generate_sequences(
    n: int,
    seq_len_bases: int = 100,
    encoding: str = 'simple',
    max_homopolymer: int = 3,
    gc_min: float = 0.40,
    gc_max: float = 0.60,
    seed: int = 42,
) -> List[Tuple[str, bytes]]:
    """Generate *n* DNA sequences and return (dna_sequence, binary_payload) pairs.

    Parameters
    ----------
    n               : number of sequences to generate
    seq_len_bases   : length of each sequence in bases
    encoding        : ``'simple'`` or ``'constrained'``
    max_homopolymer : maximum allowed homopolymer run (constrained mode only)
    gc_min / gc_max : GC content bounds (constrained mode only)
    seed            : random seed for reproducibility

    Returns
    -------
    List of (dna_str, payload_bytes) tuples.
    """
    rng = np.random.default_rng(seed)
    random.seed(seed)

    n_bits = seq_len_bases * 2          # 2 bits per base
    n_bytes = n_bits // 8

    results = []
    for i in range(n):
        payload = bytes(rng.integers(0, 256, size=n_bytes, dtype=np.uint8).tolist())
        if encoding == 'simple':
            dna = _simple_encode(payload, seq_len_bases)
        elif encoding == 'constrained':
            dna = _constrained_encode(
                payload, seq_len_bases, max_homopolymer, gc_min, gc_max, rng
            )
        else:
            raise ValueError(f"Unknown encoding scheme: {encoding!r}")
        results.append((dna, payload))

    return results


def decode_sequence(dna: str, encoding: str = 'simple') -> bytes:
    """Decode a DNA string back to its binary payload (best-effort; decoding
    constrained sequences ignores constraint metadata for simplicity)."""
    bits = []
    for base in dna:
        if base not in SIMPLE_INV:
            bits.extend([0, 0])   # unknown base -> zero bits (conservative)
        else:
            code = SIMPLE_INV[base]
            bits.append((code >> 1) & 1)
            bits.append(code & 1)

    # Pack bits into bytes
    n_bytes = len(bits) // 8
    payload = bytearray()
    for i in range(n_bytes):
        byte_bits = bits[i * 8: i * 8 + 8]
        payload.append(int(''.join(str(b) for b in byte_bits), 2))
    return bytes(payload)


def sequence_checksum(dna: str) -> str:
    """Return a short MD5 hex digest for integrity verification."""
    return hashlib.md5(dna.encode()).hexdigest()[:8]


# -- Internal encoders --------------------------------------------------------

def _simple_encode(payload: bytes, seq_len: int) -> str:
    """Direct 2-bit-per-base encoding.  No constraints enforced."""
    bits = _bytes_to_bits(payload)
    # Pad or truncate to exactly seq_len * 2 bits
    bits = (bits + [0] * (seq_len * 2))[:seq_len * 2]
    return ''.join(SIMPLE_TABLE[bits[i * 2] * 2 + bits[i * 2 + 1]] for i in range(seq_len))


def _constrained_encode(
    payload: bytes,
    seq_len: int,
    max_hp: int,
    gc_min: float,
    gc_max: float,
    rng: np.random.Generator,
) -> str:
    """Constrained encoder: start with simple encoding, then apply local
    substitutions to satisfy homopolymer and GC content constraints.

    Strategy
    --------
    1. Encode payload using simple 2-bit mapping.
    2. Scan left-to-right; whenever a homopolymer run would exceed *max_hp*,
       substitute the offending base with an alternative that:
         a) is complement-different (breaks the run)
         b) is preferred based on local GC balance
    3. After the scan, apply a global GC correction pass if the sequence
       GC content still falls outside [gc_min, gc_max].

    Note: this encoder does NOT maintain invertible lossless decoding—it is
    designed to generate sequences with realistic constrained-code statistics
    for the ML benchmark, not for operational data recovery.
    """
    bases = list(_simple_encode(payload, seq_len))

    # Pass 1: break homopolymer runs
    for i in range(max_hp, seq_len):
        if all(bases[i - j] == bases[i] for j in range(max_hp)):
            gc_count = sum(1 for b in bases[:i] if b in _GC_BASES)
            gc_ratio = gc_count / i if i > 0 else 0.5
            # Prefer a base that doesn't extend the run and adjusts GC balance
            candidates = [b for b in 'ACGT' if b != bases[i]]
            # Sort: if GC is low prefer G/C, else prefer A/T
            if gc_ratio < gc_min:
                candidates.sort(key=lambda b: (b not in _GC_BASES, rng.random()))
            else:
                candidates.sort(key=lambda b: (b in _GC_BASES, rng.random()))
            bases[i] = candidates[0]

    # Pass 2: global GC correction
    gc_count = sum(1 for b in bases if b in _GC_BASES)
    gc_ratio  = gc_count / seq_len
    max_iters = seq_len * 2
    iteration = 0

    while not (gc_min <= gc_ratio <= gc_max) and iteration < max_iters:
        iteration += 1
        if gc_ratio < gc_min:
            # Swap an AT base to a GC base at a position that won't create a new hp run
            candidates = [i for i, b in enumerate(bases) if b in _AT_BASES
                          and not _would_extend_hp(bases, i, _pick_gc(bases, i, rng), max_hp)]
            if not candidates:
                break
            idx = rng.choice(candidates)
            bases[idx] = _pick_gc(bases, idx, rng)
        else:
            # Swap a GC base to an AT base
            candidates = [i for i, b in enumerate(bases) if b in _GC_BASES
                          and not _would_extend_hp(bases, i, _pick_at(bases, i, rng), max_hp)]
            if not candidates:
                break
            idx = rng.choice(candidates)
            bases[idx] = _pick_at(bases, idx, rng)

        gc_count = sum(1 for b in bases if b in _GC_BASES)
        gc_ratio  = gc_count / seq_len

    return ''.join(bases)


def _would_extend_hp(bases: List[str], pos: int, new_base: str, max_hp: int) -> bool:
    """Return True if placing *new_base* at *pos* creates a homopolymer of length > max_hp."""
    test = bases.copy()
    test[pos] = new_base
    # Count run around pos
    run = 1
    i = pos - 1
    while i >= 0 and test[i] == new_base:
        run += 1
        i -= 1
    i = pos + 1
    while i < len(test) and test[i] == new_base:
        run += 1
        i += 1
    return run > max_hp


def _pick_gc(bases: List[str], pos: int, rng: np.random.Generator) -> str:
    """Pick a GC base that differs from the current base at pos."""
    options = [b for b in 'GC' if b != bases[pos]]
    return rng.choice(options) if options else 'G'


def _pick_at(bases: List[str], pos: int, rng: np.random.Generator) -> str:
    """Pick an AT base that differs from the current base at pos."""
    options = [b for b in 'AT' if b != bases[pos]]
    return rng.choice(options) if options else 'A'


def _bytes_to_bits(data: bytes) -> List[int]:
    """Convert bytes to a flat list of bits (MSB first per byte)."""
    bits = []
    for byte in data:
        for shift in range(7, -1, -1):
            bits.append((byte >> shift) & 1)
    return bits


# -- Feature statistics helpers (used by diagnostics) -------------------------

def gc_content(dna: str) -> float:
    if not dna:
        return 0.0
    return sum(1 for b in dna if b in 'GC') / len(dna)


def max_homopolymer_run(dna: str) -> int:
    if not dna:
        return 0
    max_run = current_run = 1
    for i in range(1, len(dna)):
        if dna[i] == dna[i - 1]:
            current_run += 1
            max_run = max(max_run, current_run)
        else:
            current_run = 1
    return max_run


# -- CLI entry point ----------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Generate DNA sequences for the benchmark.')
    parser.add_argument('--config', default='configs/experiment_config.yaml')
    parser.add_argument('--n', type=int, default=None)
    parser.add_argument('--scheme', choices=['simple', 'constrained'], default='simple')
    parser.add_argument('--out', default='data/sequences/')
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    n          = args.n or cfg['sequence']['n_sequences']
    seq_len    = cfg['sequence']['seq_len_bases']
    max_hp     = cfg['sequence']['max_homopolymer']
    gc_min_val = cfg['sequence']['gc_min']
    gc_max_val = cfg['sequence']['gc_max']
    seed       = cfg['random_seed']

    print(f"[sequence_generator] Generating {n} sequences (scheme={args.scheme}, "
          f"len={seq_len}, seed={seed})")

    seqs = generate_sequences(
        n, seq_len, args.scheme, max_hp, gc_min_val, gc_max_val, seed
    )

    os.makedirs(args.out, exist_ok=True)
    out_path = os.path.join(args.out, f'seqs_{args.scheme}_n{n}.txt')
    with open(out_path, 'w') as fout:
        fout.write('# dna_sequence\tgc_content\tmax_hp_run\n')
        for dna, _ in seqs:
            fout.write(f'{dna}\t{gc_content(dna):.4f}\t{max_homopolymer_run(dna)}\n')

    print(f"[sequence_generator] Saved to {out_path}")

    # Quick diagnostic stats
    gc_vals  = [gc_content(dna) for dna, _ in seqs]
    hp_vals  = [max_homopolymer_run(dna) for dna, _ in seqs]
    print(f"  GC content  : mean={np.mean(gc_vals):.3f}, std={np.std(gc_vals):.3f}, "
          f"min={min(gc_vals):.3f}, max={max(gc_vals):.3f}")
    print(f"  Max HP run  : mean={np.mean(hp_vals):.2f}, max={max(hp_vals)}")


if __name__ == '__main__':
    main()
