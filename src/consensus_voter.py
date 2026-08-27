"""
consensus_voter.py
==================
Positional majority-vote consensus reconstruction from a cluster of noisy reads.

Handles insertion/deletion-induced length variation through a simplified
Bitwise Majority Alignment (BMA) approach (Batu et al. 2004 / reliability
skew literature):

  1. Normalise all reads to the expected length by trimming (long reads) or
     padding with gaps (short reads).
  2. Apply positional majority vote, ignoring gap characters.

This mirrors the consensus-finding step in the DNA storage pipeline and is
consistent with the reliability skew observation: errors accumulate towards
the middle of the sequence during alignment.

Usage:
    from consensus_voter import majority_vote_consensus
    consensus = majority_vote_consensus(reads, expected_length=100)
"""

from collections import Counter
from typing import List, Optional

import numpy as np


GAP_CHAR = 'N'   # placeholder for missing positions


def majority_vote_consensus(reads: List[str], expected_len: int) -> str:
    """Reconstruct a consensus sequence by positional majority vote.

    Parameters
    ----------
    reads        : list of noisy reads (strings over ACGT, possibly with
                   insertions and deletions so lengths may vary)
    expected_len : expected length of the original sequence

    Returns
    -------
    Consensus DNA string of length *expected_len*.  Positions where no read
    contributed a base are filled with GAP_CHAR ('N').
    """
    if not reads:
        return GAP_CHAR * expected_len

    # Step 1: normalise read lengths
    normalised = [_normalise_read(r, expected_len) for r in reads]

    # Step 2: positional plurality vote
    consensus = []
    for pos in range(expected_len):
        votes = [r[pos] for r in normalised if r[pos] != GAP_CHAR]
        if not votes:
            consensus.append(GAP_CHAR)
        else:
            # Plurality: most common base; ties broken by ACGT order
            c = Counter(votes)
            best = max(c, key=lambda b: (c[b], -'ACGT'.index(b)))
            consensus.append(best)

    return ''.join(consensus)


def count_errors(original: str, consensus: str) -> dict:
    """Compare original and consensus sequences; return error statistics.

    Returns
    -------
    dict with keys:
      'base_errors'      : number of positions where bases differ (excluding gaps)
      'gap_positions'    : number of consensus positions filled with GAP_CHAR
      'error_positions'  : list of (pos, orig_base, consensus_base) triples
      'byte_errors'      : estimated number of corrupted 8-bit bytes (4 bases = 1 byte)
    """
    if len(original) != len(consensus):
        raise ValueError(
            f"Length mismatch: original={len(original)}, consensus={len(consensus)}"
        )

    base_errors = 0
    gap_positions = 0
    error_positions = []
    corrupted_bytes = set()

    for i, (orig, cons) in enumerate(zip(original, consensus)):
        if cons == GAP_CHAR:
            gap_positions += 1
            # A gap counts as an unknown byte
            corrupted_bytes.add(i // 4)
        elif orig != cons:
            base_errors += 1
            error_positions.append((i, orig, cons))
            corrupted_bytes.add(i // 4)

    return {
        'base_errors'   : base_errors,
        'gap_positions' : gap_positions,
        'error_positions': error_positions,
        'byte_errors'   : len(corrupted_bytes),
    }


def is_decoding_failure(
    n_byte_errors: int,
    l_rs: int,
) -> bool:
    """Return True if the number of byte errors exceeds RS correction capacity.

    RS(n, k) with *l_rs* parity bytes can correct floor(l_rs / 2) byte errors.

    Parameters
    ----------
    n_byte_errors : number of corrupted bytes after consensus
    l_rs          : number of RS parity bytes allocated to this sequence
    """
    correction_capacity = l_rs // 2
    return n_byte_errors > correction_capacity


# -- Internal helpers ---------------------------------------------------------

def _normalise_read(read: str, expected_len: int) -> str:
    """Trim or pad a read to match *expected_len*.

    Long reads  : trim excess symmetrically from both ends.
    Short reads : pad with GAP_CHAR symmetrically at both ends.

    Rationale: ends of DNA oligos are anchored by primers (in real systems),
    so interior insertions/deletions accumulate towards the middle.  Symmetric
    trimming/padding is a simple approximation of proper MSA alignment and
    replicates the reliability-skew-inducing behaviour described by Lin et al.
    """
    n = len(read)
    if n == expected_len:
        return read

    if n > expected_len:
        excess = n - expected_len
        left   = excess // 2
        right  = excess - left
        return read[left: n - right] if right > 0 else read[left:]

    # n < expected_len: pad
    shortage = expected_len - n
    left_pad = shortage // 2
    right_pad = shortage - left_pad
    return GAP_CHAR * left_pad + read + GAP_CHAR * right_pad


def simulate_coverage_statistics(
    originals: List[str],
    channel,
    coverage: int,
    rng_seed: int = 0,
) -> dict:
    """Run consensus over a list of sequences and return aggregate statistics.

    Useful for the go/no-go sanity checks described in the implementation plan:
      - K=30, 1% sub   -> consensus error rate < 0.1%
      - K=10, 12% sub  -> consensus error rate ~= 3–5%

    Parameters
    ----------
    originals : list of original DNA sequences
    channel   : DNAStorageChannel instance
    coverage  : K (number of reads per sequence)
    rng_seed  : seed for reproducibility

    Returns
    -------
    dict with 'mean_base_error_rate', 'mean_byte_errors', 'n_failures' etc.
    """
    from channel_model import DNAStorageChannel

    np.random.seed(rng_seed)
    base_error_rates = []
    byte_errors_list = []

    for orig in originals:
        reads     = channel.simulate(orig, coverage)
        consensus = majority_vote_consensus(reads, len(orig))
        stats     = count_errors(orig, consensus)
        base_error_rates.append(stats['base_errors'] / len(orig))
        byte_errors_list.append(stats['byte_errors'])

    return {
        'mean_base_error_rate': float(np.mean(base_error_rates)),
        'std_base_error_rate' : float(np.std(base_error_rates)),
        'mean_byte_errors'    : float(np.mean(byte_errors_list)),
        'max_byte_errors'     : int(max(byte_errors_list)) if byte_errors_list else 0,
    }
