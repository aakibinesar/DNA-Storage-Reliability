"""
channel_model.py  —  Vectorised DNA storage channel simulator
=============================================================
DeSP-inspired pipeline (Yuan et al. BMC Bioinformatics 2022):

  1. Synthesis   — n_initial_copies created with per-copy substitution errors
  2. Decay       — fraction of copies lost
  3. PCR         — amplification + replication errors  (VECTORISED)
  4. Sequencing  — sample eff_K reads, apply platform substitutions + indels

Reliability-skew model  (key addition for feature-predictable failures)
-----------------------------------------------------------------------
In a real DNA storage library, two sequence-level features systematically
reduce effective sequencing depth, making failure probability learnable:

  1. GC-content amplification bias (Lin et al. ISCA 2022; DeSP).
     Extreme-GC sequences are under-amplified across all PCR cycles:
         gc_factor = (1 - pcr_bias × |gc-0.5| × 4)^pcr_cycles

  2. Homopolymer PCR-stutter coverage loss.
     Long HP runs generate length-variant artifacts that are discarded
     before consensus, reducing effective K:
         hp_factor = 1 / (1 + hp_stutter_factor × max(0, max_hp_run - 2))

Combined:
    eff_K = max(1, round(K × gc_factor × hp_factor))

Both GC deviation and max_hp_run are in the ML feature set, giving the
model a mechanistically grounded signal to learn.

Vectorisation note
------------------
Pool is maintained as a 2-D uint8 array (n_copies × L).  All per-base
substitution and PCR replication operations are NumPy matrix ops.
Speedup vs. original character-loop code: ~63×.
Full label generation run (~0.4 hrs) is feasible.

Public API unchanged — no other file needs modification.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_BASES     : str            = 'ACGT'
_BASE_IDX  : dict[str, int] = {b: i for i, b in enumerate(_BASES)}
_INT_TO_BASE = np.array(['A', 'C', 'G', 'T'], dtype='U1')

_DEFAULT_TSUB = np.array([
    # ->A      ->C      ->G      ->T
    [0.0000, 0.0050, 0.0030, 0.0020],   # A->
    [0.0010, 0.0000, 0.0070, 0.0020],   # C->
    [0.0000, 0.0020, 0.0000, 0.0080],   # G->
    [0.0040, 0.0010, 0.0050, 0.0000],   # T->
], dtype=np.float64)

_PCR_POOL_CAP : int   = 2000
_HP_STUTTER   : float = 0.08   # HP coverage-loss factor per extra HP base beyond 2


# ---------------------------------------------------------------------------
# Utility: string <-> integer array
# ---------------------------------------------------------------------------

_LOOKUP = np.zeros(256, dtype=np.uint8)
for _b, _i in _BASE_IDX.items():
    _LOOKUP[ord(_b)] = _i


def _str_to_int(seq: str) -> np.ndarray:
    """DNA string -> uint8 array  (A=0, C=1, G=2, T=3)."""
    return _LOOKUP[np.frombuffer(seq.encode('ascii'), dtype=np.uint8)].copy()


def _int_to_str(arr: np.ndarray) -> str:
    """uint8 array -> DNA string."""
    return ''.join(_INT_TO_BASE[np.clip(arr, 0, 3)])


def _max_hp_run_from_int(arr: np.ndarray) -> int:
    """Compute max homopolymer run length from integer array (vectorised)."""
    if len(arr) == 0:
        return 0
    same  = np.concatenate([[1], (arr[1:] == arr[:-1]).astype(np.int32)])
    # Reset run counter at each new base
    runs  = np.zeros(len(arr), dtype=np.int32)
    runs[0] = 1
    for i in range(1, len(arr)):
        runs[i] = runs[i-1] + 1 if same[i] else 1
    return int(runs.max())


# ---------------------------------------------------------------------------
# Channel class
# ---------------------------------------------------------------------------

class DNAStorageChannel:
    """Vectorised, reliability-skew-aware DNA storage channel.

    Parameters
    ----------
    sub_rate          : per-base substitution rate (channel stage)
    indel_rate        : combined per-base insertion + deletion rate
    ins_fraction      : fraction of indel_rate that are insertions
    n_initial_copies  : copies created at synthesis
    synthesis_sub     : additional per-base sub rate during synthesis
    synthesis_indel   : per-base indel rate during synthesis
    pcr_cycles        : number of PCR amplification cycles
    pcr_fidelity      : per-base replication fidelity during PCR
    pcr_bias          : GC amplification bias coefficient.
                        Used in both per-copy PCR stage and effective-K model.
    decay_loss        : fraction of copies lost during decay
    seq_sub_rate      : sequencing platform per-base substitution rate
    seed              : RNG seed
    """

    def __init__(
        self,
        sub_rate:          float         = 0.05,
        indel_rate:        float         = 0.001,
        ins_fraction:      float         = 0.5,
        n_initial_copies:  int           = 100,
        synthesis_sub:     float         = 0.001,
        synthesis_indel:   float         = 0.001,
        pcr_cycles:        int           = 10,
        pcr_fidelity:      float         = 0.9999,
        pcr_bias:          float         = 0.05,
        decay_loss:        float         = 0.02,
        seq_sub_rate:      float         = 0.002,
        hp_stutter:        float         = _HP_STUTTER,
        seed:              Optional[int] = None,
    ) -> None:
        self.sub_rate          = sub_rate
        self.indel_rate        = indel_rate
        self.ins_frac          = ins_fraction
        self.n_initial_copies  = n_initial_copies
        self.synthesis_sub     = synthesis_sub
        self.synthesis_indel   = synthesis_indel
        self.pcr_cycles        = pcr_cycles
        self.pcr_fidelity      = pcr_fidelity
        self.pcr_bias          = pcr_bias
        self.decay_loss        = decay_loss
        self.seq_sub_rate      = seq_sub_rate
        self.hp_stutter        = hp_stutter
        self._rng              = np.random.default_rng(seed)

        self._tsub_synth = _build_tsub_cumulative(sub_rate + synthesis_sub)
        self._tsub_seq   = _build_tsub_cumulative(seq_sub_rate)
        self._pcr_err    = max(0.0, 1.0 - pcr_fidelity)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def simulate(self, original: str, coverage: int = 30) -> List[str]:
        """Run the full pipeline and return eff_K <= coverage noisy reads.

        eff_K is reduced for sequences with extreme GC content or long
        homopolymer runs (reliability-skew model — see module docstring).
        eff_K is always >= 1.

        Parameters
        ----------
        original : error-free input DNA sequence
        coverage : nominal sequencing depth K

        Returns
        -------
        List of eff_K DNA strings (variable length possible due to indels).
        """
        orig_int = _str_to_int(original)
        L        = len(orig_int)

        # Sequence-level coverage factors (reliability skew)
        gc_frac    = float(np.mean((orig_int == 1) | (orig_int == 2)))
        max_hp_run = _max_hp_run_from_int(orig_int)
        eff_cov    = self._effective_coverage(coverage, gc_frac, max_hp_run)

        pool = self._synthesis(orig_int)
        pool = self._decay(pool)
        pool = self._pcr(pool, gc_frac)
        reads = self._sequencing(pool, eff_cov, L)

        return reads

    def clone(self, seed: Optional[int] = None) -> 'DNAStorageChannel':
        """Return an independent copy of this channel with a fresh RNG."""
        return DNAStorageChannel(
            sub_rate         = self.sub_rate,
            indel_rate       = self.indel_rate,
            ins_fraction     = self.ins_frac,
            n_initial_copies = self.n_initial_copies,
            synthesis_sub    = self.synthesis_sub,
            synthesis_indel  = self.synthesis_indel,
            pcr_cycles       = self.pcr_cycles,
            pcr_fidelity     = self.pcr_fidelity,
            pcr_bias         = self.pcr_bias,
            decay_loss       = self.decay_loss,
            seq_sub_rate     = self.seq_sub_rate,
            hp_stutter       = self.hp_stutter,
            seed             = seed,
        )

    # ------------------------------------------------------------------
    # Reliability-skew: effective coverage
    # ------------------------------------------------------------------

    def _effective_coverage(
        self, coverage: int, gc_frac: float, max_hp_run: int
    ) -> int:
        """Compute joint GC- and HP-dependent effective sequencing depth.

        GC component (library amplification skew):
            gc_factor = max(0.05, 1 - pcr_bias × |gc-0.5| × 4) ^ pcr_cycles

        HP component (PCR stutter artifact filtering):
            hp_factor = 1 / (1 + HP_STUTTER × max(0, max_hp_run - 2))

        Combined:
            eff_K = max(1, round(K × gc_factor × hp_factor))

        At pcr_bias=0.05, pcr_cycles=10, HP_STUTTER=0.08 (K=5 examples):
            GC=0.50, HP=1 -> eff_K = 5  (no reduction)
            GC=0.40, HP=5 -> eff_K = 3  (moderate reduction)
            GC=0.30, HP=8 -> eff_K = 2  (large reduction)
        """
        # GC amplification factor
        gc_dev     = abs(gc_frac - 0.5)
        per_cycle  = max(0.05, 1.0 - self.pcr_bias * gc_dev * 4.0)
        gc_factor  = per_cycle ** self.pcr_cycles

        # HP stutter coverage loss
        hp_extra   = max(0, max_hp_run - 2)
        hp_factor  = 1.0 / (1.0 + self.hp_stutter * hp_extra)

        combined   = gc_factor * hp_factor
        return max(1, int(round(coverage * combined)))

    # ------------------------------------------------------------------
    # Pipeline stages
    # ------------------------------------------------------------------

    def _synthesis(self, orig_int: np.ndarray) -> np.ndarray:
        """Create n_initial_copies and apply synthesis-stage errors."""
        L = len(orig_int)
        n = max(1, int(self._rng.binomial(self.n_initial_copies, 0.99)))

        pool = np.tile(orig_int, (n, 1)).astype(np.uint8)
        pool = _apply_subs(pool, self._tsub_synth, self._rng)

        if self.synthesis_indel > 0:
            pool = self._sparse_indel_pass(pool, self.synthesis_indel, L)

        return pool

    def _decay(self, pool: np.ndarray) -> np.ndarray:
        """Remove decay_loss fraction of copies."""
        n = pool.shape[0]
        n_survive = max(1, int(self._rng.binomial(n, 1.0 - self.decay_loss)))
        idx = self._rng.choice(n, size=n_survive, replace=False)
        return pool[idx]

    def _pcr(self, pool: np.ndarray, gc_frac: float) -> np.ndarray:
        """Vectorised PCR amplification with GC-content replication bias."""
        gc_dev         = abs(gc_frac - 0.5)
        gc_bias_factor = max(0.1, 1.0 - self.pcr_bias * gc_dev * 4.0)

        for _ in range(self.pcr_cycles):
            n, L = pool.shape

            rep_mask   = self._rng.random(n) < gc_bias_factor
            n_rep      = int(rep_mask.sum())
            if n_rep == 0:
                continue

            new_copies = pool[rep_mask].copy()

            if self._pcr_err > 0:
                err_mask = self._rng.random((n_rep, L)) < self._pcr_err
                if err_mask.any():
                    shifts     = self._rng.integers(1, 4, size=(n_rep, L), dtype=np.uint8)
                    rand_bases = (new_copies + shifts) % 4
                    new_copies = np.where(err_mask, rand_bases, new_copies).astype(np.uint8)

            pool = np.concatenate([pool, new_copies], axis=0)
            if pool.shape[0] > _PCR_POOL_CAP:
                keep = self._rng.choice(pool.shape[0], size=_PCR_POOL_CAP, replace=False)
                pool = pool[keep]

        return pool

    def _sequencing(
        self, pool: np.ndarray, eff_cov: int, L: int
    ) -> List[str]:
        """Sample eff_cov reads and apply platform substitutions + indels."""
        n   = pool.shape[0]
        idx = self._rng.choice(n, size=eff_cov, replace=True)
        reads_int = pool[idx].copy()

        reads_int = _apply_subs(reads_int, self._tsub_seq, self._rng)

        reads: List[str] = []
        for k in range(eff_cov):
            s = _int_to_str(reads_int[k])
            if self.indel_rate > 0:
                s = _apply_indels(s, self.indel_rate, self.ins_frac, self._rng)
            reads.append(s)

        return reads

    def _sparse_indel_pass(
        self, pool: np.ndarray, indel_rate: float, target_len: int
    ) -> np.ndarray:
        """Apply indels only to rows that have >=1 indel event; trim/pad to L."""
        n, L     = pool.shape
        p_any    = 1.0 - np.exp(-L * indel_rate)
        affected = np.where(self._rng.random(n) < p_any)[0]

        for idx in affected:
            seq = _int_to_str(pool[idx])
            seq = _apply_indels(seq, indel_rate, self.ins_frac, self._rng)
            arr = _str_to_int(seq)
            if len(arr) >= target_len:
                pool[idx] = arr[:target_len]
            else:
                pool[idx, :len(arr)] = arr
                pool[idx, len(arr):] = 0

        return pool


# ---------------------------------------------------------------------------
# Vectorised helpers (module-level)
# ---------------------------------------------------------------------------

def _build_tsub_cumulative(rate: float) -> np.ndarray:
    """Build (4, 4) cumulative transition matrix scaled to *rate*.

    tsub_cumul[i, j] = P(base i -> any base in {0…j}).
    Diagonal = P(no substitution) = 1 - rate.

    Inverse-CDF decode: new_base = (r > tsub_cumul[old_base]).sum()
    """
    rate = float(np.clip(rate, 0.0, 0.99))
    tsub = _DEFAULT_TSUB.copy()
    for i in range(4):
        row_sum = tsub[i].sum()
        if row_sum > 0:
            tsub[i] *= rate / row_sum
        tsub[i, i] = 1.0 - tsub[i].sum()
    return np.cumsum(tsub, axis=1)


def _apply_subs(
    pool:       np.ndarray,
    tsub_cumul: np.ndarray,
    rng:        np.random.Generator,
) -> np.ndarray:
    """Vectorised per-base substitutions on pool array (n_seqs × L)."""
    n, L = pool.shape
    r         = rng.random((n, L))
    sub_probs = tsub_cumul[pool]                               # (n, L, 4)
    new_bases = (r[:, :, np.newaxis] > sub_probs).sum(axis=-1)
    return np.clip(new_bases, 0, 3).astype(np.uint8)


def _apply_indels(
    seq:        str,
    indel_rate: float,
    ins_frac:   float,
    rng:        np.random.Generator,
) -> str:
    """Apply insertions and deletions to a single read string.

    Fast path: ~90% of reads at indel_rate=0.001 skip via Poisson check.
    """
    L = len(seq)
    if L == 0:
        return seq
    if rng.random() > (1.0 - np.exp(-L * indel_rate)):
        return seq   # fast path

    ins_rate = indel_rate * ins_frac
    del_rate = indel_rate * (1.0 - ins_frac)
    r_ins    = rng.random(L)
    r_del    = rng.random(L)
    r_rand   = rng.integers(0, 4, size=L, dtype=np.uint8)

    result: List[str] = []
    for i, base in enumerate(seq):
        if r_ins[i] < ins_rate:
            result.append(_BASES[r_rand[i]])
        if r_del[i] < del_rate:
            continue
        result.append(base)

    return ''.join(result)


# ---------------------------------------------------------------------------
# Config factory
# ---------------------------------------------------------------------------

def build_channel_from_config(
    cfg:      dict,
    sub_rate: float,
    seed:     Optional[int] = None,
) -> DNAStorageChannel:
    """Instantiate a DNAStorageChannel from an experiment YAML config dict."""
    ch = cfg['channel']
    return DNAStorageChannel(
        sub_rate         = sub_rate,
        indel_rate       = ch['indel_rate'],
        ins_fraction     = ch['ins_del_ratio'],
        n_initial_copies = ch['n_initial_copies'],
        synthesis_sub    = ch['synthesis_sub_rate'],
        synthesis_indel  = ch['synthesis_del_rate'] + ch['synthesis_ins_rate'],
        pcr_cycles       = ch['n_pcr_cycles'],
        pcr_fidelity     = ch['pcr_fidelity'],
        pcr_bias         = ch['pcr_bias'],
        decay_loss       = ch['decay_loss_rate'],
        seq_sub_rate     = ch['sequencing_sub_rate'],
        hp_stutter       = ch.get('hp_stutter', _HP_STUTTER),
        seed             = seed,
    )


# R7: ablation variants for channel-mechanism validation
_ABLATION_VARIANTS = {
    'full'          : dict(),                              # original channel — reference
    'no_gc_skew'    : dict(pcr_bias=0.0),                 # disable GC-coverage bias
    'no_hp_stutter' : dict(hp_stutter=0.0),               # disable HP stutter coverage loss
    'flat_skew'     : dict(pcr_bias=0.0, hp_stutter=0.0), # both mechanisms disabled
}


def build_ablated_channel(
    cfg:      dict,
    sub_rate: float,
    ablation: str,
    seed:     Optional[int] = None,
) -> DNAStorageChannel:
    """Build a channel with one or both reliability-skew mechanisms disabled.

    Parameters
    ----------
    ablation : one of 'full', 'no_gc_skew', 'no_hp_stutter', 'flat_skew'
        'full'          — identical to build_channel_from_config (reference)
        'no_gc_skew'    — pcr_bias=0  -> gc_factor=1 for all sequences
        'no_hp_stutter' — hp_stutter=0 -> hp_factor=1 for all sequences
        'flat_skew'     — both mechanisms off; failures driven by sub/indel rates only

    Used in analysis/channel_ablation.py to test whether model predictions
    degrade when the injected reliability-skew features are removed (R7 fix).
    """
    if ablation not in _ABLATION_VARIANTS:
        raise ValueError(
            f"Unknown ablation '{ablation}'. "
            f"Choose from: {list(_ABLATION_VARIANTS)}"
        )
    overrides = _ABLATION_VARIANTS[ablation]
    ch = cfg['channel']
    return DNAStorageChannel(
        sub_rate         = sub_rate,
        indel_rate       = ch['indel_rate'],
        ins_fraction     = ch['ins_del_ratio'],
        n_initial_copies = ch['n_initial_copies'],
        synthesis_sub    = ch['synthesis_sub_rate'],
        synthesis_indel  = ch['synthesis_del_rate'] + ch['synthesis_ins_rate'],
        pcr_cycles       = ch['n_pcr_cycles'],
        pcr_fidelity     = ch['pcr_fidelity'],
        pcr_bias         = overrides.get('pcr_bias',    ch['pcr_bias']),
        decay_loss       = ch['decay_loss_rate'],
        seq_sub_rate     = ch['sequencing_sub_rate'],
        hp_stutter       = overrides.get('hp_stutter',  ch.get('hp_stutter', _HP_STUTTER)),
        seed             = seed,
    )
