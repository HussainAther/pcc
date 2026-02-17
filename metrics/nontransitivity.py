"""
metrics/nontransitivity.py

Non-transitivity utilities for cyclic dominance / PCC.

Supports:
- Payoff matrices A (NxN): directed edge i->j if i "beats" j (A_ij > A_ji or A_ij>0 depending on mode)
- Empirical win-rate matrices W (NxN): directed edge i->j if W_ij > 0.5 (or margin threshold)

Core metric:
- Intransitivity index = (# directed 3-cycles) / (total # triples)
  where a directed 3-cycle is i->j, j->k, k->i.

Also includes:
- rps_index_3x3: score in [0,1] for how close a 3x3 dominance structure is to a clean RPS loop.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable, List, Tuple, Optional
import numpy as np


@dataclass(frozen=True)
class CycleSummary:
    n: int
    triple_count: int
    cycle_count: int
    intransitivity_index: float
    cycles: List[Tuple[int, int, int]]  # oriented triples (i,j,k) with i->j->k->i


def dominance_from_payoff(
    A: np.ndarray,
    *,
    mode: str = "pairwise",   # "pairwise" uses A_ij > A_ji ; "sign" uses A_ij > 0
    margin: float = 0.0       # require difference > margin
) -> np.ndarray:
    """
    Build a directed dominance adjacency matrix D from payoff matrix A.

    Returns D where D[i,j]=True means i beats j.
    - mode="pairwise": i beats j if A[i,j] - A[j,i] > margin
    - mode="sign":     i beats j if A[i,j] > margin  (less recommended unless A encodes direct advantage)
    """
    A = np.asarray(A, float)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("A must be square (NxN).")
    n = A.shape[0]

    D = np.zeros((n, n), dtype=bool)
    if mode == "pairwise":
        diff = A - A.T
        D = diff > margin
    elif mode == "sign":
        D = A > margin
        np.fill_diagonal(D, False)
    else:
        raise ValueError("mode must be 'pairwise' or 'sign'.")

    np.fill_diagonal(D, False)
    return D


def dominance_from_winrate(
    W: np.ndarray,
    *,
    threshold: float = 0.5,   # i beats j if W_ij > threshold
    margin: float = 0.0       # require W_ij - W_ji > margin (optional)
) -> np.ndarray:
    """
    Build dominance adjacency D from empirical win-rates W (NxN),
    where W[i,j] is prob i beats j (or fraction of wins).
    """
    W = np.asarray(W, float)
    if W.ndim != 2 or W.shape[0] != W.shape[1]:
        raise ValueError("W must be square (NxN).")
    n = W.shape[0]

    # default: compare against threshold (0.5)
    D = W > threshold

    # optional: enforce asymmetry margin (more robust if W noisy)
    if margin > 0:
        D = (W - W.T) > margin

    np.fill_diagonal(D, False)
    return D


def list_3cycles(D: np.ndarray) -> List[Tuple[int, int, int]]:
    """
    Enumerate directed 3-cycles i->j->k->i (one orientation).
    Returns oriented triples (i,j,k) with i<j<k enforced via canonicalization.
    """
    D = np.asarray(D, bool)
    if D.ndim != 2 or D.shape[0] != D.shape[1]:
        raise ValueError("D must be square (NxN).")
    n = D.shape[0]

    cycles = []
    # iterate all triples i<j<k
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                # Check the two possible directed 3-cycles on {i,j,k}:
                # i->j->k->i OR i->k->j->i
                if D[i, j] and D[j, k] and D[k, i]:
                    cycles.append((i, j, k))
                elif D[i, k] and D[k, j] and D[j, i]:
                    cycles.append((i, k, j))  # oriented representation
    return cycles


def intransitivity_summary(D: np.ndarray) -> CycleSummary:
    """
    Compute intransitivity index and cycle count for directed adjacency D.
    """
    D = np.asarray(D, bool)
    n = D.shape[0]
    triple_count = n * (n - 1) * (n - 2) // 6
    cycles = list_3cycles(D)
    cycle_count = len(cycles)
    idx = (cycle_count / triple_count) if triple_count > 0 else 0.0
    return CycleSummary(
        n=n,
        triple_count=triple_count,
        cycle_count=cycle_count,
        intransitivity_index=float(idx),
        cycles=cycles
    )


def rps_index_3x3(D: np.ndarray) -> float:
    """
    For a 3x3 dominance matrix D, return a score in [0,1] measuring how close
    it is to a clean single RPS cycle (one directed 3-cycle and no ties).

    Score definition:
    - 1.0 if exactly one directed edge per unordered pair AND the triple forms a 3-cycle
    - 0.0 otherwise (ties or transitive ordering)
    """
    D = np.asarray(D, bool)
    if D.shape != (3, 3):
        raise ValueError("rps_index_3x3 expects a 3x3 dominance matrix.")

    # For each unordered pair (i,j), require exactly one direction is True.
    pairs = [(0, 1), (0, 2), (1, 2)]
    for i, j in pairs:
        if (D[i, j] == D[j, i]):  # both True (inconsistent) or both False (tie)
            return 0.0

    # Now check if it's a directed 3-cycle rather than transitive.
    cycles = list_3cycles(D)
    return 1.0 if len(cycles) == 1 else 0.0


def winrate_from_counts(wins: np.ndarray, losses: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    Convenience: convert pairwise win/loss counts into winrate matrix W.
    wins[i,j] = number of times i beat j
    losses[i,j] = number of times i lost to j (usually losses = wins.T)
    """
    wins = np.asarray(wins, float)
    losses = np.asarray(losses, float)
    if wins.shape != losses.shape or wins.ndim != 2 or wins.shape[0] != wins.shape[1]:
        raise ValueError("wins and losses must be same square shape.")

    denom = wins + losses + eps
    W = wins / denom
    np.fill_diagonal(W, 0.0)
    return W


def pretty_cycle_report(summary: CycleSummary, max_show: int = 10) -> str:
    """
    Human-readable report for logs / manuscript appendix.
    """
    lines = []
    lines.append(f"n={summary.n} strategies")
    lines.append(f"triples={summary.triple_count}")
    lines.append(f"3-cycles={summary.cycle_count}")
    lines.append(f"intransitivity_index={summary.intransitivity_index:.3f}")
    if summary.cycle_count > 0:
        lines.append(f"example cycles (up to {max_show}):")
        for t in summary.cycles[:max_show]:
            lines.append(f"  {t[0]} -> {t[1]} -> {t[2]} -> {t[0]}")
    return "\n".join(lines)

