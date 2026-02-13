# metrics.py
import numpy as np

def shannon_entropy_from_counts(counts: np.ndarray) -> float:
    probs = counts / np.sum(counts)
    return float(-np.sum(probs * np.log2(probs + 1e-12)))

def shannon_entropy_grid(grid: np.ndarray) -> float:
    _, counts = np.unique(grid, return_counts=True)
    return shannon_entropy_from_counts(counts)

def proportions_from_grid(grid: np.ndarray) -> np.ndarray:
    vals, counts = np.unique(grid, return_counts=True)
    p = np.zeros(3, dtype=float)
    for v, c in zip(vals, counts):
        p[int(v)-1] = c
    p /= p.sum()
    return p

def first_time_below(traj: np.ndarray, eps: float = 1e-3):
    """Return index of first time any component drops below eps, else None."""
    m = np.min(traj, axis=1)
    idx = np.where(m < eps)[0]
    return int(idx[0]) if idx.size else None

