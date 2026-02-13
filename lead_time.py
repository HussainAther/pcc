import numpy as np
from typing import Optional, Tuple

def smooth(y: np.ndarray, w: int = 11) -> np.ndarray:
    """Simple moving average, odd window."""
    y = np.asarray(y, dtype=float)
    if w <= 1:
        return y
    w = int(w)
    if w % 2 == 0:
        w += 1
    kernel = np.ones(w) / w
    return np.convolve(y, kernel, mode="same")

def warn_time_from_trace(
    t: np.ndarray,
    H: np.ndarray,
    *,
    k: int = 5,
    threshold: Optional[float] = None,
    threshold_quantile: float = 0.10,
    smooth_w: int = 15,
    min_valid: int = 10,
    t_stop: Optional[float] = None,
) -> Tuple[Optional[float], float]:
    """
    Detects warning time. 
    If t_stop is provided, only data up to that time is considered.
    """
    t = np.asarray(t, dtype=float)
    H = np.asarray(H, dtype=float)

    # Restrict analysis to data before fixation if t_stop is provided
    if t_stop is not None:
        mask = t <= float(t_stop)
        t = t[mask]
        H = H[mask]

    if t.size < max(min_valid, k + 2):
        return None, np.nan

    Sdot = np.gradient(H, t)
    Sdot = smooth(Sdot, smooth_w)

    if threshold is None:
        finite_vals = Sdot[np.isfinite(Sdot)]
        if finite_vals.size == 0:
            return None, np.nan
        qv = np.quantile(finite_vals, threshold_quantile)
        used_thr = min(0.0, float(qv))
    else:
        used_thr = float(threshold)

    consec = 0
    for i in range(Sdot.size):
        v = Sdot[i]
        if np.isfinite(v) and v < used_thr:
            consec += 1
            if consec >= k:
                return float(t[i - (k - 1)]), used_thr
        else:
            consec = 0

    return None, used_thr
