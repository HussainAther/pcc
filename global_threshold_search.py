import numpy as np
import matplotlib.pyplot as plt
from typing import Union, Optional, Tuple, List, Dict

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
    t, H, *, k=5, threshold=None, threshold_quantile=0.10,
    smooth_w=15, t_stop=None
):
    t = np.asarray(t, float)
    H = np.asarray(H, float)

    if t_stop is not None:
        m = t <= float(t_stop)
        t = t[m]
        H = H[m]

    if len(t) < k + 2:
        return None, np.nan

    Sdot = np.gradient(H, t)
    Sdot = smooth(Sdot, smooth_w)

    if threshold is None:
        qv = np.quantile(Sdot[np.isfinite(Sdot)], threshold_quantile)
        used_thr = min(0.0, float(qv))
    else:
        used_thr = float(threshold)

    consec = 0
    for i, v in enumerate(Sdot):
        if not np.isfinite(v):
            consec = 0
            continue
        if v < used_thr:
            consec += 1
            if consec >= k:
                start = i - (k - 1)
                return float(t[start]), used_thr
        else:
            consec = 0

    return None, used_thr


def shannon_entropy(p: np.ndarray, eps: float = 1e-12) -> float:
    """Calculates S = -sum p log p."""
    p = np.asarray(p, dtype=float)
    p = np.clip(p, eps, 1.0)
    p = p / p.sum()
    return -float(np.sum(p * np.log(p)))

def entropy_timeseries(p_t: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Returns S_t shape (T,) from p_t shape (T, n_strat)."""
    p_t = np.asarray(p_t, dtype=float)
    S = np.empty(p_t.shape[0], dtype=float)
    for t in range(p_t.shape[0]):
        S[t] = shannon_entropy(p_t[t], eps=eps)
    return S

def centered_entropy_rate(S_t: np.ndarray, dt: float = 1.0) -> np.ndarray:
    """Centered finite-difference entropy rate."""
    S_t = np.asarray(S_t, dtype=float)
    Sdot = np.full_like(S_t, np.nan, dtype=float)
    if len(S_t) >= 3:
        Sdot[1:-1] = (S_t[2:] - S_t[:-2]) / (2.0 * dt)
    return Sdot

def fixation_time_from_fractions(p_t: np.ndarray, dt: float = 1.0, tol: float = 1e-12) -> Optional[int]:
    """Returns T_fix index or None."""
    p_t = np.asarray(p_t, dtype=float)
    for t in range(p_t.shape[0]):
        if np.any(p_t[t] >= 1.0 - tol):
            return int(round(t * dt))
    return None

def compute_lead_times(
    runs_p_t: List[np.ndarray],
    dt: float = 1.0,
    threshold: float = -1e-4,
    k: int = 5,
    fixation_tol: float = 1e-12,
) -> Tuple[np.ndarray, List[Dict]]:
    """Compute lead times tau = T_fix - T_warn for many runs."""
    taus = []
    details = []

    for i, p_t in enumerate(runs_p_t):
        p_t = np.asarray(p_t, dtype=float)
        S_t = entropy_timeseries(p_t)
        
        # We use a simple grid here; warn_time_from_trace is more robust for sampled data
        t_grid = np.arange(len(S_t)) * dt
        Twarn, _ = warn_time_from_trace(t_grid, S_t, k=k, threshold=threshold)
        Tfix = fixation_time_from_fractions(p_t, dt=dt, tol=fixation_tol)

        info = {"run": i, "T_warn": Twarn, "T_fix": Tfix, "tau": None}
        if (Twarn is not None) and (Tfix is not None) and (Tfix >= Twarn):
            tau = float(Tfix - Twarn)
            taus.append(tau)
            info["tau"] = tau

        details.append(info)

    return np.asarray(taus, dtype=float), details

def plot_tau_hist_from_npz(npz_path: str, eps: float, bins: int = 30):
    """Plots histogram of lead times for a specific epsilon from saved data."""
    data = np.load(npz_path, allow_pickle=True)
    if "taus_by_eps" not in data:
        print("taus_by_eps key not found.")
        return
        
    taus_by_eps = data["taus_by_eps"].item()
    keys = np.array([float(k) for k in taus_by_eps.keys()])
    k_val = float(keys[np.argmin(np.abs(keys - eps))])
    taus = np.asarray(taus_by_eps[str(k_val)], dtype=float)

    if taus.size == 0:
        print(f"No taus stored for eps={k_val:.3f}.")
        return

    plt.figure()
    plt.hist(taus, bins=bins, edgecolor='black')
    plt.xlabel(r"Lead time $\tau = T_{\mathrm{fix}} - T_{\mathrm{warn}}$")
    plt.ylabel("Count")
    plt.title(rf"Lead time distribution ($\epsilon \approx$ {k_val:.3f})")
    plt.tight_layout()

if __name__ == "__main__":
    # Example usage with synthetic data
    rng = np.random.default_rng(0)
    fake_runs = []
    for _ in range(30):
        T = 1000
        p = np.zeros((T, 3))
        # Start at 1/3, then drift to fixation
        p[0] = [0.33, 0.33, 0.34]
        for t in range(1, T):
            p[t] = np.clip(p[t-1] + rng.normal(0, 0.01, size=3), 0, 1)
            p[t] /= p[t].sum()
            if np.any(p[t] > 0.99):
                p[t:] = 0.0
                p[t:, np.argmax(p[t])] = 1.0
                break
        fake_runs.append(p)

    taus, _ = compute_lead_times(fake_runs, threshold=-1e-4, k=5)
    print(f"Mean tau: {np.mean(taus) if taus.size > 0 else 'N/A'}")
