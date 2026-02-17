import numpy as np
from numba import njit
from dataclasses import dataclass

@dataclass
class TrialResult:
    fixated: bool
    t_fix: float
    t_drop: float        # NEW
    t: np.ndarray
    H: np.ndarray
    p: np.ndarray


@njit
def _run_mcs_step(grid, counts, L, attempts, neighbors, base_p, eps_gain, epsilon, sigma_noise):
    """
    Perform interaction attempts for a single Monte Carlo Step.
    """
    N_inv = 1.0 / (L * L)
    for _ in range(attempts):
        i, j = np.random.randint(0, L), np.random.randint(0, L)
        idx = np.random.randint(0, 8)
        di, dj = neighbors[idx]
        ni, nj = (i + di) % L, (j + dj) % L
        
        a = grid[i, j]
        b = grid[ni, nj]

        if a == b:
            continue

        # RPS Cyclic Dominance: 0 eats 1, 1 eats 2, 2 eats 0
        if (a - b) % 3 == 2:
            winner, loser, wi, wj = a, b, ni, nj
        elif (b - a) % 3 == 2:
            winner, loser, wi, wj = b, a, i, j
        else:
            continue

        p_w = counts[winner] * N_inv
        p_l = counts[loser] * N_inv
        delta = p_w - p_l
        
        p_win = base_p + eps_gain * epsilon * delta
        if sigma_noise > 0.0:
            p_win += np.random.normal(0.0, sigma_noise)

        # Clamp p_win
        if p_win < 0.0: 
            p_win = 0.0
        elif p_win > 1.0: 
            p_win = 1.0

        if np.random.random() < p_win:
            old = grid[wi, wj]
            grid[wi, wj] = winner
            counts[old] -= 1
            counts[winner] += 1

def run_spatial_trial(
    epsilon: float,
    sigma_noise: float,
    seed: int,
    *,
    L: int = 100,
    T_max: int = 50_000,
    sample_every: int = 50,
    base_p: float = 0.5,
    eps_gain: float = 6.0,
    attempts_per_mcs: int | None = None,
    fixation_tol: float | None = None,
    m_sites: int = 5,
) -> TrialResult:
    np.random.seed(seed)
    grid = np.random.randint(0, 3, size=(L, L), dtype=np.int8)
    counts = np.bincount(grid.ravel(), minlength=3).astype(np.int64)
    N = L * L

    attempts = N if attempts_per_mcs is None else int(attempts_per_mcs)

    if fixation_tol is None:
        fix_tol = float(m_sites) / N
    else:
        fix_tol = float(fixation_tol)

    max_samples = (T_max // sample_every) + 2
    t_list = np.zeros(max_samples)
    p_list = np.zeros((max_samples, 3))
    s = 0

    fixated = False
    t_fix = float(T_max)
    t_drop = np.nan  

    neighbors = np.array([
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1),           (0, 1),
        (1, -1),  (1, 0),  (1, 1)
    ], dtype=np.int32)

    for mcs in range(T_max + 1):
        p_current = counts.astype(np.float64) / N
        
        # A) Track first extinction using m_sites threshold
        if np.isnan(t_drop) and (counts.min() <= m_sites):
            t_drop = float(mcs)

        # Sampling block
        if (mcs % sample_every) == 0:
            t_list[s] = float(mcs)
            p_list[s] = p_current
            s += 1

            # Fixation check (one species dominates)
            if p_current.max() >= 1.0 - fix_tol:
                fixated = True
                t_fix = float(mcs)
                break
        
        # B) Stop simulation if any species hits the m_sites threshold
        if counts.min() <= m_sites:
            fixated = True
            t_fix = float(mcs)
            if np.isnan(t_drop):
                t_drop = float(mcs)
            
            # Ensure final state is recorded even if not on a sample_every step
            if s < max_samples:
                t_list[s], p_list[s] = float(mcs), p_current
                s += 1
            break

        if mcs == T_max:
            break

        _run_mcs_step(
            grid, counts, L, attempts, neighbors, 
            base_p, eps_gain, epsilon, sigma_noise
        )

    # Truncate results
    t = t_list[:s].copy()
    p = p_list[:s].copy()

    # Shannon Entropy calculation
    q = np.clip(p, 1e-12, 1.0)
    q = q / q.sum(axis=1, keepdims=True)
    H = -np.sum(q * np.log(q), axis=1)

    return TrialResult(
        fixated=fixated,
        t_fix=t_fix,
        t_drop=t_drop,
        t=t,
        H=H,
        p=p
    )

if __name__ == "__main__":
    params = {
        "epsilon": 0.50,
        "sigma_noise": 0.05,
        "seed": 42,
        "L": 40,
        "T_max": 30000,
        "eps_gain": 20.0,
        "sample_every": 100,
        "m_sites": 5
    }
    
    print("Executing spatial trial with aggressive epsilon drift...")
    res = run_spatial_trial(**params)

    print(f"Fixated: {res.fixated}")
    print(f"Time to Fixation/Extinction: {res.t_fix}")
    print(f"Time of First Extinction: {res.t_drop}")
    print(f"Final Abundance: {res.p[-1]}")
    print(f"Final Entropy: {res.H[-1]:.4f}")