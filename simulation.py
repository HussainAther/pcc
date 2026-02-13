import numpy as np
from dataclasses import dataclass
from scipy.integrate import solve_ivp
from scipy.optimize import root

@dataclass(frozen=True)
class Params:
    mu: float = 0.1          # stabilizing leakage to uniform
    kappa: float = 5.0       # cubic saturation
    sigma0: float = 0.0      # baseline gain
    sigma1: float = 0.0      # endogenous gain coefficient
    use_entropy: bool = False  # if True, sigma_eff depends on entropy instead of r^2

def rhs(t: float, x: np.ndarray, p: Params) -> np.ndarray:
    x = np.asarray(x, dtype=float)

    # --- cyclic PCC core (symmetric LV form) ---
    # P: Producer, Co: Consumer, Ch: Changer (or Rock, Paper, Scissors)
    P, Co, Ch = x
    alpha = beta = gamma = 1.0  # fixed for bifurcation stability

    core = np.array([
        P  * (alpha * Ch - beta * Co),
        Co * (beta  * P  - gamma * Ch),
        Ch * (gamma * Co - alpha * P),
    ], dtype=float)

    # --- leakage toward uniform (stabilizing) ---
    u = np.array([1/3, 1/3, 1/3], dtype=float)
    leak = p.mu * (u - x)

    # --- state measure: polarization r^2 or entropy H ---
    if p.use_entropy:
        # Shannon entropy (0..log2(3))
        probs = np.clip(x, 1e-12, 1.0)
        H = -np.sum(probs * np.log2(probs))
        Hmax = np.log2(3.0)
        # instability rises as entropy drops (away from uniform)
        state_signal = 1.0 - (H / Hmax)
    
    else:
        # polarization (L2 norm distance from uniform)
        # Using the square root is essential to make gain linear-order
        state_signal = np.sqrt(np.sum((x - u) ** 2))

    sigma_eff = p.sigma0 + p.sigma1 * state_signal


    # --- destabilizing gain (push away from uniform) ---
    gain = sigma_eff * (x - u)

    # --- cubic saturation (limits amplitude) ---
    r2 = float(np.sum((x - u) ** 2))
    sat = -p.kappa * r2 * (x - u)

    dx = core + leak + gain + sat
    return dx

def rhs_2d(t: float, z: np.ndarray, p: Params) -> np.ndarray:
    """Reduced dynamics on 2D simplex: x3 = 1 - x1 - x2."""
    x1, x2 = z
    x3 = 1.0 - (x1 + x2)
    dx = rhs(t, np.array([x1, x2, x3]), p)
    return dx[:2]

def jacobian_2d_numeric(z, p, h=1e-7):
    """Numerical Jacobian on the 2D manifold for stability analysis."""
    J = np.zeros((2, 2))
    for k in range(2):
        dz = np.zeros(2)
        dz[k] = h
        # Compute partial derivatives using central difference
        fp = rhs_2d(0.0, z + dz, p)
        fm = rhs_2d(0.0, z - dz, p)
        J[:, k] = (fp - fm) / (2.0 * h)
    return J

def max_realpart_eigs_at_eq(params: Params) -> float:
    """Calculates stability of the uniform equilibrium (1/3, 1/3, 1/3)."""
    z_star = np.array([1/3, 1/3])
    J2 = jacobian_2d_numeric(z_star, params)
    eig = np.linalg.eigvals(J2)
    return float(np.max(np.real(eig)))

def simulate(y0=(0.33, 0.33, 0.34), t_span=(0.0, 800.0), n_points=16000, params=Params()):
    y0 = np.array(y0, dtype=float)
    y0 = np.clip(y0, 1e-12, 1.0)
    y0 = y0 / y0.sum()

    t_eval = np.linspace(t_span[0], t_span[1], n_points)
    sol = solve_ivp(
        fun=lambda t, y: rhs(t, y, params), 
        t_span=t_span, 
        y0=y0, 
        t_eval=t_eval, 
        rtol=1e-8, 
        atol=1e-11
    )
    
    y = sol.y.T
    y = np.clip(y, 0.0, 1.0)
    s = y.sum(axis=1, keepdims=True)
    y = np.divide(y, s, out=y, where=s > 0)
    
    return sol.t, y
