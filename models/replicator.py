import numpy as np

def replicator_rhs(x: np.ndarray, A: np.ndarray) -> np.ndarray:
    Ax = A @ x
    phi = float(x @ Ax)
    return x * (Ax - phi)

def simulate_replicator(A: np.ndarray, x0=None, dt=1e-3, T=10.0):
    A = np.asarray(A, float)
    n = A.shape[0]
    if x0 is None:
        x = np.ones(n) / n
        x = x + 0.01*np.random.default_rng(0).normal(size=n)
        x = np.clip(x, 1e-6, None)
        x = x / x.sum()
    else:
        x = np.asarray(x0, float)
        x = np.clip(x, 1e-9, None)
        x = x / x.sum()

    steps = int(T/dt)
    xs = np.zeros((steps+1, n))
    ts = np.linspace(0, T, steps+1)
    xs[0] = x

    for i in range(steps):
        k1 = replicator_rhs(x, A)
        x = x + dt*k1
        x = np.clip(x, 1e-12, None)
        x = x / x.sum()
        xs[i+1] = x
    return ts, xs

