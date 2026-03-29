#!/usr/bin/env python3
"""
pcc_ebid_sim.py

GPU-accelerated PCC + EBID simulation using PyTorch CUDA.

Conceptual mapping:
- Pressure: drives local amplification / state transitions
- Control: smooths / stabilizes toward neighborhood consensus
- Chaos: injects stochastic perturbation
- EBID: tracks entropy-based instability dynamics over time

This script:
1. Creates batches of 2D continuous-state grids
2. Evolves them under Pressure / Control / Chaos dynamics
3. Computes entropy per simulation over time
4. Saves results for analysis / plotting

Example:
    python pcc_ebid_sim.py \
        --device cuda \
        --grid-size 128 \
        --steps 300 \
        --batch-size 64 \
        --sweep-size 4 \
        --out-dir results_pcc_ebid

A 4x4x4 sweep creates 64 simulations in parallel:
- pressure in linspace(p_min, p_max, 4)
- control in linspace(c_min, c_max, 4)
- chaos in linspace(k_min, k_max, 4)
"""

from __future__ import annotations

import argparse
import csv
import math
import os
from dataclasses import dataclass
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F


@dataclass
class SimConfig:
    grid_size: int
    steps: int
    batch_size: int
    seed: int
    dt: float
    pressure_min: float
    pressure_max: float
    control_min: float
    control_max: float
    chaos_min: float
    chaos_max: float
    sweep_size: int
    entropy_bins: int
    init_mode: str
    init_density: float
    save_every: int
    out_dir: str
    device: str


def parse_args() -> SimConfig:
    parser = argparse.ArgumentParser(description="PCC + EBID PyTorch CUDA simulation")

    parser.add_argument("--grid-size", type=int, default=128)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=None,
                        help="If omitted, inferred from sweep-size^3")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dt", type=float, default=0.20)

    parser.add_argument("--pressure-min", type=float, default=0.00)
    parser.add_argument("--pressure-max", type=float, default=1.00)
    parser.add_argument("--control-min", type=float, default=0.00)
    parser.add_argument("--control-max", type=float, default=1.00)
    parser.add_argument("--chaos-min", type=float, default=0.00)
    parser.add_argument("--chaos-max", type=float, default=0.50)

    parser.add_argument("--sweep-size", type=int, default=4,
                        help="Number of values per PCC axis; total batch = sweep_size^3")
    parser.add_argument("--entropy-bins", type=int, default=32)

    parser.add_argument("--init-mode", choices=["random", "sparse", "center_blob"], default="random")
    parser.add_argument("--init-density", type=float, default=0.25)

    parser.add_argument("--save-every", type=int, default=50,
                        help="Save intermediate snapshot every N steps; 0 disables")
    parser.add_argument("--out-dir", type=str, default="results_pcc_ebid")
    parser.add_argument("--device", type=str, default="cuda")

    args = parser.parse_args()

    batch_size = args.batch_size
    inferred_batch = args.sweep_size ** 3
    if batch_size is None:
        batch_size = inferred_batch
    elif batch_size != inferred_batch:
        raise ValueError(
            f"batch-size must equal sweep-size^3 for full PCC sweep. "
            f"Got batch_size={batch_size}, sweep_size^3={inferred_batch}"
        )

    return SimConfig(
        grid_size=args.grid_size,
        steps=args.steps,
        batch_size=batch_size,
        seed=args.seed,
        dt=args.dt,
        pressure_min=args.pressure_min,
        pressure_max=args.pressure_max,
        control_min=args.control_min,
        control_max=args.control_max,
        chaos_min=args.chaos_min,
        chaos_max=args.chaos_max,
        sweep_size=args.sweep_size,
        entropy_bins=args.entropy_bins,
        init_mode=args.init_mode,
        init_density=args.init_density,
        save_every=args.save_every,
        out_dir=args.out_dir,
        device=args.device,
    )


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_device(device_str: str) -> torch.device:
    if device_str == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but not available. Falling back to CPU.")
        return torch.device("cpu")
    return torch.device(device_str)


def make_output_dirs(out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(os.path.join(out_dir, "snapshots"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "plots"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "data"), exist_ok=True)


def create_parameter_grid(cfg: SimConfig, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    p_vals = torch.linspace(cfg.pressure_min, cfg.pressure_max, cfg.sweep_size, device=device)
    c_vals = torch.linspace(cfg.control_min, cfg.control_max, cfg.sweep_size, device=device)
    k_vals = torch.linspace(cfg.chaos_min, cfg.chaos_max, cfg.sweep_size, device=device)

    P, C, K = torch.meshgrid(p_vals, c_vals, k_vals, indexing="ij")
    pressure = P.reshape(-1)
    control = C.reshape(-1)
    chaos = K.reshape(-1)

    return pressure, control, chaos


def initialize_grid(cfg: SimConfig, device: torch.device) -> torch.Tensor:
    """
    Returns tensor of shape [B, 1, H, W] with values in [0, 1].
    """
    B = cfg.batch_size
    H = W = cfg.grid_size

    if cfg.init_mode == "random":
        grid = torch.rand((B, 1, H, W), device=device)

    elif cfg.init_mode == "sparse":
        grid = (torch.rand((B, 1, H, W), device=device) < cfg.init_density).float()

    elif cfg.init_mode == "center_blob":
        grid = torch.zeros((B, 1, H, W), device=device)
        yy, xx = torch.meshgrid(
            torch.arange(H, device=device),
            torch.arange(W, device=device),
            indexing="ij"
        )
        cy, cx = H // 2, W // 2
        radius = max(4, H // 10)
        dist2 = (yy - cy) ** 2 + (xx - cx) ** 2
        blob = (dist2 <= radius ** 2).float()[None, None, :, :]
        grid = grid + blob
        grid = torch.clamp(grid + 0.1 * torch.rand_like(grid), 0.0, 1.0)

    else:
        raise ValueError(f"Unknown init_mode: {cfg.init_mode}")

    return grid


def laplacian_kernel(device: torch.device, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    kernel = torch.tensor(
        [[0.0,  1.0, 0.0],
         [1.0, -4.0, 1.0],
         [0.0,  1.0, 0.0]],
        device=device,
        dtype=dtype
    )
    return kernel.view(1, 1, 3, 3)


def smooth_kernel(device: torch.device, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    kernel = torch.tensor(
        [[1.0, 1.0, 1.0],
         [1.0, 1.0, 1.0],
         [1.0, 1.0, 1.0]],
        device=device,
        dtype=dtype
    ) / 9.0
    return kernel.view(1, 1, 3, 3)


def local_ops(x: torch.Tensor, lap_kernel: torch.Tensor, avg_kernel: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Uses circular padding for toroidal boundary conditions.
    """
    x_pad = F.pad(x, (1, 1, 1, 1), mode="circular")
    lap = F.conv2d(x_pad, lap_kernel)
    avg = F.conv2d(x_pad, avg_kernel)
    return lap, avg


def compute_entropy_batch(x: torch.Tensor, bins: int = 32, eps: float = 1e-8) -> torch.Tensor:
    """
    Approximate Shannon entropy over cell-value distribution for each simulation.
    x shape: [B, 1, H, W], values expected in [0, 1]
    returns: [B]
    """
    B = x.shape[0]
    flat = x.view(B, -1)
    hist_list = []

    # torch.histc is per-tensor, so loop over batch for robustness.
    # This is okay because B is usually moderate and the heavy sim work stays on GPU.
    for i in range(B):
        h = torch.histc(flat[i], bins=bins, min=0.0, max=1.0)
        hist_list.append(h)

    hist = torch.stack(hist_list, dim=0)
    probs = hist / hist.sum(dim=1, keepdim=True).clamp_min(eps)
    entropy = -(probs * torch.log(probs.clamp_min(eps))).sum(dim=1)
    return entropy


def instability_metric(entropy_t: torch.Tensor, entropy_prev: torch.Tensor) -> torch.Tensor:
    """
    EBID-style simple instability metric: absolute entropy change.
    """
    return torch.abs(entropy_t - entropy_prev)


@torch.no_grad()
def pcc_update(
    x: torch.Tensor,
    pressure: torch.Tensor,
    control: torch.Tensor,
    chaos: torch.Tensor,
    dt: float,
    lap_kernel: torch.Tensor,
    avg_kernel: torch.Tensor,
) -> torch.Tensor:
    """
    Continuous PCC dynamics on [0,1] grid.

    Terms:
    - Pressure: nonlinear amplification using local contrast and reaction term
    - Control: smoothing toward neighborhood mean
    - Chaos: Gaussian noise injection

    pressure/control/chaos shapes: [B]
    x shape: [B,1,H,W]
    """
    B = x.shape[0]
    lap, avg = local_ops(x, lap_kernel, avg_kernel)

    p = pressure.view(B, 1, 1, 1)
    c = control.view(B, 1, 1, 1)
    k = chaos.view(B, 1, 1, 1)

    # Pressure term:
    # - x*(1-x) encourages active transitions around midrange states
    # - lap encourages local spread / amplification of spatial change
    pressure_term = (x * (1.0 - x)) + 0.5 * lap

    # Control term:
    # - pulls system toward local consensus / smoothness
    control_term = avg - x

    # Chaos term:
    # - stochastic perturbation
    noise = torch.randn_like(x)
    chaos_term = noise

    dx = p * pressure_term + c * control_term + k * chaos_term
    x_next = x + dt * dx

    return torch.clamp(x_next, 0.0, 1.0)


def save_snapshot_grid(
    x: torch.Tensor,
    pressure: torch.Tensor,
    control: torch.Tensor,
    chaos: torch.Tensor,
    out_path: str,
    max_images: int = 16,
) -> None:
    """
    Save a grid of simulation snapshots to PNG.
    """
    x_cpu = x.detach().cpu().squeeze(1).numpy()
    p_cpu = pressure.detach().cpu().numpy()
    c_cpu = control.detach().cpu().numpy()
    k_cpu = chaos.detach().cpu().numpy()

    n = min(max_images, x_cpu.shape[0])
    cols = int(math.ceil(math.sqrt(n)))
    rows = int(math.ceil(n / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows))
    axes = np.array(axes).reshape(rows, cols)

    for idx in range(rows * cols):
        ax = axes.flat[idx]
        ax.axis("off")
        if idx < n:
            ax.imshow(x_cpu[idx], interpolation="nearest")
            ax.set_title(
                f"P={p_cpu[idx]:.2f}\nC={c_cpu[idx]:.2f}\nK={k_cpu[idx]:.2f}",
                fontsize=8
            )

    plt.tight_layout()
    plt.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def save_entropy_plot(
    entropy_history: np.ndarray,
    pressure: torch.Tensor,
    control: torch.Tensor,
    chaos: torch.Tensor,
    out_path: str,
    max_lines: int = 16,
) -> None:
    """
    entropy_history shape: [T, B]
    """
    p_cpu = pressure.detach().cpu().numpy()
    c_cpu = control.detach().cpu().numpy()
    k_cpu = chaos.detach().cpu().numpy()

    B = entropy_history.shape[1]
    n = min(B, max_lines)

    plt.figure(figsize=(10, 6))
    for i in range(n):
        label = f"P={p_cpu[i]:.2f}, C={c_cpu[i]:.2f}, K={k_cpu[i]:.2f}"
        plt.plot(entropy_history[:, i], label=label)

    plt.xlabel("Step")
    plt.ylabel("Entropy")
    plt.title("Entropy over Time for Selected PCC Regimes")
    if n <= 12:
        plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close()


def save_summary_csv(
    entropy_history: np.ndarray,
    instability_history: np.ndarray,
    pressure: torch.Tensor,
    control: torch.Tensor,
    chaos: torch.Tensor,
    out_path: str,
) -> None:
    p_cpu = pressure.detach().cpu().numpy()
    c_cpu = control.detach().cpu().numpy()
    k_cpu = chaos.detach().cpu().numpy()

    final_entropy = entropy_history[-1]
    mean_entropy = entropy_history.mean(axis=0)
    max_entropy = entropy_history.max(axis=0)

    mean_instability = instability_history.mean(axis=0)
    max_instability = instability_history.max(axis=0)

    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "sim_id",
            "pressure",
            "control",
            "chaos",
            "final_entropy",
            "mean_entropy",
            "max_entropy",
            "mean_instability",
            "max_instability",
        ])
        for i in range(len(p_cpu)):
            writer.writerow([
                i,
                float(p_cpu[i]),
                float(c_cpu[i]),
                float(k_cpu[i]),
                float(final_entropy[i]),
                float(mean_entropy[i]),
                float(max_entropy[i]),
                float(mean_instability[i]),
                float(max_instability[i]),
            ])


def save_phase_table(
    entropy_history: np.ndarray,
    pressure: torch.Tensor,
    control: torch.Tensor,
    chaos: torch.Tensor,
    out_path: str,
) -> None:
    """
    Saves a simple long-format CSV that can be pivoted later into heatmaps.
    """
    p_cpu = pressure.detach().cpu().numpy()
    c_cpu = control.detach().cpu().numpy()
    k_cpu = chaos.detach().cpu().numpy()

    mean_entropy = entropy_history.mean(axis=0)
    final_entropy = entropy_history[-1]

    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "sim_id",
            "pressure",
            "control",
            "chaos",
            "mean_entropy",
            "final_entropy",
        ])
        for i in range(len(p_cpu)):
            writer.writerow([
                i,
                float(p_cpu[i]),
                float(c_cpu[i]),
                float(k_cpu[i]),
                float(mean_entropy[i]),
                float(final_entropy[i]),
            ])


@torch.no_grad()
def run_simulation(cfg: SimConfig) -> None:
    device = ensure_device(cfg.device)
    make_output_dirs(cfg.out_dir)
    set_seed(cfg.seed)

    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    pressure, control, chaos = create_parameter_grid(cfg, device)
    x = initialize_grid(cfg, device)

    lap_kernel = laplacian_kernel(device)
    avg_kernel = smooth_kernel(device)

    entropy_history = []
    instability_history = []

    entropy_prev = compute_entropy_batch(x, bins=cfg.entropy_bins)
    entropy_history.append(entropy_prev.detach().cpu().numpy())
    instability_history.append(torch.zeros_like(entropy_prev).cpu().numpy())

    initial_snapshot_path = os.path.join(cfg.out_dir, "snapshots", "snapshot_step_0000.png")
    save_snapshot_grid(x, pressure, control, chaos, initial_snapshot_path)

    for step in range(1, cfg.steps + 1):
        x = pcc_update(
            x=x,
            pressure=pressure,
            control=control,
            chaos=chaos,
            dt=cfg.dt,
            lap_kernel=lap_kernel,
            avg_kernel=avg_kernel,
        )

        entropy_t = compute_entropy_batch(x, bins=cfg.entropy_bins)
        instability_t = instability_metric(entropy_t, entropy_prev)

        entropy_history.append(entropy_t.detach().cpu().numpy())
        instability_history.append(instability_t.detach().cpu().numpy())

        entropy_prev = entropy_t

        if cfg.save_every > 0 and step % cfg.save_every == 0:
            snap_path = os.path.join(cfg.out_dir, "snapshots", f"snapshot_step_{step:04d}.png")
            save_snapshot_grid(x, pressure, control, chaos, snap_path)
            print(f"Saved snapshot: {snap_path}")

    entropy_history_np = np.stack(entropy_history, axis=0)       # [T+1, B]
    instability_history_np = np.stack(instability_history, axis=0)

    save_entropy_plot(
        entropy_history=entropy_history_np,
        pressure=pressure,
        control=control,
        chaos=chaos,
        out_path=os.path.join(cfg.out_dir, "plots", "entropy_curves.png"),
    )

    save_summary_csv(
        entropy_history=entropy_history_np,
        instability_history=instability_history_np,
        pressure=pressure,
        control=control,
        chaos=chaos,
        out_path=os.path.join(cfg.out_dir, "data", "summary.csv"),
    )

    save_phase_table(
        entropy_history=entropy_history_np,
        pressure=pressure,
        control=control,
        chaos=chaos,
        out_path=os.path.join(cfg.out_dir, "data", "phase_table.csv"),
    )

    np.save(os.path.join(cfg.out_dir, "data", "entropy_history.npy"), entropy_history_np)
    np.save(os.path.join(cfg.out_dir, "data", "instability_history.npy"), instability_history_np)
    np.save(os.path.join(cfg.out_dir, "data", "final_state.npy"), x.detach().cpu().numpy())

    final_snapshot_path = os.path.join(cfg.out_dir, "snapshots", "snapshot_final.png")
    save_snapshot_grid(x, pressure, control, chaos, final_snapshot_path)

    print("Simulation complete.")
    print(f"Outputs written to: {cfg.out_dir}")
    print(f"- plots/entropy_curves.png")
    print(f"- snapshots/snapshot_final.png")
    print(f"- data/summary.csv")
    print(f"- data/phase_table.csv")
    print(f"- data/entropy_history.npy")
    print(f"- data/instability_history.npy")
    print(f"- data/final_state.npy")


def main() -> None:
    cfg = parse_args()
    run_simulation(cfg)


if __name__ == "__main__":
    main()