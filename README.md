# PCC Dynamics: A Statistical Mechanical Framework for Non-Transitive Competition

**PCC** (Pressure, Control, Chaos) is an agent-based modeling framework designed to simulate and analyze **non-transitive competition** within complex systems.

Unlike hierarchical systems where a single strategy dominates, the PCC model demonstrates a stable "Limit Cycle" where three distinct metabolic strategies check and balance one another. This project provides the mathematical and computational tools to observe these dynamics in real-time.

---

## 📐 The Core Triangle

The system is governed by a non-transitive "Rock-Paper-Scissors" logic, defined by energy expenditure and information processing:

| Strategy | Primary Mechanism | Target | Vulnerability |
| --- | --- | --- | --- |
| **Pressure (P)** | **Massive Force.** High-energy, low-entropy directional vectors. | **Chaos** | **Control** |
| **Control (Co)** | **Informational Precision.** Reduction of local variance through modeling. | **Pressure** | **Chaos** |
| **Chaos (Ch)** | **Stochastic Disruption.** High Kolmogorov complexity/unpredictability. | **Control** | **Pressure** |

---

## 🔬 Scientific Foundations

### 1. Statistical Mechanics

We treat the system as a collection of agents whose macro-state is defined by **Entropy ()** and **Work ()**. The PCC model prevents the system from reaching "Heat Death" (a state of zero-change) by ensuring that no single strategy can ever achieve a Nash Equilibrium.

### 2. Information Theory

* **Control** acts as a filter, decreasing Shannon Entropy to gain a tactical advantage.
* **Chaos** introduces "Noise" that exceeds the processing capacity of the Control filter, rendering the model's predictions null.

### 3. Evolutionary Stable Strategies (ESS)

In biology, this is observed in "Side-blotched Lizard" mating cycles and microbial competitions. This repo simulates these interactions to find the **"Stability Zone"** where all three strategies coexist.

---

## 🛠 Features

* **`simulation.py`**: A differential equation solver (ODE) that tracks the population density of P, Co, and Ch over time.
* **`spatial_grid.py`**: A Cellular Automata simulation demonstrating the emergence of **Spiral Waves**—the mathematical fingerprint of non-transitive stability.
* **Limit Cycle Analysis**: Tools to measure if a system is heading toward extinction or healthy oscillation.

---

## 🚀 Getting Started

### Prerequisites

* Python 3.8+
* NumPy
* Matplotlib
* SciPy

### Installation

```bash
git clone https://github.com/HussainAther/pcc.git
cd pcc
pip install -r requirements.txt

```

### Running the Simulation

To see the population densities over time:

```bash
python simulation.py

```

---

## 📜 Limitations & Guardrails

For the PCC model to remain empirically valid, the following conditions must be met:

1. **Agency**: Agents must have the capacity to interact or change state based on encounters.
2. **Balance**: The interaction coefficients () must be tuned to prevent "monoculture collapse."
3. **Stochasticity**: Chaos must be mathematically random (stochastic), not just a complex pattern that a better Control agent could predict.

---

## 🤝 Contributing

This is an open-source project aimed at unifying behavioral strategies through math. We welcome contributions in:

* Multi-Agent Reinforcement Learning (MARL) implementations.
* Thermodynamic analysis of the PCC cycle.
* New visualization modules for spatial dynamics.

