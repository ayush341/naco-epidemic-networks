# SIR Epidemic Spreading on Network Topologies

**Natural Computing 2025/26 — Radboud University**  
**Ayush Kumar | Student No. 1170210 | Group 23**

---

## Project Overview

This project investigates how contact network topology shapes SIR epidemic dynamics by comparing three canonical network types — Erdos-Renyi random, Watts-Strogatz small-world, and Barabasi-Albert scale-free — under matched average degree and systematic variation of transmission rate.

**Main findings:**
- Small-world networks suppress epidemics near the threshold due to high clustering
- Hub seeding on scale-free networks causes 4x larger outbreaks than random seeding
- All network types converge to similar outbreak sizes at high R0

---

## Requirements

Python 3.8 or higher. Install dependencies:

```bash
pip install networkx numpy matplotlib
```

Or using the requirements file:

```bash
pip install -r requirements.txt
```

---

## How to Run

### Option 1: Run directly (recommended)

```bash
python sir_simulation.py
```

This runs all 4 experiments and saves 5 figures to the current directory.

### Option 2: Google Colab

1. Upload `sir_simulation.py` to Colab
2. Run: `exec(open('sir_simulation.py').read())`
3. Figures are automatically downloaded at the end

---

## Experiments

| Experiment | Description | Output figure |
|---|---|---|
| 1 | Beta sweep across 3 network types (R0: 0.3–4.8) | `exp1_beta_sweep.png`, `exp1_curves.png` |
| 2 | Seed node identity on scale-free networks | `exp2_seed_nodes.png` |
| 3 | Rewiring probability sweep in small-world networks | `exp3_rewiring.png` |
| 4 | Hub vs random seeding across full R0 range | `exp4_hub_vs_random.png` |

---

## Parameters

| Parameter | Value | Description |
|---|---|---|
| N | 1000 | Number of nodes per network |
| avg_k | 6 | Average degree (matched across all networks) |
| mu | 0.1 | Recovery probability per timestep |
| T_MAX | 200 | Maximum simulation timesteps |
| N_SEEDS | 30 | Random seeds per condition |
| beta range | 0.005 – 0.08 | R0 range: 0.3 – 4.8 |

Theoretical epidemic threshold under homogeneous mixing: `beta_c = mu / avg_k = 0.1 / 6 = 0.0167` (R0 = 1)

---

## Network Setup

| Network | Model | Key parameter | Clustering |
|---|---|---|---|
| Random | Erdos-Renyi G(n,p) | p = 0.006 | ~0.006 (low) |
| Small-World | Watts-Strogatz | k=6, p_rewire=0.1 | ~0.44 (high) |
| Scale-Free | Barabasi-Albert | m=3 edges per new node | ~0.03 (low) |

---

## Sample Output

Running the code produces epidemic curves like this:

- **Near R0=1:** Small-world stays near zero; random and scale-free show visible outbreaks
- **At R0=4.8:** All three networks produce similarly large outbreaks (76–83%)
- **Hub seeding at R0=1.5:** 46% outbreak vs 12% for random seeding

---

## File Structure

```
naco-epidemic-networks/
├── sir_simulation.py     # Main simulation code (all 4 experiments)
├── requirements.txt      # Python dependencies
└── README.md             # This file
```

---

## License

This code was developed for the Natural Computing course at Radboud University, 2025/26.
