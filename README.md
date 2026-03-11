# SAT Hardness Atlas

**Runtime Landscapes and Ridge Structure Driven by Connectivity (V)**

Christof Krieg — MAAT Research

---

## Overview

This repository provides the code and data for the paper:

> *A SAT Hardness Atlas: Runtime Landscapes and Ridge Structure Driven by Connectivity (V)*

The project introduces an empirical hardness mapping pipeline for random 3-SAT instances. It constructs a **Hardness Atlas** that visualizes solver runtime as a function of clause density α and structural connectivity V, revealing a ridge-like region of maximal runtime in the hardness landscape.

---

## Pipeline

1. Generate random 3-SAT instances
2. Solve with MiniSAT
3. Compute structural instance features (MAAT fields):

| Field | Symbol | Role |
|-------|--------|------|
| Harmony | H | Clause coherence |
| Balance | B | Literal balance |
| Creativity | S | Structural diversity |
| Connectivity | V | Constraint connectivity |
| Respect | R | Constraint satisfaction ratio |

4. Compute derived projection: `Ĉ = (H + B + S + V + R) / 5`
5. Generate outputs: phase transition plots, Hardness Atlas, ridge detection, runtime predictors

---

## Repository Structure

```
maat_cosmos_engine/
├── engine.py
├── benchmark.py
├── generator.py
├── solver.py
├── maat_fields.py
└── complexity.py

analysis/
├── hardness_atlas.py
├── ridge_detection.py
├── phase_surface.py
├── hardness_predictor_v2.py
└── hardness_field_equation.py

results/          # CSV outputs, atlas grids, ridge tables
figures/          # Plots and visualizations
```

---

## Requirements

Python 3.10+

```bash
pip install numpy pandas matplotlib scikit-learn statsmodels
```

---

## Reproducing the Paper

Run the full pipeline in order:

```bash
python3 engine.py
python3 analysis/hardness_atlas.py
python3 analysis/ridge_detection.py --y V
python3 analysis/phase_surface.py --y V --mode median
python3 analysis/hardness_predictor_v2.py
```

### Output files

| Script | Output |
|--------|--------|
| `engine.py` | `results/maat_cosmos_full_results.csv` |
| `hardness_atlas.py` | `results/hardness_atlas_grid.csv`, atlas plots |
| `ridge_detection.py` | `results/hardness_ridge_V.csv`, ridge plot |
| `phase_surface.py` | 3D phase surface and contour plots |
| `hardness_predictor_v2.py` | Runtime prediction model |

---

## Experimental Setup

| Parameter | Value |
|-----------|-------|
| Variables per instance | n = 30 |
| Clause type | 3-SAT |
| Instances | 2000 |
| Solver | MiniSAT |
| Runtime metric | log₁₀(runtime) |
| Clause density range | α ∈ [3.0, 5.0] |

---

## Key Result

Constraint connectivity **V** acts as a primary organizing variable for SAT hardness. A ridge-like region of maximal runtime appears in the hardness landscape over (α, V), consistent with the MAAT prediction that V governs structural phase transitions.

---

## License

MIT License

---

## Citation

```
Christof Krieg. A SAT Hardness Atlas: Runtime Landscapes and Ridge
Structure Driven by Connectivity (V). MAAT Research.
https://maat-research.com
```
