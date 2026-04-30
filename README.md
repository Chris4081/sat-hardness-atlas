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
| Respect | R | Variable-usage fairness / robustness proxy |

4. Compute derived projection: `Ĉ = H · B · S · V · 1/(1+R) + ε`
5. Generate outputs: phase transition plots, Hardness Atlas, ridge detection, runtime predictors

---

## Repository Structure

```
./
├── engine.py
├── benchmark.py
├── generator.py
├── solver.py
├── maat_fields.py
├── complexity.py
├── config.py
└── requirements.txt

analysis/
├── hardness_atlas.py
├── ridge_detection.py
├── phase_surface.py
├── hardness_predictor_v2.py
├── hardness_field_equation.py
├── sat_hardness_covariance_classifier.py
└── sat_hardness_scaling_test.py

results/                         # CSV outputs, atlas grids, plots
results/covariance_classifier/   # local covariance-classifier outputs
results/scaling_test/            # multi-size scaling benchmark outputs
```

---

## Requirements

Python 3.10+

```bash
pip install numpy pandas matplotlib scikit-learn statsmodels python-sat
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
python3 analysis/sat_hardness_covariance_classifier.py
python3 analysis/sat_hardness_scaling_test.py --samples 60
```

### Output files

| Script | Output |
|--------|--------|
| `engine.py` | `results/maat_cosmos_full_results.csv` |
| `hardness_atlas.py` | `results/hardness_atlas_grid.csv`, atlas plots |
| `ridge_detection.py` | `results/hardness_ridge_V.csv`, ridge plot |
| `phase_surface.py` | 3D phase surface and contour plots |
| `hardness_predictor_v2.py` | Runtime prediction model |
| `sat_hardness_covariance_classifier.py` | `results/covariance_classifier/` |
| `sat_hardness_scaling_test.py` | `results/scaling_test/` |

---

## Experimental Setup

| Parameter | Value |
|-----------|-------|
| Variables per instance | n = 120 |
| Clause type | 3-SAT |
| Instances | 2000 |
| Solver | MiniSAT |
| Runtime metric | log₁₀(runtime) |
| Clause density range | α ∈ [3.0, 5.0] |
| Random seed | 42 |

The scaling benchmark uses a smaller default grid:

| Parameter | Value |
|-----------|-------|
| n values | 60, 90, 120, 180 |
| α values | 3.8, 4.1, 4.26, 4.4, 4.8 |
| Samples per (n, α) | 60 |

---

## Key Result

Constraint connectivity **V** acts as a primary organizing variable for SAT hardness. A ridge-like region of maximal runtime appears in the hardness landscape over (α, V), consistent with the MAAT prediction that V governs structural phase transitions.

The covariance-classifier extension tests a sharper hypothesis:

> SAT runtime hardness is partially organized by the local covariance geometry of primitive MAAT defects.

The target is not SAT vs UNSAT. Instead, an instance is labelled as structurally hard if its runtime lies in the top 20% within its clause-density α group. This avoids confusing logical satisfiability with computational hardness.

Current benchmark status:

| Test | Result |
|------|--------|
| Ensemble covariance conditioning | Strong separation between SAT/UNSAT, stable regimes, and frozen/degenerate regimes |
| Local instance-level hard-runtime classifier | Weak positive signal only; balanced accuracy ≈ 0.50 in the current 2000-instance dataset |
| Runtime regression | Dominated by α and V; local covariance features add modest information |
| Best local covariance indicators | Effective rank and log-determinant are more informative than local condition number alone |

Interpretation: covariance geometry is a strong ensemble-level diagnostic, but the current local k-nearest-neighbour classifier is not yet sufficient as a standalone predictor of individual hard instances. Larger scaling experiments over multiple problem sizes are the next required test.

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
