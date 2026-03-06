# SAT Hardness Atlas

Runtime Landscapes and Ridge Structure Driven by Connectivity (V)

This repository contains the code and data used in the paper:

"A SAT Hardness Atlas: Runtime Landscapes and Ridge Structure Driven by
Connectivity (V)" Christof Krieg -- MAAT Research

The project introduces an empirical hardness mapping pipeline for random
SAT instances. It constructs a Hardness Atlas that visualizes solver
runtime as a function of:

-   clause density α
-   structural connectivity V

The experiments reveal a ridge-like region of maximal runtime in the
hardness landscape.

------------------------------------------------------------------------

# Overview

Pipeline steps:

1.  Generate random 3‑SAT instances
2.  Solve them using MiniSAT
3.  Compute structural instance features:
    -   H -- Harmony
    -   B -- Balance
    -   S -- Creativity
    -   V -- Connectivity
    -   R -- Respect

Derived projection:

Ĉ = (H + B + S + V + R) / 5

Outputs:

-   SAT phase transition plots
-   Hardness Atlas
-   Hardness ridge detection
-   Phase surface plots
-   Runtime predictors

------------------------------------------------------------------------

# Repository Structure

maat_cosmos_engine/

engine.py\
benchmark.py\
generator.py\
solver.py\
maat_fields.py\
complexity.py

analysis/

hardness_atlas.py\
ridge_detection.py\
phase_surface.py\
hardness_predictor_v2.py\
hardness_field_equation.py

results/

CSV outputs\
atlas grids\
ridge tables

figures/

phase transition plots\
atlas visualizations\
ridge plots\
3D phase surfaces

------------------------------------------------------------------------

# Requirements

Python 3.10+

Packages:

-   numpy
-   pandas
-   matplotlib
-   scikit-learn
-   statsmodels

Install:

pip install numpy pandas matplotlib scikit-learn statsmodels

------------------------------------------------------------------------

# Running the Benchmark

python3 engine.py

This generates:

results/maat_cosmos_full_results.csv

------------------------------------------------------------------------

# Generating the Hardness Atlas

python3 analysis/hardness_atlas.py

Outputs:

results/hardness_atlas_grid.csv\
results/hardness_atlas_alpha_vs_V.png\
results/sat_prob_atlas_alpha_vs_V.png

------------------------------------------------------------------------

# Detecting the Hardness Ridge

python3 analysis/ridge_detection.py --y V

Outputs:

results/hardness_ridge_V.csv\
results/hardness_ridge_V.png

------------------------------------------------------------------------

# Phase Surface

python3 analysis/phase_surface.py --y V --mode median

Outputs:

results/sat_phase_surface_V_median.png\
results/sat_phase_contour_V_median.png

------------------------------------------------------------------------

# Runtime Prediction

python3 analysis/hardness_predictor_v2.py

------------------------------------------------------------------------

# Experimental Setup

Variables: n = 30\
Clause type: 3‑SAT\
Instances: 2000\
Solver: MiniSAT\
Runtime metric: log10(runtime)

Clause density range:

α ∈ \[3.0, 5.0\]

------------------------------------------------------------------------

# Reproducing the Paper

python3 engine.py python3 analysis/hardness_atlas.py python3
analysis/ridge_detection.py --y V python3 analysis/phase_surface.py --y
V --mode median python3 analysis/hardness_predictor_v2.py

------------------------------------------------------------------------

# Key Result

Constraint connectivity **V** acts as a primary organizing variable for
SAT hardness in the sampled regime.

A ridge-like region of maximal runtime appears in the hardness landscape
over (α, V).

------------------------------------------------------------------------

# License

MIT License

------------------------------------------------------------------------

# Citation

Christof Krieg. A SAT Hardness Atlas: Runtime Landscapes and Ridge
Structure Driven by Connectivity (V)

------------------------------------------------------------------------

https://maat-research.com
