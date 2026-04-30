# SAT Hardness Scaling Test

This folder contains the output of:

```bash
python3 analysis/sat_hardness_scaling_test.py --samples 60
```

The script tests whether MAAT defect-covariance geometry scales with runtime
hardness across random 3-SAT ensembles.

## Configuration

| Parameter | Value |
|-----------|-------|
| n values | 60,90,120,180 |
| alpha values | 3.8,4.1,4.26,4.4,4.8 |
| samples per (n, alpha) | 60 |
| random seed | 42 |

## Main Result

The hardest median-runtime cell is:

| n | alpha | median log10(runtime) | effective rank | log10 kappa |
|---:|------:|----------------------:|---------------:|------------:|
| 180 | 4.26 | -1.7386 | 1.8794 | 8.2867 |

The highest covariance-effective-rank cell is:

| n | alpha | median log10(runtime) | effective rank | log10 kappa |
|---:|------:|----------------------:|---------------:|------------:|
| 60 | 4.8 | -3.8800 | 2.3749 | 8.5828 |

## Correlations across (n, alpha) cells

| Quantity | Pearson with median log runtime | Spearman with median log runtime |
|----------|--------------------------------:|---------------------------------:|
| covariance effective rank | -0.7193 | -0.7444 |
| covariance log determinant | -0.6674 | -0.8316 |
| covariance log kappa | -0.8288 | -0.8722 |
| connectivity V | -0.8071 | -0.8286 |

## Interpretation

This is an ensemble-level scaling test. It does not prove NP-hardness and does
not classify individual formulas. It tests whether runtime-hardness ridges
coincide with changes in the geometry of primitive MAAT defects.

The first scaling run gives a nuanced result. Median runtime peaks in the
expected transition region near `alpha = 4.26--4.4` and grows strongly with
problem size. However, raw covariance conditioning does **not** increase
monotonically with hardness across all sizes. Across all `(n, alpha)` cells,
the correlations with median log-runtime are negative because the defect
covariance scale itself changes with `n`.

Within fixed `n`, connectivity `V` is the most stable ridge marker in this
moderate sample run. Raw `kappa(C)` should therefore be treated as an
ensemble-geometry diagnostic, not yet as a universal scalar hardness law.

## Within-size Spearman correlations

| n | V vs runtime | log kappa vs runtime | effective rank vs runtime |
|---:|-------------:|---------------------:|--------------------------:|
| 60 | 0.6000 | 0.5000 | -0.3000 |
| 90 | 0.7000 | -0.7000 | 0.5000 |
| 120 | 0.7000 | 0.4000 | -0.3000 |
| 180 | 0.6000 | -0.3000 | 0.5000 |

For publication-grade claims, increase the sample count and repeat with
several seeds.
