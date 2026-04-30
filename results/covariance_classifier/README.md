# SAT Hardness Covariance Classifier

This folder contains the output of:

```bash
python3 analysis/sat_hardness_covariance_classifier.py
```

The script tests whether local covariance geometry of primitive MAAT defects
adds predictive information for SAT runtime hardness.

## Target Definition

The classifier does **not** predict SAT vs UNSAT.

An instance is labelled as `hard_runtime = 1` if its runtime lies in the top
20% within its clause-density `alpha` group.

This avoids confusing logical satisfiability with computational hardness.

## Main Outputs

| File | Meaning |
|------|---------|
| `sat_hardness_covariance_classifier_results.json` | Metrics, correlations, feature importances |
| `sat_hardness_with_local_covariance.csv` | Original SAT data plus local covariance features |
| `classifier_model_comparison.png` | Hard-runtime classifier comparison |
| `runtime_regression_r2.png` | Runtime regression baseline comparison |
| `local_kappa_vs_runtime.png` | Runtime vs local covariance conditioning |
| `local_kappa_by_hardness_label.png` | Local conditioning split by hard-runtime label |
| `full_model_feature_importances.png` | Full model feature importances |

## Current Result

The present 2000-instance benchmark shows:

| Model | ROC-AUC | Balanced accuracy |
|-------|--------:|------------------:|
| `alpha_only` | 0.431 | 0.450 |
| `alpha_c_hat` | 0.475 | 0.471 |
| `alpha_fields` | 0.466 | 0.479 |
| `alpha_fields_local_covariance` | 0.474 | 0.504 |

The local covariance features are informative in the full-model feature
importance ranking, but they do not yet produce a strong standalone
instance-level classifier.

The strongest runtime correlations in the current dataset are:

| Feature | Pearson correlation with `log_runtime` |
|---------|---------------------------------------:|
| `V` | 0.714 |
| `alpha` | 0.708 |
| `R` | 0.559 |
| `local_effective_rank` | 0.452 |
| `local_log_det` | 0.432 |
| `C_hat` | 0.397 |

## Interpretation

The current evidence supports a cautious conclusion:

> Defect covariance geometry is a strong ensemble-level diagnostic and contains
> local runtime information, but local k-nearest-neighbour covariance features
> are not yet sufficient as a standalone predictor of individual hard
> instances.

The next required test is a scaling benchmark over multiple problem sizes
(`n = 60, 90, 120, 180, ...`) with repeated seeds.
