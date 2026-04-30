#!/usr/bin/env python3
"""Scaling test for MAAT covariance geometry in random 3-SAT.

This script tests the ensemble-level hypothesis:

    structural hardness is organized by defect-covariance geometry.

It generates random 3-SAT ensembles for several problem sizes n and clause
densities alpha, solves them with MiniSat22, computes MAAT fields, converts
them into primitive defects, and measures the covariance geometry of each
(n, alpha) ensemble.

The test is deliberately moderate by default so it can run on a laptop.
Increase --samples for publication-grade statistics.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import sys

BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from generator import generate_sat_instance
from maat_fields import (
    balance_field,
    connection_field_sparse,
    creativity_field,
    harmony_field,
    respect_field_usage_fairness,
)
from complexity import maat_score
from solver import solve_sat


OUTDIR = BASE_DIR / "results" / "scaling_test"
FIELDS = ["H", "B", "S", "V", "R"]
DEFECTS = [f"d_{name}" for name in FIELDS]
DEFAULT_N = [60, 90, 120, 180]
DEFAULT_ALPHA = [3.8, 4.1, 4.26, 4.4, 4.8]
EPS = 1e-12


def parse_float_list(text: str) -> list[float]:
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def parse_int_list(text: str) -> list[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def run_ensemble(
    n_vars: int,
    alpha: float,
    samples: int,
    seed: int,
) -> pd.DataFrame:
    rng = random.Random(seed)
    n_clauses = int(round(alpha * n_vars))
    rows = []

    for sample_id in range(samples):
        clauses = generate_sat_instance(n_vars, n_clauses, rng=rng)

        h = harmony_field(clauses, n_vars)
        b = balance_field(clauses)
        s = creativity_field(clauses, n_vars)
        v = connection_field_sparse(clauses, n_vars)
        r = respect_field_usage_fairness(clauses, n_vars)
        c_hat = maat_score(h, b, s, v, r)

        sat, runtime = solve_sat(clauses)
        row = {
            "n": n_vars,
            "alpha": alpha,
            "sample_id": sample_id,
            "clauses": n_clauses,
            "runtime": runtime,
            "log_runtime": float(np.log10(max(runtime, EPS))),
            "sat": bool(sat),
            "H": h,
            "B": b,
            "S": s,
            "V": v,
            "R": r,
            "C_hat": c_hat,
        }
        for field in FIELDS:
            row[f"d_{field}"] = 1.0 - row[field]
        rows.append(row)

    return pd.DataFrame(rows)


def covariance_summary(group: pd.DataFrame) -> dict[str, float | int | bool]:
    x = group[DEFECTS].to_numpy(float)
    c = np.cov(x, rowvar=False)
    eig = np.linalg.eigvalsh(c)
    eig_safe = np.maximum(eig, EPS)
    p = eig_safe / eig_safe.sum()

    kappa = float(eig_safe.max() / eig_safe.min())
    trace = float(np.trace(c))
    log_det = float(np.sum(np.log10(eig_safe)))
    spectral_entropy = float(-(p * np.log(p)).sum())
    effective_rank = float(np.exp(spectral_entropy))
    rank = int(np.linalg.matrix_rank(c, tol=1e-10))
    singular = bool(eig.min() < 1e-10)

    return {
        "count": int(len(group)),
        "sat_prob": float(group["sat"].mean()),
        "mean_runtime": float(group["runtime"].mean()),
        "median_runtime": float(group["runtime"].median()),
        "mean_log_runtime": float(group["log_runtime"].mean()),
        "median_log_runtime": float(group["log_runtime"].median()),
        "mean_C_hat": float(group["C_hat"].mean()),
        "mean_H": float(group["H"].mean()),
        "mean_B": float(group["B"].mean()),
        "mean_S": float(group["S"].mean()),
        "mean_V": float(group["V"].mean()),
        "mean_R": float(group["R"].mean()),
        "cov_trace": trace,
        "cov_log_det": log_det,
        "cov_kappa": kappa,
        "cov_log_kappa": float(np.log10(kappa + 1.0)),
        "cov_rank": rank,
        "cov_singular": singular,
        "cov_lambda_min": float(eig.min()),
        "cov_lambda_max": float(eig.max()),
        "cov_effective_rank": effective_rank,
    }


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (n_vars, alpha), group in df.groupby(["n", "alpha"]):
        row = {"n": int(n_vars), "alpha": float(alpha)}
        row.update(covariance_summary(group))
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["n", "alpha"])


def pivot_plot(
    summary: pd.DataFrame,
    value: str,
    title: str,
    filename: str,
    colorbar_label: str,
) -> None:
    pivot = summary.pivot(index="n", columns="alpha", values=value)
    plt.figure(figsize=(8.4, 5.6))
    im = plt.imshow(pivot.values, aspect="auto", origin="lower")
    plt.colorbar(im, label=colorbar_label)
    plt.xticks(np.arange(len(pivot.columns)), [str(c) for c in pivot.columns])
    plt.yticks(np.arange(len(pivot.index)), [str(i) for i in pivot.index])
    plt.xlabel("alpha")
    plt.ylabel("n variables")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(OUTDIR / filename, dpi=240)
    plt.close()


def scatter_plot(summary: pd.DataFrame) -> None:
    plt.figure(figsize=(7.5, 5.8))
    scatter = plt.scatter(
        summary["cov_effective_rank"],
        summary["median_log_runtime"],
        c=summary["n"],
        s=90,
        alpha=0.85,
    )
    for _, row in summary.iterrows():
        plt.annotate(
            f"{row['alpha']:.2g}",
            (row["cov_effective_rank"], row["median_log_runtime"]),
            textcoords="offset points",
            xytext=(5, 4),
            fontsize=8,
        )
    plt.colorbar(scatter, label="n")
    plt.xlabel("Defect covariance effective rank")
    plt.ylabel("Median log10(runtime)")
    plt.title("Runtime hardness vs covariance effective rank")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTDIR / "runtime_vs_effective_rank.png", dpi=240)
    plt.close()

    plt.figure(figsize=(7.5, 5.8))
    scatter = plt.scatter(
        summary["cov_log_det"],
        summary["median_log_runtime"],
        c=summary["n"],
        s=90,
        alpha=0.85,
    )
    for _, row in summary.iterrows():
        plt.annotate(
            f"{row['alpha']:.2g}",
            (row["cov_log_det"], row["median_log_runtime"]),
            textcoords="offset points",
            xytext=(5, 4),
            fontsize=8,
        )
    plt.colorbar(scatter, label="n")
    plt.xlabel("Defect covariance log10 determinant")
    plt.ylabel("Median log10(runtime)")
    plt.title("Runtime hardness vs covariance volume")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTDIR / "runtime_vs_log_det.png", dpi=240)
    plt.close()


def ridge_plot(summary: pd.DataFrame) -> None:
    plt.figure(figsize=(8.4, 5.6))
    for n_vars, group in summary.groupby("n"):
        group = group.sort_values("alpha")
        plt.plot(
            group["alpha"],
            group["median_log_runtime"],
            marker="o",
            linewidth=1.8,
            label=f"n={n_vars}",
        )
    plt.xlabel("alpha")
    plt.ylabel("Median log10(runtime)")
    plt.title("Runtime ridge across problem sizes")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTDIR / "runtime_ridge_by_n.png", dpi=240)
    plt.close()


def write_readme(args, summary: pd.DataFrame, correlations: dict, within_n: dict) -> None:
    best_runtime = summary.sort_values("median_log_runtime", ascending=False).iloc[0]
    best_rank = summary.sort_values("cov_effective_rank", ascending=False).iloc[0]

    text = f"""# SAT Hardness Scaling Test

This folder contains the output of:

```bash
python3 analysis/sat_hardness_scaling_test.py --samples {args.samples}
```

The script tests whether MAAT defect-covariance geometry scales with runtime
hardness across random 3-SAT ensembles.

## Configuration

| Parameter | Value |
|-----------|-------|
| n values | {args.n_values} |
| alpha values | {args.alpha_values} |
| samples per (n, alpha) | {args.samples} |
| random seed | {args.seed} |

## Main Result

The hardest median-runtime cell is:

| n | alpha | median log10(runtime) | effective rank | log10 kappa |
|---:|------:|----------------------:|---------------:|------------:|
| {int(best_runtime['n'])} | {best_runtime['alpha']:.3g} | {best_runtime['median_log_runtime']:.4f} | {best_runtime['cov_effective_rank']:.4f} | {best_runtime['cov_log_kappa']:.4f} |

The highest covariance-effective-rank cell is:

| n | alpha | median log10(runtime) | effective rank | log10 kappa |
|---:|------:|----------------------:|---------------:|------------:|
| {int(best_rank['n'])} | {best_rank['alpha']:.3g} | {best_rank['median_log_runtime']:.4f} | {best_rank['cov_effective_rank']:.4f} | {best_rank['cov_log_kappa']:.4f} |

## Correlations across (n, alpha) cells

| Quantity | Pearson with median log runtime | Spearman with median log runtime |
|----------|--------------------------------:|---------------------------------:|
| covariance effective rank | {correlations['pearson_effective_rank']:.4f} | {correlations['spearman_effective_rank']:.4f} |
| covariance log determinant | {correlations['pearson_log_det']:.4f} | {correlations['spearman_log_det']:.4f} |
| covariance log kappa | {correlations['pearson_log_kappa']:.4f} | {correlations['spearman_log_kappa']:.4f} |
| connectivity V | {correlations['pearson_V']:.4f} | {correlations['spearman_V']:.4f} |

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
{within_n['markdown_rows']}

For publication-grade claims, increase the sample count and repeat with
several seeds.
"""
    (OUTDIR / "README.md").write_text(text)


def run(args) -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)

    n_values = parse_int_list(args.n_values)
    alpha_values = parse_float_list(args.alpha_values)

    all_frames = []
    for n_index, n_vars in enumerate(n_values):
        for alpha_index, alpha in enumerate(alpha_values):
            seed = args.seed + 1000 * n_index + alpha_index
            print(f"running n={n_vars}, alpha={alpha}, samples={args.samples}, seed={seed}")
            all_frames.append(run_ensemble(n_vars, alpha, args.samples, seed))

    df = pd.concat(all_frames, ignore_index=True)
    summary = summarize(df)

    raw_path = OUTDIR / "sat_scaling_raw.csv"
    summary_path = OUTDIR / "sat_scaling_summary.csv"
    df.to_csv(raw_path, index=False)
    summary.to_csv(summary_path, index=False)

    pivot_plot(
        summary,
        "median_log_runtime",
        "Median log10(runtime) over (n, alpha)",
        "heatmap_median_log_runtime.png",
        "median log10(runtime)",
    )
    pivot_plot(
        summary,
        "cov_effective_rank",
        "Defect covariance effective rank over (n, alpha)",
        "heatmap_cov_effective_rank.png",
        "effective rank",
    )
    pivot_plot(
        summary,
        "cov_log_kappa",
        "Defect covariance log conditioning over (n, alpha)",
        "heatmap_cov_log_kappa.png",
        "log10(1 + kappa)",
    )
    pivot_plot(
        summary,
        "sat_prob",
        "SAT probability over (n, alpha)",
        "heatmap_sat_probability.png",
        "P(SAT)",
    )
    scatter_plot(summary)
    ridge_plot(summary)

    pearson = summary[
        [
            "median_log_runtime",
            "cov_effective_rank",
            "cov_log_det",
            "cov_log_kappa",
            "mean_V",
        ]
    ].corr(numeric_only=True)["median_log_runtime"]
    spearman = summary[
        [
            "median_log_runtime",
            "cov_effective_rank",
            "cov_log_det",
            "cov_log_kappa",
            "mean_V",
        ]
    ].corr(method="spearman", numeric_only=True)["median_log_runtime"]

    correlations = {
        "pearson_effective_rank": float(pearson["cov_effective_rank"]),
        "spearman_effective_rank": float(spearman["cov_effective_rank"]),
        "pearson_log_det": float(pearson["cov_log_det"]),
        "spearman_log_det": float(spearman["cov_log_det"]),
        "pearson_log_kappa": float(pearson["cov_log_kappa"]),
        "spearman_log_kappa": float(spearman["cov_log_kappa"]),
        "pearson_V": float(pearson["mean_V"]),
        "spearman_V": float(spearman["mean_V"]),
    }

    within_n = {}
    rows = []
    for n_vars, group in summary.groupby("n"):
        corr = group[
            [
                "median_log_runtime",
                "mean_V",
                "cov_log_kappa",
                "cov_effective_rank",
            ]
        ].corr(method="spearman", numeric_only=True)["median_log_runtime"]
        item = {
            "spearman_V": float(corr["mean_V"]),
            "spearman_log_kappa": float(corr["cov_log_kappa"]),
            "spearman_effective_rank": float(corr["cov_effective_rank"]),
        }
        within_n[str(int(n_vars))] = item
        rows.append(
            f"| {int(n_vars)} | {item['spearman_V']:.4f} | "
            f"{item['spearman_log_kappa']:.4f} | "
            f"{item['spearman_effective_rank']:.4f} |"
        )
    within_n["markdown_rows"] = "\n".join(rows)

    result = {
        "n_values": n_values,
        "alpha_values": alpha_values,
        "samples_per_cell": args.samples,
        "seed": args.seed,
        "total_instances": int(len(df)),
        "correlations": correlations,
        "within_n_spearman": {
            key: value for key, value in within_n.items() if key != "markdown_rows"
        },
        "summary_csv": str(summary_path),
        "raw_csv": str(raw_path),
    }
    (OUTDIR / "sat_scaling_results.json").write_text(json.dumps(result, indent=2))
    write_readme(args, summary, correlations, within_n)

    print("\n=== Scaling Summary ===")
    print(summary.to_string(index=False))
    print("\n=== Correlations with median log runtime ===")
    print(json.dumps(correlations, indent=2))
    print(f"\nSaved scaling outputs to: {OUTDIR}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-values", default=",".join(map(str, DEFAULT_N)))
    parser.add_argument("--alpha-values", default=",".join(map(str, DEFAULT_ALPHA)))
    parser.add_argument("--samples", type=int, default=60)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
