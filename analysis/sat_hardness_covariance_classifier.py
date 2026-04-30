#!/usr/bin/env python3
"""SAT hardness classifier with local MAAT defect-covariance features.

The target is runtime hardness, not satisfiability. An instance is labelled
"hard" if its runtime belongs to the top 20 percent within its alpha group.
This tests whether local covariance geometry of primitive MAAT defects adds
predictive information beyond clause density and raw MAAT fields.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    mean_absolute_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


BASE_DIR = Path(__file__).resolve().parents[1]
RESULTS_PATH = BASE_DIR / "results" / "maat_cosmos_full_results.csv"
OUTDIR = BASE_DIR / "results" / "covariance_classifier"

FIELDS = ["H", "B", "S", "V", "R"]
DEFECTS = [f"d_{name}" for name in FIELDS]
RANDOM_STATE = 42
LOCAL_K = 40
HARD_QUANTILE = 0.80


def load_results(path: Path = RESULTS_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = [c for c in ["alpha", "runtime", "C_hat", *FIELDS] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df.dropna(subset=["alpha", "runtime", "C_hat", *FIELDS]).copy()
    for field in FIELDS:
        df[field] = df[field].clip(0.0, 1.0)
        df[f"d_{field}"] = 1.0 - df[field]

    eps = 1e-12
    df["runtime_safe"] = df["runtime"].clip(lower=eps)
    df["log_runtime"] = np.log10(df["runtime_safe"])

    thresholds = df.groupby("alpha")["runtime"].transform(
        lambda x: x.quantile(HARD_QUANTILE)
    )
    df["hard_runtime"] = (df["runtime"] >= thresholds).astype(int)
    return df


def covariance_features(df: pd.DataFrame, k: int = LOCAL_K) -> pd.DataFrame:
    """Attach local covariance features computed from k nearest neighbours."""
    k = max(len(DEFECTS) + 3, min(k, len(df)))

    neighbour_space = df[["alpha", *DEFECTS]].to_numpy(float)
    scaled = StandardScaler().fit_transform(neighbour_space)
    neigh = NearestNeighbors(n_neighbors=k)
    neigh.fit(scaled)
    _, indices = neigh.kneighbors(scaled)

    defect_values = df[DEFECTS].to_numpy(float)
    rows: list[dict[str, float | int]] = []
    eps = 1e-12

    for idx in indices:
        local = defect_values[idx]
        c = np.cov(local, rowvar=False)
        eig = np.linalg.eigvalsh(c)
        eig_safe = np.maximum(eig, eps)

        kappa = float(eig_safe.max() / eig_safe.min())
        trace = float(np.trace(c))
        rank = int(np.linalg.matrix_rank(c, tol=1e-10))
        singular = int(eig.min() < 1e-10)
        log_det = float(np.sum(np.log10(eig_safe)))

        p = eig_safe / eig_safe.sum()
        spectral_entropy = float(-(p * np.log(p)).sum())
        effective_rank = float(np.exp(spectral_entropy))

        rows.append(
            {
                "local_kappa": kappa,
                "local_log_kappa": float(np.log10(kappa + 1.0)),
                "local_trace": trace,
                "local_log_trace": float(np.log10(trace + eps)),
                "local_rank": rank,
                "local_singular": singular,
                "local_lambda_min": float(eig.min()),
                "local_lambda_max": float(eig.max()),
                "local_log_det": log_det,
                "local_effective_rank": effective_rank,
            }
        )

    return pd.concat([df.reset_index(drop=True), pd.DataFrame(rows)], axis=1)


def classifier_metrics(y_true, prob, pred) -> dict[str, float]:
    return {
        "roc_auc": float(roc_auc_score(y_true, prob)),
        "accuracy": float(accuracy_score(y_true, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, pred)),
        "f1": float(f1_score(y_true, pred)),
    }


def fit_classifiers(df: pd.DataFrame) -> tuple[dict, dict]:
    feature_sets = {
        "alpha_only": ["alpha"],
        "alpha_c_hat": ["alpha", "C_hat"],
        "alpha_fields": ["alpha", *FIELDS],
        "alpha_fields_local_covariance": [
            "alpha",
            *FIELDS,
            "local_log_kappa",
            "local_log_trace",
            "local_rank",
            "local_singular",
            "local_lambda_min",
            "local_lambda_max",
            "local_log_det",
            "local_effective_rank",
        ],
    }

    train_idx, test_idx = train_test_split(
        df.index,
        test_size=0.30,
        random_state=RANDOM_STATE,
        stratify=df["hard_runtime"],
    )

    metrics: dict[str, dict] = {}
    importances: dict[str, dict[str, float]] = {}

    for name, features in feature_sets.items():
        model = RandomForestClassifier(
            n_estimators=500,
            random_state=RANDOM_STATE,
            class_weight="balanced",
            min_samples_leaf=5,
        )
        model.fit(df.loc[train_idx, features], df.loc[train_idx, "hard_runtime"])

        prob = model.predict_proba(df.loc[test_idx, features])[:, 1]
        pred = (prob >= 0.5).astype(int)
        metrics[name] = classifier_metrics(df.loc[test_idx, "hard_runtime"], prob, pred)
        importances[name] = {
            feature: float(value)
            for feature, value in zip(features, model.feature_importances_)
        }

    return metrics, importances


def fit_regressors(df: pd.DataFrame) -> dict[str, dict[str, float]]:
    feature_sets = {
        "alpha_only": ["alpha"],
        "alpha_c_hat": ["alpha", "C_hat"],
        "alpha_fields": ["alpha", *FIELDS],
        "alpha_fields_local_covariance": [
            "alpha",
            *FIELDS,
            "local_log_kappa",
            "local_log_trace",
            "local_rank",
            "local_singular",
            "local_lambda_min",
            "local_lambda_max",
            "local_log_det",
            "local_effective_rank",
        ],
    }

    train_idx, test_idx = train_test_split(
        df.index,
        test_size=0.30,
        random_state=RANDOM_STATE,
    )

    metrics: dict[str, dict[str, float]] = {}
    for name, features in feature_sets.items():
        model = RandomForestRegressor(
            n_estimators=500,
            random_state=RANDOM_STATE,
            min_samples_leaf=5,
        )
        model.fit(df.loc[train_idx, features], df.loc[train_idx, "log_runtime"])
        pred = model.predict(df.loc[test_idx, features])
        y_true = df.loc[test_idx, "log_runtime"]
        metrics[name] = {
            "r2": float(r2_score(y_true, pred)),
            "mae": float(mean_absolute_error(y_true, pred)),
        }
    return metrics


def correlation_summary(df: pd.DataFrame) -> dict[str, dict[str, float]]:
    cols = [
        "alpha",
        "C_hat",
        *FIELDS,
        "local_log_kappa",
        "local_log_trace",
        "local_log_det",
        "local_effective_rank",
        "local_lambda_max",
    ]
    pearson = df[cols + ["log_runtime"]].corr(numeric_only=True)["log_runtime"]
    spearman = df[cols + ["log_runtime"]].corr(
        method="spearman", numeric_only=True
    )["log_runtime"]
    return {
        "pearson_log_runtime": {
            key: float(value)
            for key, value in pearson.drop("log_runtime").sort_values(ascending=False).items()
        },
        "spearman_log_runtime": {
            key: float(value)
            for key, value in spearman.drop("log_runtime").sort_values(ascending=False).items()
        },
    }


def plot_classifier_comparison(metrics: dict) -> None:
    names = list(metrics)
    auc = [metrics[name]["roc_auc"] for name in names]
    bal = [metrics[name]["balanced_accuracy"] for name in names]

    x = np.arange(len(names))
    width = 0.38
    plt.figure(figsize=(10, 5.5))
    plt.bar(x - width / 2, auc, width, label="ROC-AUC")
    plt.bar(x + width / 2, bal, width, label="Balanced accuracy")
    plt.xticks(x, names, rotation=25, ha="right")
    plt.ylim(0.0, 1.05)
    plt.ylabel("Score")
    plt.title("Hard-runtime classifier: covariance features vs baselines")
    plt.grid(axis="y", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTDIR / "classifier_model_comparison.png", dpi=240)
    plt.close()


def plot_regression_comparison(metrics: dict) -> None:
    names = list(metrics)
    r2 = [metrics[name]["r2"] for name in names]

    plt.figure(figsize=(9, 5.2))
    plt.bar(names, r2)
    plt.xticks(rotation=25, ha="right")
    plt.ylabel("R2 on log10(runtime)")
    plt.title("Runtime regression: covariance features vs baselines")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTDIR / "runtime_regression_r2.png", dpi=240)
    plt.close()


def plot_covariance_scatter(df: pd.DataFrame) -> None:
    plt.figure(figsize=(8, 5.8))
    scatter = plt.scatter(
        df["local_log_kappa"],
        df["log_runtime"],
        c=df["alpha"],
        s=30,
        alpha=0.72,
    )
    plt.colorbar(scatter, label="alpha")
    plt.xlabel("Local log10(1 + kappa(C_d))")
    plt.ylabel("log10(runtime)")
    plt.title("Runtime vs local defect-covariance conditioning")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTDIR / "local_kappa_vs_runtime.png", dpi=240)
    plt.close()


def plot_hardness_boxes(df: pd.DataFrame) -> None:
    hard = df[df["hard_runtime"] == 1]["local_log_kappa"]
    soft = df[df["hard_runtime"] == 0]["local_log_kappa"]

    plt.figure(figsize=(6.8, 5.2))
    plt.boxplot([soft, hard], tick_labels=["not top 20%", "top 20% runtime"])
    plt.ylabel("Local log10(1 + kappa(C_d))")
    plt.title("Local covariance conditioning by runtime-hardness label")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTDIR / "local_kappa_by_hardness_label.png", dpi=240)
    plt.close()


def plot_full_model_importances(importances: dict) -> None:
    full = importances["alpha_fields_local_covariance"]
    items = sorted(full.items(), key=lambda kv: kv[1], reverse=True)
    labels = [k for k, _ in items]
    values = [v for _, v in items]

    plt.figure(figsize=(9.5, 5.8))
    plt.barh(labels[::-1], values[::-1])
    plt.xlabel("Random forest feature importance")
    plt.title("Hard-runtime classifier: full model feature importances")
    plt.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTDIR / "full_model_feature_importances.png", dpi=240)
    plt.close()


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)

    df = load_results()
    df = covariance_features(df)

    classifier, importances = fit_classifiers(df)
    regression = fit_regressors(df)
    correlations = correlation_summary(df)

    df.to_csv(OUTDIR / "sat_hardness_with_local_covariance.csv", index=False)
    result = {
        "n": int(len(df)),
        "hard_quantile_within_alpha": HARD_QUANTILE,
        "local_k": LOCAL_K,
        "hard_count": int(df["hard_runtime"].sum()),
        "classifier_metrics": classifier,
        "regression_metrics": regression,
        "correlations": correlations,
        "feature_importances": importances,
    }
    with open(OUTDIR / "sat_hardness_covariance_classifier_results.json", "w") as f:
        json.dump(result, f, indent=2)

    plot_classifier_comparison(classifier)
    plot_regression_comparison(regression)
    plot_covariance_scatter(df)
    plot_hardness_boxes(df)
    plot_full_model_importances(importances)

    print(json.dumps(result, indent=2))
    print(f"\nSaved classifier outputs to: {OUTDIR}")


if __name__ == "__main__":
    main()
