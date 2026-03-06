# analysis/hardness_atlas.py
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

RESULTS_PATH = "../results/maat_cosmos_full_results.csv"
OUTDIR = "../results"
os.makedirs(OUTDIR, exist_ok=True)

# --- config ---
ALPHA_COL = "alpha"
Y_COL = "V"              # <--- Atlas-Y: V ist dein stärkster Predictor
RT_COL = "runtime"
SAT_COL = "sat"

ALPHA_BINS = None        # None -> pro alpha als eigene Spalte (dein Setup)
Y_BINS = 18              # mehr = feiner, aber mehr NaNs
MIN_COUNT_PER_CELL = 8   # Zellen mit weniger Samples -> NaN (white)

def safe_log10(x):
    x = np.asarray(x, dtype=float)
    x = np.clip(x, 1e-12, None)
    return np.log10(x)

def main():
    df = pd.read_csv(RESULTS_PATH)

    # sanity
    for c in [ALPHA_COL, Y_COL, RT_COL]:
        if c not in df.columns:
            raise SystemExit(f"Missing column: {c}. Available: {list(df.columns)}")

    df["log10_rt"] = safe_log10(df[RT_COL])

    # alpha bins
    if ALPHA_BINS is None:
        # use unique alpha values as categorical bins (sorted)
        alphas = np.sort(df[ALPHA_COL].unique())
        df["alpha_bin"] = pd.Categorical(df[ALPHA_COL], categories=alphas, ordered=True)
        alpha_labels = [str(a) for a in alphas]
    else:
        df["alpha_bin"] = pd.cut(df[ALPHA_COL], bins=ALPHA_BINS)
        alpha_labels = [str(x) for x in df["alpha_bin"].cat.categories]

    # y bins
    y_min, y_max = df[Y_COL].min(), df[Y_COL].max()
    df["y_bin"] = pd.cut(df[Y_COL], bins=Y_BINS)

    # aggregate grid
    agg = df.groupby(["y_bin", "alpha_bin"]).agg(
        median_log_rt=("log10_rt", "median"),
        mean_log_rt=("log10_rt", "mean"),
        sat_prob=(SAT_COL, "mean") if SAT_COL in df.columns else ("log10_rt", "size"),
        count=("log10_rt", "size"),
    ).reset_index()

    # pivot to matrices
    pivot_med = agg.pivot(index="y_bin", columns="alpha_bin", values="median_log_rt")
    pivot_sat = agg.pivot(index="y_bin", columns="alpha_bin", values="sat_prob") if SAT_COL in df.columns else None
    pivot_n   = agg.pivot(index="y_bin", columns="alpha_bin", values="count")

    # mask low-count cells
    pivot_med = pivot_med.mask(pivot_n < MIN_COUNT_PER_CELL)
    if pivot_sat is not None:
        pivot_sat = pivot_sat.mask(pivot_n < MIN_COUNT_PER_CELL)

    # save grid csv
    grid_path = os.path.join(OUTDIR, "hardness_atlas_grid.csv")
    agg.to_csv(grid_path, index=False)
    print("grid saved:", grid_path)

    # --------- Plot 1: Hardness Atlas (median runtime) ----------
    fig = plt.figure(figsize=(14, 8))
    plt.title(f"SAT Hardness Atlas: median log10(runtime) over (alpha, {Y_COL})")
    im = plt.imshow(
        pivot_med.values,
        aspect="auto",
        origin="lower"
    )
    plt.colorbar(im, label="median log10(runtime)")
    plt.xlabel("alpha")
    plt.ylabel(f"{Y_COL} bin")

    # x ticks
    plt.xticks(ticks=np.arange(len(pivot_med.columns)), labels=[str(x) for x in pivot_med.columns], rotation=45)

    # y ticks: use interval labels but make it readable
    y_labels = [str(i) for i in pivot_med.index]
    step = max(1, len(y_labels)//12)
    yticks = np.arange(0, len(y_labels), step)
    plt.yticks(ticks=yticks, labels=[y_labels[i] for i in yticks])

    out1 = os.path.join(OUTDIR, f"hardness_atlas_alpha_vs_{Y_COL}.png")
    plt.tight_layout()
    plt.savefig(out1, dpi=200)
    print("atlas saved:", out1)
    plt.close(fig)

    # --------- Plot 2: SAT Probability Atlas ----------
    if pivot_sat is not None:
        fig = plt.figure(figsize=(14, 8))
        plt.title(f"SAT Probability Atlas: P(SAT) over (alpha, {Y_COL})")
        im = plt.imshow(
            pivot_sat.values,
            aspect="auto",
            origin="lower",
            vmin=0.0,
            vmax=1.0
        )
        plt.colorbar(im, label="P(SAT)")
        plt.xlabel("alpha")
        plt.ylabel(f"{Y_COL} bin")
        plt.xticks(ticks=np.arange(len(pivot_sat.columns)), labels=[str(x) for x in pivot_sat.columns], rotation=45)

        y_labels = [str(i) for i in pivot_sat.index]
        step = max(1, len(y_labels)//12)
        yticks = np.arange(0, len(y_labels), step)
        plt.yticks(ticks=yticks, labels=[y_labels[i] for i in yticks])

        out2 = os.path.join(OUTDIR, f"sat_prob_atlas_alpha_vs_{Y_COL}.png")
        plt.tight_layout()
        plt.savefig(out2, dpi=200)
        print("sat prob atlas saved:", out2)
        plt.close(fig)

    # --------- Plot 3: Confidence Map (counts) ----------
    fig = plt.figure(figsize=(14, 8))
    plt.title(f"Atlas Confidence Map: sample count per cell (alpha, {Y_COL})")
    im = plt.imshow(
        pivot_n.values,
        aspect="auto",
        origin="lower"
    )
    plt.colorbar(im, label="count")
    plt.xlabel("alpha")
    plt.ylabel(f"{Y_COL} bin")
    plt.xticks(ticks=np.arange(len(pivot_n.columns)), labels=[str(x) for x in pivot_n.columns], rotation=45)

    y_labels = [str(i) for i in pivot_n.index]
    step = max(1, len(y_labels)//12)
    yticks = np.arange(0, len(y_labels), step)
    plt.yticks(ticks=yticks, labels=[y_labels[i] for i in yticks])

    out3 = os.path.join(OUTDIR, f"atlas_confidence_alpha_vs_{Y_COL}.png")
    plt.tight_layout()
    plt.savefig(out3, dpi=200)
    print("confidence map saved:", out3)
    plt.close(fig)

    print("\n✅ SAT Hardness Atlas done.")

if __name__ == "__main__":
    main()