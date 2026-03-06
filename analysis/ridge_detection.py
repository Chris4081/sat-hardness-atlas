# analysis/ridge_detection.py
import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def ensure_cols(df, cols):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}. Found: {list(df.columns)}")


def ridge_detection(
    csv_path: str,
    y_col: str = "C_hat",          # "C_hat" oder "V" (oder H,B,S,R)
    alpha_col: str = "alpha",
    rt_col: str = "runtime",
    nbins_y: int = 20,
    min_count: int = 10,
    outdir: str = "../results",
):
    df = pd.read_csv(csv_path)
    ensure_cols(df, [alpha_col, y_col, rt_col])

    # log10(runtime) robust
    eps = 1e-12
    df = df.copy()
    df["log_rt"] = np.log10(df[rt_col].astype(float) + eps)

    # α als "diskrete Stützstellen" (bei dir sind es eh feste Werte)
    alpha_vals = np.sort(df[alpha_col].unique())

    # y-bins global
    y_min, y_max = df[y_col].min(), df[y_col].max()
    y_bins = np.linspace(y_min, y_max, nbins_y + 1)

    # Aggregation: median(log_rt) pro (alpha, y_bin) + count
    df["y_bin"] = pd.cut(df[y_col], bins=y_bins, include_lowest=True)
    agg = (
        df.groupby([alpha_col, "y_bin"], observed=True)
          .agg(median_log_rt=("log_rt", "median"),
               count=("log_rt", "size"),
               y_mid=(y_col, "median"))
          .reset_index()
    )

    # Hardness-Grid (für Heatmap) + Ridge (für Linie)
    pivot_med = agg.pivot(index="y_bin", columns=alpha_col, values="median_log_rt")
    pivot_cnt = agg.pivot(index="y_bin", columns=alpha_col, values="count")

    ridge_rows = []
    for a in alpha_vals:
        col_med = pivot_med[a]
        col_cnt = pivot_cnt[a]

        # nur Zellen mit genug Samples
        valid = (col_cnt >= min_count) & col_med.notna()
        if valid.sum() == 0:
            ridge_rows.append((a, np.nan, np.nan, 0))
            continue

        # "max hardness" = größtes median_log_rt (weil log_rt höher => runtime länger)
        best_idx = col_med[valid].idxmax()
        best_med = float(col_med.loc[best_idx])
        best_cnt = int(col_cnt.loc[best_idx])

        # y_mid aus agg holen
        y_mid = float(
            agg[(agg[alpha_col] == a) & (agg["y_bin"] == best_idx)]["y_mid"].iloc[0]
        )
        ridge_rows.append((a, y_mid, best_med, best_cnt))

    ridge_df = pd.DataFrame(ridge_rows, columns=["alpha", f"{y_col}_ridge", "median_log_rt", "count"])

    # Peak der Ridge
    peak = ridge_df.dropna().sort_values("median_log_rt", ascending=False).head(1)
    if len(peak):
        peak_alpha = float(peak["alpha"].iloc[0])
        peak_y = float(peak[f"{y_col}_ridge"].iloc[0])
        peak_med = float(peak["median_log_rt"].iloc[0])
    else:
        peak_alpha = peak_y = peak_med = np.nan

    os.makedirs(outdir, exist_ok=True)
    out_csv = os.path.join(outdir, f"hardness_ridge_{y_col}.csv")
    ridge_df.to_csv(out_csv, index=False)

    # Plot: Heatmap + Ridge-Line
    # y-axis labels aus bins
    y_labels = [str(i) for i in pivot_med.index]

    plt.figure(figsize=(14, 7))
    img = plt.imshow(
        pivot_med.values,
        aspect="auto",
        origin="lower",
        interpolation="nearest",
    )
    plt.colorbar(img, label="median log10(runtime)")

    plt.xticks(
        ticks=np.arange(len(pivot_med.columns)),
        labels=[f"{a:.2f}" for a in pivot_med.columns],
        rotation=45
    )
    plt.yticks(
        ticks=np.arange(len(y_labels)),
        labels=y_labels
    )

    # Ridge -> in Heatmap-Koordinaten mappen: y_mid => bin index
    # Wir nehmen für jede alpha die y_bin, die am besten war, und plotten deren index
    ridge_y_idx = []
    for a in pivot_med.columns:
        # finde y_bin mit max hardness unter min_count
        col_med = pivot_med[a]
        col_cnt = pivot_cnt[a]
        valid = (col_cnt >= min_count) & col_med.notna()
        if valid.sum() == 0:
            ridge_y_idx.append(np.nan)
            continue
        best_bin = col_med[valid].idxmax()
        ridge_y_idx.append(list(pivot_med.index).index(best_bin))

    plt.plot(np.arange(len(pivot_med.columns)), ridge_y_idx, marker="o", linewidth=2, label="Hardness Ridge")

    # Peak markieren (nächste alpha position)
    if not np.isnan(peak_alpha):
        # alpha index
        ax = list(pivot_med.columns).index(peak_alpha) if peak_alpha in pivot_med.columns else None
        if ax is not None:
            # y index aus ridge_y_idx
            ay = ridge_y_idx[ax]
            plt.scatter([ax], [ay], marker="*", s=250, label="Peak")

    plt.title(f"Hardness Landscape (median log10(runtime)) + Ridge over {y_col}")
    plt.xlabel("alpha")
    plt.ylabel(f"{y_col} bin")
    plt.legend()

    out_png = os.path.join(outdir, f"hardness_ridge_{y_col}.png")
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()

    print(f"ridge csv saved: {out_csv}")
    print(f"ridge plot saved: {out_png}")
    if not np.isnan(peak_alpha):
        print("\n--- HARDNESS RIDGE PEAK ---")
        print(f"alpha: {peak_alpha}")
        print(f"{y_col}_ridge(mid): {peak_y}")
        print(f"median log10(runtime): {peak_med}")
    else:
        print("No peak found (maybe min_count too high).")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="../results/maat_cosmos_full_results.csv")
    ap.add_argument("--y", default="C_hat", help="C_hat oder V oder H/B/S/R")
    ap.add_argument("--nbins", type=int, default=20)
    ap.add_argument("--min_count", type=int, default=10)
    ap.add_argument("--outdir", default="../results")
    args = ap.parse_args()

    ridge_detection(
        csv_path=args.csv,
        y_col=args.y,
        nbins_y=args.nbins,
        min_count=args.min_count,
        outdir=args.outdir,
    )