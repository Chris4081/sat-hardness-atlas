# analysis/phase_surface.py
import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def ensure_cols(df, cols):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}. Found: {list(df.columns)}")


def build_surface(
    csv_path: str,
    y_col: str = "C_hat",          # "C_hat" oder "V"
    alpha_col: str = "alpha",
    rt_col: str = "runtime",
    mode: str = "median",          # "median" oder "mean"
    nbins_y: int = 25,
    min_count: int = 8,
    outdir: str = "../results",
):
    df = pd.read_csv(csv_path)
    ensure_cols(df, [alpha_col, y_col, rt_col])

    eps = 1e-12
    df = df.copy()
    df["log_rt"] = np.log10(df[rt_col].astype(float) + eps)

    alpha_vals = np.sort(df[alpha_col].unique())
    y_min, y_max = df[y_col].min(), df[y_col].max()
    y_bins = np.linspace(y_min, y_max, nbins_y + 1)

    df["y_bin"] = pd.cut(df[y_col], bins=y_bins, include_lowest=True)

    if mode == "mean":
        agg = df.groupby([alpha_col, "y_bin"], observed=True).agg(
            z=("log_rt", "mean"),
            count=("log_rt", "size"),
            y_mid=(y_col, "median"),
        ).reset_index()
    else:
        agg = df.groupby([alpha_col, "y_bin"], observed=True).agg(
            z=("log_rt", "median"),
            count=("log_rt", "size"),
            y_mid=(y_col, "median"),
        ).reset_index()

    pivot_z = agg.pivot(index="y_bin", columns=alpha_col, values="z")
    pivot_c = agg.pivot(index="y_bin", columns=alpha_col, values="count")

    # Maske: zu wenig samples -> NaN
    z = pivot_z.values.astype(float)
    c = pivot_c.values.astype(float)
    z[(c < min_count) | np.isnan(z)] = np.nan

    # Meshgrid
    X, Y = np.meshgrid(np.arange(len(alpha_vals)), np.arange(len(pivot_z.index)))
    # echte Achsenwerte
    x_labels = alpha_vals
    # y-mid je bin
    y_mids = []
    for b in pivot_z.index:
        # Bin-Mitte aus Intervall
        y_mids.append((b.left + b.right) / 2.0)
    y_mids = np.array(y_mids)

    # 3D Surface: NaNs raus
    # plot_surface kann NaNs, sieht aber oft "löchrig" aus – ist ok (zeigt Datenlücken ehrlich)
    os.makedirs(outdir, exist_ok=True)

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection="3d")

    # In echten Koordinaten plotten:
    X_real, Y_real = np.meshgrid(x_labels, y_mids)

    surf = ax.plot_surface(X_real, Y_real, z, linewidth=0, antialiased=True)
    ax.set_title(f"SAT Phase Surface ({mode} log10(runtime)) over (alpha, {y_col})")
    ax.set_xlabel("alpha")
    ax.set_ylabel(y_col)
    ax.set_zlabel("log10(runtime)")

    out_png = os.path.join(outdir, f"sat_phase_surface_{y_col}_{mode}.png")
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()

    # Zusätzlich: 2D Contour (oft leichter zu lesen)
    plt.figure(figsize=(14, 7))
    plt.imshow(z, aspect="auto", origin="lower", interpolation="nearest")
    plt.colorbar(label=f"{mode} log10(runtime)")

    plt.xticks(np.arange(len(alpha_vals)), [f"{a:.2f}" for a in alpha_vals], rotation=45)
    plt.yticks(np.arange(len(y_mids)), [f"{v:.3f}" for v in y_mids])

    plt.title(f"SAT Phase Contour ({mode} log10(runtime)) over (alpha, {y_col})")
    plt.xlabel("alpha")
    plt.ylabel(y_col)

    out_png2 = os.path.join(outdir, f"sat_phase_contour_{y_col}_{mode}.png")
    plt.tight_layout()
    plt.savefig(out_png2, dpi=200)
    plt.close()

    print("surface saved:", out_png)
    print("contour saved:", out_png2)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="../results/maat_cosmos_full_results.csv")
    ap.add_argument("--y", default="C_hat")
    ap.add_argument("--mode", default="median", choices=["median", "mean"])
    ap.add_argument("--nbins", type=int, default=25)
    ap.add_argument("--min_count", type=int, default=8)
    ap.add_argument("--outdir", default="../results")
    args = ap.parse_args()

    build_surface(
        csv_path=args.csv,
        y_col=args.y,
        mode=args.mode,
        nbins_y=args.nbins,
        min_count=args.min_count,
        outdir=args.outdir,
    )