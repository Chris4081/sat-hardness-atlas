import os
import numpy as np
import matplotlib.pyplot as plt

def plot_sat_hardness_heatmap(df, outdir="results"):
    """
    Heatmap: alpha (x) vs C_hat bins (y) -> mean runtime
    """

    os.makedirs(outdir, exist_ok=True)

    # --- Beispiel-Binning ---
    alphas = np.sort(df["alpha"].unique())
    cmin, cmax = float(df["C_hat"].min()), float(df["C_hat"].max())
    bins = np.linspace(cmin, cmax, 25)  # 24 rows

    heat = np.full((len(bins)-1, len(alphas)), np.nan, dtype=float)

    for j, a in enumerate(alphas):
        sub = df[df["alpha"] == a]
        # C_hat in bins
        inds = np.digitize(sub["C_hat"].values, bins) - 1
        for i in range(len(bins)-1):
            mask = inds == i
            if np.any(mask):
                heat[i, j] = float(np.mean(sub["runtime"].values[mask]))

    plt.figure(figsize=(10, 6))
    im = plt.imshow(
        heat,
        aspect="auto",
        origin="lower",
        interpolation="nearest"
    )
    plt.colorbar(im, label="Mean runtime")

    plt.xticks(np.arange(len(alphas)), [str(a) for a in alphas], rotation=45)
    plt.yticks(
        np.arange(len(bins)-1),
        [f"{bins[i]:.3f}-{bins[i+1]:.3f}" for i in range(len(bins)-1)]
    )

    plt.xlabel("alpha")
    plt.ylabel("C_hat bin")
    plt.title("SAT hardness heatmap (mean runtime)")

    path = os.path.join(outdir, "sat_hardness_heatmap.png")
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()

    print("heatmap saved:", path)