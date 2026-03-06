import os
import numpy as np
import matplotlib.pyplot as plt


def plot_median_runtime_vs_alpha(df, outdir="results"):

    os.makedirs(outdir, exist_ok=True)

    # log runtime stabilisiert den Plot
    df["log_runtime"] = np.log10(df["runtime"] + 1e-6)

    grouped = df.groupby("alpha")["log_runtime"].median()

    plt.figure(figsize=(8,5))

    plt.plot(
        grouped.index,
        grouped.values,
        marker="o"
    )

    plt.xlabel("alpha (clauses / variables)")
    plt.ylabel("median log10(runtime)")
    plt.title("SAT Hardness Curve")

    plt.grid(True)

    path = os.path.join(outdir, "sat_hardness_curve.png")

    plt.savefig(path, dpi=300)
    plt.close()

    print("hardness curve saved:", path)