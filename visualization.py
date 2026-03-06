import os
import matplotlib.pyplot as plt

def plot_maat_vs_runtime(df, outdir="results"):

    plt.figure(figsize=(6,4))

    plt.scatter(df["C_hat"], df["runtime"], alpha=0.6)

    plt.xlabel("MAAT Complexity Score (Ĉ)")
    plt.ylabel("SAT Solver Runtime")
    plt.title("MAAT Score vs Solver Runtime")

    os.makedirs(outdir, exist_ok=True)

    path = os.path.join(outdir, "maat_vs_runtime.png")
    plt.savefig(path, dpi=300)

    print("plot saved:", path)

    plt.close()