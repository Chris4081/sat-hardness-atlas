import os
import numpy as np
import matplotlib.pyplot as plt


def plot_chat_vs_runtime(df, outdir="results"):

    os.makedirs(outdir, exist_ok=True)

    runtime = np.log10(df["runtime"] + 1e-6)

    plt.figure(figsize=(8,5))

    plt.scatter(
        df["C_hat"],
        runtime,
        alpha=0.4
    )

    plt.xlabel("C_hat")
    plt.ylabel("log10(runtime)")
    plt.title("C_hat vs Runtime")

    plt.grid(True)

    path = os.path.join(outdir, "chat_vs_runtime.png")

    plt.savefig(path, dpi=300)
    plt.close()

    print("plot saved:", path)



def plot_sat_prob_vs_chat(df, outdir="results", bins=12):

    os.makedirs(outdir, exist_ok=True)

    c = df["C_hat"]

    bin_edges = np.linspace(c.min(), c.max(), bins + 1)

    sat_prob = []
    centers = []

    for i in range(bins):

        mask = (c >= bin_edges[i]) & (c < bin_edges[i+1])

        if mask.sum() == 0:
            continue

        prob = df.loc[mask, "sat"].mean()

        sat_prob.append(prob)

        centers.append((bin_edges[i] + bin_edges[i+1]) / 2)


    plt.figure(figsize=(8,5))

    plt.plot(
        centers,
        sat_prob,
        marker="o"
    )

    plt.xlabel("C_hat")
    plt.ylabel("SAT probability")
    plt.title("SAT probability vs C_hat")

    plt.grid(True)

    path = os.path.join(outdir, "sat_prob_vs_chat.png")

    plt.savefig(path, dpi=300)
    plt.close()

    print("plot saved:", path)