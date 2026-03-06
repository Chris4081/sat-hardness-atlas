import os
import matplotlib.pyplot as plt
import pandas as pd


def plot_sat_probability(df, outdir="results"):

    os.makedirs(outdir, exist_ok=True)

    # SAT probability pro alpha
    sat_prob = df.groupby("alpha")["sat"].mean()

    plt.figure(figsize=(8,5))

    plt.plot(
        sat_prob.index,
        sat_prob.values,
        marker="o"
    )

    plt.xlabel("alpha (clauses / variables)")
    plt.ylabel("SAT probability")
    plt.title("SAT Phase Transition")

    plt.grid(True)

    path = os.path.join(outdir, "sat_phase_transition.png")

    plt.savefig(path, dpi=300)
    plt.close()

    print("phase transition plot saved:", path)