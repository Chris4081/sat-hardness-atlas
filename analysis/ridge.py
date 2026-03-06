import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("../results/maat_cosmos_full_results.csv")

eps = 1e-12
df["runtime_safe"] = df["runtime"].clip(lower=eps)
df["log_rt"] = np.log10(df["runtime_safe"])

# choose binning
bins = 25
df["c_bin"] = pd.cut(df["C_hat"], bins=bins)

g = (df.groupby(["alpha", "c_bin"], observed=True)["log_rt"]
       .median()
       .reset_index(name="median_log_rt")
       .dropna())

# ridge
ridge = g.loc[g.groupby("alpha")["median_log_rt"].idxmax()].sort_values("alpha")
peak = ridge.loc[ridge["median_log_rt"].idxmax()]

# grid for heatmap
pivot = g.pivot(index="c_bin", columns="alpha", values="median_log_rt")

plt.figure(figsize=(12,6))
plt.imshow(pivot.values, aspect="auto", origin="lower")
plt.title("Hardness Landscape (median log10(runtime)) + Ridge")
plt.xlabel("alpha")
plt.ylabel("C_hat bin")
plt.colorbar(label="median log10(runtime)")

# axis ticks
alphas = list(pivot.columns)
plt.xticks(range(len(alphas)), [f"{a:.2f}" for a in alphas], rotation=45)

c_bins = list(pivot.index)
plt.yticks(range(len(c_bins)), [str(b) for b in c_bins])

# ridge line mapping: convert c_bin to row index
bin_to_row = {b:i for i,b in enumerate(c_bins)}
x = [alphas.index(a) for a in ridge["alpha"]]
y = [bin_to_row[b] for b in ridge["c_bin"]]

plt.plot(x, y, marker="o")

# peak marker
px = alphas.index(float(peak["alpha"]))
py = bin_to_row[peak["c_bin"]]
plt.scatter([px], [py], s=150, marker="*", label="Peak")
plt.legend()

out = "../results/hardness_landscape_ridge.png"
plt.tight_layout()
plt.savefig(out, dpi=200)
print("saved:", out)