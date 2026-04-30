import pandas as pd
from pathlib import Path

src = Path("results/maat_cosmos_full_results.csv")
out = Path("../maat_defects_sat.csv")

df = pd.read_csv(src)

fields = ["H", "B", "S", "V", "R"]

# Sicherstellen: Werte in [0,1]
for col in fields:
    df[col] = df[col].clip(0, 1)

defects = pd.DataFrame({
    "d_H": 1 - df["H"],
    "d_B": 1 - df["B"],
    "d_S": 1 - df["S"],
    "d_V": 1 - df["V"],
    "d_R": 1 - df["R"],
    "label": df["sat"].map({True: "sat", False: "unsat"}),
    "source": "sat_hardness_atlas",
    "alpha": df["alpha"],
    "runtime": df["runtime"],
    "C_hat": df["C_hat"],
})

defects.to_csv(out, index=False)

print(f"Wrote {out}")
print(defects.head())
print(defects.describe())
