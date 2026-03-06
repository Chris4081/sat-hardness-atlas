import pandas as pd
import numpy as np
import statsmodels.api as sm

df = pd.read_csv("../results/maat_cosmos_full_results.csv")

df["log_rt"] = np.log10(df["runtime"].clip(lower=1e-12))

X = df[["alpha","H","B","S","V","R"]]
X = sm.add_constant(X)

y = df["log_rt"]

model = sm.OLS(y,X).fit()

print(model.summary())