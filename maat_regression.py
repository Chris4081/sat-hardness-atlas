import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


def run_regression(df):

    # Feature Matrix
    X = df[["alpha", "C_hat"]].values

    # Zielvariable
    y = np.log(df["runtime"] + 1e-6)

    model = LinearRegression()
    model.fit(X, y)

    print("\n--- Regression Results ---")

    print("alpha weight:", model.coef_[0])
    print("C_hat weight:", model.coef_[1])
    print("intercept:", model.intercept_)

    r2 = model.score(X, y)

    print("R²:", r2)

    print("\nModel:")
    print(f"log(runtime) ≈ {model.coef_[0]:.3f} * alpha + {model.coef_[1]:.3f} * C_hat + {model.intercept_:.3f}")