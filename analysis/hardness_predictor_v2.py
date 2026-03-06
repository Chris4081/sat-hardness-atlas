import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split


def run_predictor():

    print("\n--- SAT Hardness Predictor 2.0 ---")

    df = pd.read_csv("../results/maat_cosmos_full_results.csv")

    # sichere log runtime
    eps = 1e-12
    df["runtime_safe"] = df["runtime"].clip(lower=eps)
    df["log_rt"] = np.log10(df["runtime_safe"])

    # Features
    features = ["alpha","H","B","S","V","R"]

    X = df[features]
    y = df["log_rt"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,y,test_size=0.3,random_state=42
    )

    model = RandomForestRegressor(
        n_estimators=300,
        random_state=42
    )

    model.fit(X_train,y_train)

    score = model.score(X_test,y_test)

    print("\nPrediction R²:",score)

    print("\nFeature importance:")

    for name,val in zip(features,model.feature_importances_):
        print(f"{name}: {val:.3f}")


if __name__ == "__main__":
    run_predictor()