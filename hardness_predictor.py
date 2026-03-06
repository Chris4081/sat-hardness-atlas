import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split


def run_hardness_predictor(df):

    print("\n--- SAT Hardness Predictor ---")

    # Features
    X = df[["alpha", "C_hat"]]

    # Ziel
    y = np.log10(df["runtime"] + 1e-6)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    model = RandomForestRegressor(
        n_estimators=200,
        random_state=42
    )

    model.fit(X_train, y_train)

    score = model.score(X_test, y_test)

    print("Prediction R²:", score)

    print("\nFeature importance:")

    for name, imp in zip(X.columns, model.feature_importances_):
        print(f"{name}: {imp:.3f}")