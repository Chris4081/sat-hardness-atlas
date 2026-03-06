# predictor.py

import joblib
import numpy as np
from sklearn.ensemble import RandomForestRegressor


class MaatComplexityPredictor:

    def __init__(self):

        self.model = RandomForestRegressor(
            n_estimators=200,
            max_depth=8,
            random_state=42
        )

    def train(self, df):

        X = df[["H","B","S","V","R"]]
        y = df["runtime"]

        self.model.fit(X, y)

    def predict(self, H,B,S,V,R):

        X = np.array([[H,B,S,V,R]])

        return self.model.predict(X)[0]

    def save(self, path):

        joblib.dump(self.model, path)

    def load(self, path):

        self.model = joblib.load(path)