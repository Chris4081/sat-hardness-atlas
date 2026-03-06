# training.py

import pandas as pd

from predictor import MaatComplexityPredictor


df = pd.read_csv("results/maat_cosmos_full_results.csv")

predictor = MaatComplexityPredictor()

predictor.train(df)

predictor.save("maat_complexity_model.pkl")

print("Model trained and saved.")