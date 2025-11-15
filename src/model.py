from sklearn.linear_model import LinearRegression
import joblib
from pathlib import Path
import pandas as pd

class StockPredictor:
    def __init__(self):
        self.model = LinearRegression()

    def train(self, X: pd.DataFrame, y: pd.Series):
        self.model.fit(X, y)

    def predict(self, X: pd.DataFrame) -> float:
        return float(self.model.predict(X)[0])

    def save_model(self, path: Path):
        joblib.dump(self.model, path)

    def load_model(self, path: Path):
        self.model = joblib.load(path)
