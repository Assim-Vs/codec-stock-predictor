from pathlib import Path
import pandas as pd
from joblib import dump, load
from sklearn.linear_model import LinearRegression
from src.data_loader import fetch_data, load_data, prepare_data_for_training

class StockPredictor:
    def __init__(self):
        self.model = LinearRegression()

    def train(self, X: pd.DataFrame, y: pd.Series):
        self.model.fit(X, y)

    def predict(self, X: pd.DataFrame) -> float:
        return self.model.predict(X)[0]

    def save_model(self, model_path: Path):
        dump(self.model, model_path)

    def load_model(self, model_path: Path):
        self.model = load(model_path)


def train_model(ticker: str, data_dir: Path):
    """
    Fetches data, trains the model, and saves it.
    Returns the trained model and data for plotting.
    """
    # 1. Fetch historical data (or load if already exists)
    data_path = fetch_data(ticker, data_dir)
    df = load_data(data_path)

    # 2. Prepare features and target
    X, y = prepare_data_for_training(df)

    # 3. Train model
    predictor = StockPredictor()
    predictor.train(X, y)

    # 4. Save model
    model_path = data_dir / f"{ticker}_model.joblib"
    predictor.save_model(model_path)

    return predictor, df


def predict_latest(ticker: str, data_dir: Path):
    """
    Loads trained model and predicts latest closing price.
    Returns prediction and dataframe for charting.
    """
    model_path = data_dir / f"{ticker}_model.joblib"
    data_path = data_dir / f"{ticker}_data.csv"

    df = load_data(data_path)

    predictor = StockPredictor()
    predictor.load_model(model_path)

    latest_data = df.iloc[-1]
    features = ['Open', 'High', 'Low', 'Volume']
    prediction_input = pd.DataFrame([latest_data[features]], columns=features)

    predicted_price = predictor.predict(prediction_input)
    return predicted_price, df

