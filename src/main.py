import argparse
from pathlib import Path
from src.data_loader import fetch_data, load_data, prepare_data_for_training
from src.model import StockPredictor
from src.predictor import predict_latest

DATA_DIR = Path("data")

def train_model(ticker: str):
    file_path = fetch_data(ticker, DATA_DIR)
    df = load_data(file_path)
    X, y = prepare_data_for_training(df)

    predictor = StockPredictor()
    predictor.train(X, y)
    model_path = DATA_DIR / f"{ticker}_model.joblib"
    predictor.save_model(model_path)
    print(f"Model saved to {model_path}")

def make_prediction(ticker: str):
    predict_latest(ticker, DATA_DIR)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stock Price Predictor")
    parser.add_argument("command", choices=["train", "predict"], help="train or predict")
    parser.add_argument("ticker", help="Stock ticker symbol, e.g., AAPL")
    args = parser.parse_args()

    if args.command == "train":
        train_model(args.ticker)
    elif args.command == "predict":
        make_prediction(args.ticker)
