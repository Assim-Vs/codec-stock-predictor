from pathlib import Path
import pandas as pd
import yfinance as yf
from typing import Tuple

def fetch_data(ticker: str, data_dir: Path, period: str = "max") -> Path:
    data_dir.mkdir(parents=True, exist_ok=True)
    file_path = data_dir / f"{ticker}_data.csv"
    stock = yf.Ticker(ticker)
    hist = stock.history(period=period)
    if hist.empty:
        raise ValueError(f"No data found for {ticker}.")
    hist.to_csv(file_path)
    print(f"Data for {ticker} saved to {file_path}")
    return file_path

def load_data(file_path: Path) -> pd.DataFrame:
    if not file_path.exists():
        raise FileNotFoundError(f"No file found at {file_path}")
    return pd.read_csv(file_path, index_col='Date', parse_dates=True)

def prepare_data_for_training(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    features = ['Open', 'High', 'Low', 'Volume']
    target = 'Close'
    X = df[features].dropna()
    y = df.loc[X.index, target]
    return X, y
