from flask import Flask, render_template, request
from pathlib import Path
from src.predictor import predict_latest, train_model
import plotly.graph_objects as go
import pandas as pd

app = Flask(__name__)
DATA_DIR = Path(__file__).parent.parent / "data"

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    error_msg = None
    chart_html = None

    if request.method == "POST":
        ticker = request.form.get("ticker", "").upper()

        try:
            # Check if model exists, if not train automatically
            model_path = DATA_DIR / f"{ticker}_model.joblib"
            data_path = DATA_DIR / f"{ticker}_data.csv"

            if not model_path.exists() or not data_path.exists():
                train_model(ticker, DATA_DIR)

            # Predict latest price
            predicted_price, df = predict_latest(ticker, DATA_DIR)
            prediction = f"Predicted Close Price for {ticker}: {predicted_price:.2f}"

            # Prepare animated chart
            df = df.tail(60)  # last 60 days
            fig = go.Figure()

            fig.add_trace(go.Scatter(
                x=df.index,
                y=df['Close'],
                mode='lines+markers',
                name='Actual Close'
            ))

            fig.add_trace(go.Scatter(
                x=[df.index[-1]],
                y=[predicted_price],
                mode='markers',
                marker=dict(size=12, color='red'),
                name='Predicted Close'
            ))

            fig.update_layout(
                title=f"{ticker} Stock Prices",
                xaxis_title="Date",
                yaxis_title="Price",
                template="plotly_dark",
                showlegend=True
            )

            chart_html = fig.to_html(full_html=False)

        except FileNotFoundError:
            error_msg = f"No data found for ticker '{ticker}'. Please check the symbol."
        except Exception as e:
            error_msg = f"Error fetching or predicting data: {e}"

    return render_template("index.html",
                           prediction=prediction,
                           error_msg=error_msg,
                           chart_html=chart_html)


if __name__ == "__main__":
    app.run(debug=True)


