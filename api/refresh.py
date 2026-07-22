"""
api/refresh.py
================
Vercel Python serverless function. Handles GET /api/refresh.

Loads the pre-trained model (model/model.pkl), fetches FRESH prices for
the tracked tickers via yfinance, recomputes price-based features, and
returns updated predictions. Does NOT retrain the model and does NOT
re-fetch insider data (that stays static, refreshed offline via
pipeline/generate_site_data.py) — this keeps the function fast enough to
finish inside Vercel's 10-second Hobby-tier timeout.

Response shape (JSON):
{
  "generated_at": "...",
  "tickers": {
    "AAPL": {"prediction": "UP", "probability_up": 0.57, "last_price": 231.40, "as_of_date": "2026-07-21"},
    ...
  }
}
"""

import json
import os
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler

import joblib
import pandas as pd
import yfinance as yf

TICKERS = ["BAC", "SOFI", "INTC", "KDP", "DAL", "COIN", "AMZN", "SPG"]

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(REPO_ROOT, "model", "model.pkl")
RESULTS_PATH = os.path.join(REPO_ROOT, "data", "results.json")

# Price-only feature columns this function can recompute live.
# Insider feature columns are read from the last known values in
# results.json rather than recomputed, since insider data refreshes
# on a slower, separate cadence.
PRICE_FEATURE_COLS = [
    "return_1d", "return_5d", "sma_10", "sma_30", "price_vs_sma30", "volatility_10d",
]
INSIDER_FEATURE_COLS = [
    "insider_n_buys_30d", "insider_n_sells_30d", "insider_buy_sell_ratio_30d",
    "insider_net_value_30d", "insider_cluster_buy_flag",
]


def compute_price_features(close_series: pd.Series) -> dict:
    s = close_series.dropna()
    if len(s) < 31:
        return None
    return_1d = s.pct_change(1).iloc[-1]
    return_5d = s.pct_change(5).iloc[-1]
    sma_10 = s.rolling(10).mean().iloc[-1]
    sma_30 = s.rolling(30).mean().iloc[-1]
    price_vs_sma30 = s.iloc[-1] / sma_30 - 1
    volatility_10d = s.pct_change(1).rolling(10).std().iloc[-1]
    return {
        "return_1d": float(return_1d),
        "return_5d": float(return_5d),
        "sma_10": float(sma_10),
        "sma_30": float(sma_30),
        "price_vs_sma30": float(price_vs_sma30),
        "volatility_10d": float(volatility_10d),
        "last_price": float(s.iloc[-1]),
        "as_of_date": s.index[-1].strftime("%Y-%m-%d"),
    }


def run_refresh() -> dict:
    bundle = joblib.load(MODEL_PATH)
    model = bundle["model"]
    feature_cols = bundle["feature_cols"]

    with open(RESULTS_PATH) as f:
        last_results = json.load(f)

    raw = yf.download(TICKERS, period="6mo", progress=False)
    close = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw[["Close"]]
    if isinstance(close, pd.Series):
        close = close.to_frame(name=TICKERS[0])

    output = {}
    for ticker in TICKERS:
        if ticker not in close.columns:
            continue
        price_feats = compute_price_features(close[ticker])
        if price_feats is None:
            continue

        # Fall back to the last known insider feature values from the
        # static results file (insider data isn't re-fetched live here).
        prior = last_results.get("tickers", {}).get(ticker, {}).get("features", {})
        row = {c: price_feats[c] for c in PRICE_FEATURE_COLS}
        for c in INSIDER_FEATURE_COLS:
            row[c] = prior.get(c, 0)

        X = pd.DataFrame([[row[c] for c in feature_cols]], columns=feature_cols)
        prob_up = float(model.predict_proba(X)[:, 1][0])

        output[ticker] = {
            "prediction": "UP" if prob_up > 0.5 else "DOWN",
            "probability_up": round(prob_up, 4),
            "last_price": round(price_feats["last_price"], 2),
            "as_of_date": price_feats["as_of_date"],
        }

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "tickers": output,
    }


class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        try:
            result = run_refresh()
            self.send_response(200)
            self.send_header("Content-type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(json.dumps(result).encode("utf-8"))
        except Exception as e:
            self.send_response(500)
            self.send_header("Content-type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"error": str(e)}).encode("utf-8"))
        return
