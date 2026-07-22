"""
api/refresh.py
================
Modern 2026 Vercel Python Serverless Function (ASGI format).

Vercel's latest Python runtime natively supports ASGI/WSGI apps. By defining 
an `app` object, we bypass the legacy `BaseHTTPRequestHandler` entirely, 
resulting in cleaner code, better performance, and native async support.

This function loads the pre-trained model (model/model.pkl), fetches FRESH 
prices for the tracked tickers via yfinance, recomputes price-based features, 
and returns updated predictions. It does NOT retrain the model and does NOT 
re-fetch insider data — keeping the function well within Vercel's 10-second 
Hobby-tier timeout.
"""

import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd
import yfinance as yf

TICKERS = ["BAC", "SOFI", "INTC", "KDP", "DAL", "COIN", "AMZN", "SPG"]

# Bulletproof path resolution for Vercel's serverless environment.
# On Vercel, the function runs in /var/task/, so the repo root is /var/task/.
_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent

def find_file(candidates):
    """Return the first existing file from a list of candidates."""
    for p in candidates:
        if p.exists():
            return p
    return candidates[0] # Fallback to the first to throw a clear error if missing

# Vercel bundles files based on vercel.json. We check multiple locations.
MODEL_PATH = find_file([
    _REPO_ROOT / "model" / "model.pkl",
    _REPO_ROOT / "data" / "model.pkl",  # Fallback if bundled with data
    _HERE / "model.pkl",
])

RESULTS_PATH = find_file([
    _REPO_ROOT / "data" / "results.json",
    _HERE / "results.json",
])

# Price-only feature columns this function can recompute live.
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
    """Synchronous ML logic executed safely in a background thread."""
    bundle = joblib.load(MODEL_PATH)
    model = bundle["model"]
    feature_cols = bundle["feature_cols"]

    with open(RESULTS_PATH, "r", encoding="utf-8") as f:
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

async def app(scope, receive, send):
    """
    Modern ASGI application entry point.
    Vercel's runtime natively detects and executes the `app` object.
    """
    if scope["type"] != "http":
        return

    method = scope.get("method", "")

    # Handle CORS preflight requests
    if method == "OPTIONS":
        await send({
            "type": "http.response.start",
            "status": 204,
            "headers": [
                [b"access-control-allow-origin", b"*"],
                [b"access-control-allow-methods", b"GET, OPTIONS"],
                [b"access-control-allow-headers", b"Content-Type"],
            ],
        })
        await send({"type": "http.response.body", "body": b""})
        return

    if method == "GET":
        try:
            # Run synchronous, CPU/IO-bound ML logic in a separate thread 
            # to avoid blocking the ASGI event loop.
            result = await asyncio.to_thread(run_refresh)
            
            response_body = json.dumps(result).encode("utf-8")
            await send({
                "type": "http.response.start",
                "status": 200,
                "headers": [
                    [b"content-type", b"application/json"],
                    [b"access-control-allow-origin", b"*"],
                ],
            })
            await send({"type": "http.response.body", "body": response_body})
        except Exception as e:
            error_body = json.dumps({"error": str(e)}).encode("utf-8")
            await send({
                "type": "http.response.start",
                "status": 500,
                "headers": [
                    [b"content-type", b"application/json"],
                    [b"access-control-allow-origin", b"*"],
                ],
            })
            await send({"type": "http.response.body", "body": error_body})
    else:
        # Reject non-GET/OPTIONS methods
        await send({
            "type": "http.response.start",
            "status": 405,
            "headers": [[b"content-type", b"application/json"]],
        })
        await send({"type": "http.response.body", "body": b'{"error": "Method Not Allowed"}'})
