"""
api/refresh.py
================
Modern 2026 Vercel Python Serverless Function (ASGI format).

Vercel's latest Python runtime natively supports ASGI apps. By defining 
an `app` object, we bypass the legacy BaseHTTPRequestHandler entirely, 
which fixes the "No python entrypoint found" build error.
"""
import asyncio
import json
import sys
import traceback
from pathlib import Path
from datetime import datetime, timezone

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent

def find_file(name):
    """Search common Vercel paths for the data file."""
    candidates = [
        _REPO_ROOT / "data" / name,
        _HERE / "data" / name,
        _REPO_ROOT / "model" / name,
        _HERE / "model" / name,
        _REPO_ROOT / name,
        Path("/var/task") / "data" / name,
        Path("/var/task") / "model" / name,
        Path("/var/task") / name,
    ]
    for p in candidates:
        if p.exists():
            return p
    return None

PRICE_FEATURE_COLS = [
    "return_1d", "return_5d", "sma_10", "sma_30", "price_vs_sma30", "volatility_10d",
]
INSIDER_FEATURE_COLS = [
    "insider_n_buys_30d", "insider_n_sells_30d", "insider_buy_sell_ratio_30d",
    "insider_net_value_30d", "insider_cluster_buy_flag",
]
TICKERS = ["BAC", "SOFI", "INTC", "KDP", "DAL", "COIN", "AMZN", "SPG"]

def run_refresh() -> dict:
    """Synchronous ML logic executed safely in a background thread."""
    # Lazy imports prevent container crashes if a dependency is missing
    import joblib
    import pandas as pd
    import yfinance as yf

    MODEL_PATH = find_file("model.pkl")
    RESULTS_PATH = find_file("results.json")
    
    if not MODEL_PATH:
        raise FileNotFoundError("model.pkl not found. Ensure it is in the 'model' folder and vercel.json includes it.")
    if not RESULTS_PATH:
        raise FileNotFoundError("results.json not found. Ensure it is in the 'data' folder and vercel.json includes it.")

    bundle = joblib.load(MODEL_PATH)
    if isinstance(bundle, dict):
        model = bundle.get("model")
        feature_cols = bundle.get("feature_cols", PRICE_FEATURE_COLS + INSIDER_FEATURE_COLS)
    else:
        model = bundle
        feature_cols = PRICE_FEATURE_COLS + INSIDER_FEATURE_COLS

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
        
        s = close[ticker].dropna()
        if len(s) < 31:
            continue

        price_feats = {
            "return_1d": float(s.pct_change(1).iloc[-1]),
            "return_5d": float(s.pct_change(5).iloc[-1]),
            "sma_10": float(s.rolling(10).mean().iloc[-1]),
            "sma_30": float(s.rolling(30).mean().iloc[-1]),
            "price_vs_sma30": float(s.iloc[-1] / s.rolling(30).mean().iloc[-1] - 1),
            "volatility_10d": float(s.pct_change(1).rolling(10).std().iloc[-1]),
            "last_price": float(s.iloc[-1]),
            "as_of_date": s.index[-1].strftime("%Y-%m-%d"),
        }

        prior = last_results.get("tickers", {}).get(ticker, {}).get("features", {})
        row = {c: price_feats.get(c, 0.0) for c in PRICE_FEATURE_COLS}
        for c in INSIDER_FEATURE_COLS:
            row[c] = prior.get(c, 0.0)

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
            err_msg = str(e)
            err_type = str(type(e).__name__)
            traceback_str = traceback.format_exc()
            print(f"Error in /api/refresh: {err_type}: {err_msg}\n{traceback_str}", file=sys.stderr)
            
            error_body = json.dumps({
                "error": f"{err_type}: {err_msg}",
                "trace": traceback_str
            }).encode("utf-8")
            
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
        await send({
            "type": "http.response.start",
            "status": 405,
            "headers": [[b"content-type", b"application/json"]],
        })
        await send({"type": "http.response.body", "body": b'{"error": "Method Not Allowed"}'})
