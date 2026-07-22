"""
api/refresh.py
================
Vercel Python serverless function. Handles GET /api/refresh.
"""
import json
import sys
import traceback
from http.server import BaseHTTPRequestHandler
from pathlib import Path

# Vercel sets the root to /var/task
_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent

def find_file(name):
    """Search common Vercel paths for the data file."""
    candidates = [
        _REPO_ROOT / "data" / name,
        _HERE / "data" / name,
        _REPO_ROOT / "model" / name,  # Checks the model/ folder
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

class handler(BaseHTTPRequestHandler):
    def _send_json(self, code, data):
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(json.dumps(data, default=str).encode("utf-8"))

    def do_GET(self):
        try:
            # 1. Lazy imports (prevents container crash if dependency is missing)
            from datetime import datetime, timezone
            import joblib
            import pandas as pd
            import yfinance as yf

            # 2. Find files safely
            MODEL_PATH = find_file("model.pkl")
            RESULTS_PATH = find_file("results.json")
            
            if not MODEL_PATH:
                raise FileNotFoundError("model.pkl not found. Ensure it is in the 'model' folder and vercel.json includes it.")
            if not RESULTS_PATH:
                raise FileNotFoundError("results.json not found. Ensure it is in the 'data' folder and vercel.json includes it.")

            # 3. Load model safely
            bundle = joblib.load(MODEL_PATH)
            if isinstance(bundle, dict):
                model = bundle.get("model")
                feature_cols = bundle.get("feature_cols", PRICE_FEATURE_COLS + INSIDER_FEATURE_COLS)
            else:
                model = bundle
                feature_cols = PRICE_FEATURE_COLS + INSIDER_FEATURE_COLS

            # 4. Load static results
            with open(RESULTS_PATH, "r", encoding="utf-8") as f:
                last_results = json.load(f)

            # 5. Fetch live prices
            raw = yf.download(TICKERS, period="6mo", progress=False)
            close = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw[["Close"]]
            if isinstance(close, pd.Series):
                close = close.to_frame(name=TICKERS[0])

            # 6. Compute features and predict
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

            result = {
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "tickers": output,
            }
            self._send_json(200, result)

        except Exception as e:
            # Catch the error and return it as JSON so the frontend can display it
            err_msg = str(e)
            err_type = str(type(e).__name__)
            traceback_str = traceback.format_exc()
            print(f"Error in /api/refresh: {err_type}: {err_msg}\n{traceback_str}", file=sys.stderr)
            
            self._send_json(500, {
                "error": f"{err_type}: {err_msg}",
                "trace": traceback_str
            })
