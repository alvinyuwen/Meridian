"""
generate_site_data.py
======================
OFFLINE script — run this on your own computer (or github.dev's terminal,
or Google Colab), NOT on Vercel. It does the expensive work:

  1. Loads real price history from ticker_prices.csv (local, full OHLCV
     history — no network/yfinance needed for training).
  2. Loads real SEC Form 4 filing data from filings.csv (aggregated per
     filing: role, buy/sell/none signal, dollar value).
  3. Builds insider + price features, leak-safe labels.
  4. Trains + tunes a RandomForestClassifier with a real time-based
     train/test split.
  5. Saves the trained model to model/model.pkl (joblib).
  6. Exports everything the website needs into data/results.json.

NOTE ON DATA: filings.csv is aggregated per filing (one row per Form 4),
not per individual insider — there's no insider name column, only a role
(e.g. "CFO", "Director"). "Cluster buying" is therefore approximated as
"3+ separate buy filings in the window", not "3+ named individuals" —
functionally similar, just not identity-verified.

Run this once before your first deploy, and again any time you refresh
the underlying CSVs or retrain.

Usage:
    pip install pandas scikit-learn joblib numpy
    python pipeline/generate_site_data.py
"""

import json
import os
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, roc_auc_score,
    classification_report,
)

# ----------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------
# Picked from the real filings.csv: tickers with real, balanced buy AND
# sell filing activity (not just routine sell-heavy noise), all present
# in ticker_prices.csv too. Swap freely — just check a ticker has both
# buy and sell rows in filings.csv before adding it, or the insider
# features for it will be mostly zero.
TICKERS = ["BAC", "SOFI", "INTC", "KDP", "DAL", "COIN", "AMZN", "SPG"]

HORIZON_DAYS = 15
UP_THRESHOLD = 0.0
INSIDER_WINDOW_DAYS = 30

# filings.csv covers 2020-01-02 to 2025-07-14; ticker_prices.csv covers
# 2000-01-03 to 2025-10-07. Start a bit before filings begin (buffer for
# the 30-day SMA warmup) and use the fixed split below for a real
# out-of-time test period.
PRICE_START = "2019-06-01"
PRICE_END = "2025-10-07"
TRAIN_TEST_SPLIT_DATE = "2024-06-01"

PRICE_CSV_PATH = "ticker_prices.csv"
INSIDER_CSV_PATH = "filings.csv"

FEATURE_COLS = [
    "return_1d", "return_5d", "sma_10", "sma_30", "price_vs_sma30", "volatility_10d",
    "insider_n_buys_30d", "insider_n_sells_30d", "insider_buy_sell_ratio_30d",
    "insider_net_value_30d", "insider_cluster_buy_flag",
]

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(REPO_ROOT, "model", "model.pkl")
RESULTS_PATH = os.path.join(REPO_ROOT, "data", "results.json")


# ----------------------------------------------------------------------
# PRICE DATA (from local CSV — real OHLCV history)
# ----------------------------------------------------------------------
def load_price_data(tickers: list) -> pd.DataFrame:
    df = pd.read_csv(PRICE_CSV_PATH)
    df = df.rename(columns={"ticker_symbol": "Ticker", "date": "Date", "close": "Close"})
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df[df["Ticker"].isin(tickers)]
    df = df[(df["Date"] >= PRICE_START) & (df["Date"] <= PRICE_END)]
    return df[["Ticker", "Date", "Close"]].dropna().sort_values(["Ticker", "Date"])


# ----------------------------------------------------------------------
# INSIDER DATA (from local CSV — real aggregated Form 4 filings)
# ----------------------------------------------------------------------
def load_insider_data(tickers: list) -> pd.DataFrame:
    df = pd.read_csv(INSIDER_CSV_PATH)
    df = df.rename(columns={"ticker_symbol": "Ticker"})
    df = df[df["Ticker"].isin(tickers)].copy()

    # Prefer earliest_execution_date; fall back to filing_date if missing
    # or corrupted (a handful of rows have nonsense years like 0022/2035).
    exec_dt = pd.to_datetime(df["earliest_execution_date"], errors="coerce")
    filing_dt = pd.to_datetime(df["filing_date"], errors="coerce")
    sane = (exec_dt >= "2000-01-01") & (exec_dt <= pd.Timestamp.today() + pd.Timedelta(days=1))
    df["Trade Date"] = exec_dt.where(sane, filing_dt)
    df = df.dropna(subset=["Trade Date"])

    df["is_buy"] = (df["aggregated_signal"] == "buy").astype(int)
    df["is_sell"] = (df["aggregated_signal"] == "sell").astype(int)
    df["Value"] = df["aggregated_value_usd"].abs().fillna(0)
    df["Title"] = df["insider_role"].fillna("Unknown")
    # No individual insider names in this dataset — use a short filing-based
    # pseudo-id so the Deep-Dive timeline has something to display per row.
    df["Insider Name"] = df["accession_number"].astype(str).str[-8:].apply(lambda x: f"Filer #{x}")

    return df.sort_values("Trade Date").reset_index(drop=True)


# ----------------------------------------------------------------------
# FEATURES
# ----------------------------------------------------------------------
def build_insider_features(insider_df: pd.DataFrame, price_dates: pd.DataFrame) -> pd.DataFrame:
    rows = []
    window = pd.Timedelta(days=INSIDER_WINDOW_DAYS)
    for ticker, grp in price_dates.groupby("Ticker"):
        ins = insider_df[insider_df["Ticker"] == ticker].sort_values("Trade Date")
        for d in grp["Date"].values:
            d_ts = pd.Timestamp(d)
            recent = ins[(ins["Trade Date"] > d_ts - window) & (ins["Trade Date"] <= d_ts)]
            n_buys, n_sells = int(recent["is_buy"].sum()), int(recent["is_sell"].sum())
            buy_value = recent.loc[recent["is_buy"] == 1, "Value"].sum()
            sell_value = recent.loc[recent["is_sell"] == 1, "Value"].sum()
            rows.append({
                "Ticker": ticker, "Date": d_ts,
                "insider_n_buys_30d": n_buys, "insider_n_sells_30d": n_sells,
                "insider_buy_sell_ratio_30d": (n_buys + 1) / (n_sells + 1),
                "insider_net_value_30d": buy_value - sell_value,
                # Approximated as "3+ separate buy filings" — see module
                # docstring note on why we can't count unique individuals.
                "insider_cluster_buy_flag": int(n_buys >= 3),
            })
    return pd.DataFrame(rows)


def build_price_features(price_df: pd.DataFrame) -> pd.DataFrame:
    out = []
    for ticker, grp in price_df.groupby("Ticker"):
        grp = grp.sort_values("Date").copy()
        grp["return_1d"] = grp["Close"].pct_change(1)
        grp["return_5d"] = grp["Close"].pct_change(5)
        grp["sma_10"] = grp["Close"].rolling(10).mean()
        grp["sma_30"] = grp["Close"].rolling(30).mean()
        grp["price_vs_sma30"] = grp["Close"] / grp["sma_30"] - 1
        grp["volatility_10d"] = grp["return_1d"].rolling(10).std()
        out.append(grp)
    return pd.concat(out, ignore_index=True)


def build_labels(price_df: pd.DataFrame) -> pd.DataFrame:
    out = []
    for ticker, grp in price_df.groupby("Ticker"):
        grp = grp.sort_values("Date").copy()
        grp["future_close"] = grp["Close"].shift(-HORIZON_DAYS)
        grp["forward_return"] = grp["future_close"] / grp["Close"] - 1
        grp["label"] = (grp["forward_return"] > UP_THRESHOLD).astype(int)
        out.append(grp[["Ticker", "Date", "label", "forward_return"]])
    return pd.concat(out, ignore_index=True)


# ----------------------------------------------------------------------
# MAIN PIPELINE
# ----------------------------------------------------------------------
def main():
    print("Loading price data from", PRICE_CSV_PATH, "...")
    price_df = load_price_data(TICKERS)
    print(f"  {len(price_df)} price rows across {price_df['Ticker'].nunique()} tickers")

    print("Loading insider filings from", INSIDER_CSV_PATH, "...")
    insider_df = load_insider_data(TICKERS)
    print(f"  {len(insider_df)} filings ({insider_df['is_buy'].sum()} buys, {insider_df['is_sell'].sum()} sells)")

    price_features = build_price_features(price_df)
    insider_features = build_insider_features(insider_df, price_df[["Ticker", "Date"]])
    labels = build_labels(price_df)

    df = price_features.merge(insider_features, on=["Ticker", "Date"], how="left")
    df = df.merge(labels, on=["Ticker", "Date"], how="inner")
    df_full = df.copy()
    df = df.dropna(subset=FEATURE_COLS + ["label"]).reset_index(drop=True)

    print(f"Dataset: {len(df)} rows")

    train = df[df["Date"] < TRAIN_TEST_SPLIT_DATE]
    test = df[df["Date"] >= TRAIN_TEST_SPLIT_DATE]
    X_train, y_train = train[FEATURE_COLS], train["label"]
    X_test, y_test = test[FEATURE_COLS], test["label"]
    print(f"Train rows: {len(X_train)} | Test rows: {len(X_test)}")
    print(f"Train label balance:\n{y_train.value_counts(normalize=True)}")

    print("Training model...")
    param_grid = {"n_estimators": [200, 400], "max_depth": [4, 8, None], "min_samples_leaf": [5, 10]}
    grid = GridSearchCV(
        RandomForestClassifier(random_state=42, class_weight="balanced"),
        param_grid, cv=3, scoring="roc_auc", n_jobs=-1,
    )
    grid.fit(X_train, y_train)
    model = grid.best_estimator_
    print("Best params:", grid.best_params_)

    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": float(accuracy_score(y_test, preds)),
        "precision": float(precision_score(y_test, preds, zero_division=0)),
        "recall": float(recall_score(y_test, preds, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, probs)),
        "classification_report": classification_report(y_test, preds, output_dict=True, zero_division=0),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "train_test_split_date": TRAIN_TEST_SPLIT_DATE,
        "horizon_days": HORIZON_DAYS,
    }
    print("\n--- Test performance ---")
    print(f"Accuracy: {metrics['accuracy']:.3f} | ROC-AUC: {metrics['roc_auc']:.3f}")

    importances = pd.Series(model.feature_importances_, index=FEATURE_COLS).sort_values(ascending=False)
    feature_importances = {k: float(v) for k, v in importances.items()}
    print("\nFeature importances:\n", importances)

    test_bt = test.copy()
    test_bt["pred"] = preds
    test_bt["strategy_return"] = np.where(test_bt["pred"] == 1, test_bt["forward_return"], 0.0)
    test_bt = test_bt.sort_values("Date")
    backtest = {
        "dates": test_bt["Date"].dt.strftime("%Y-%m-%d").tolist(),
        "strategy_cum_return": ((1 + test_bt["strategy_return"]).cumprod() - 1).round(4).tolist(),
        "buy_hold_cum_return": ((1 + test_bt["forward_return"]).cumprod() - 1).round(4).tolist(),
    }

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump({"model": model, "feature_cols": FEATURE_COLS}, MODEL_PATH)
    print(f"\nSaved model to {MODEL_PATH}")

    tickers_payload = {}
    for ticker in TICKERS:
        t_full = df_full[df_full["Ticker"] == ticker].sort_values("Date")
        t_valid = t_full.dropna(subset=FEATURE_COLS)
        if t_valid.empty:
            print(f"  Skipping {ticker}: no valid feature rows.")
            continue
        latest = t_valid.iloc[-1]
        latest_X = latest[FEATURE_COLS].to_frame().T.astype(float)
        prob_up = float(model.predict_proba(latest_X)[:, 1][0])

        price_hist = price_df[price_df["Ticker"] == ticker].sort_values("Date").tail(180)
        insider_hist = insider_df[
            (insider_df["Ticker"] == ticker) & (insider_df["aggregated_signal"].isin(["buy", "sell"]))
        ].sort_values("Trade Date", ascending=False).head(50)

        tickers_payload[ticker] = {
            "prediction": "UP" if prob_up > 0.5 else "DOWN",
            "probability_up": round(prob_up, 4),
            "as_of_date": latest["Date"].strftime("%Y-%m-%d"),
            "last_price": float(price_hist["Close"].iloc[-1]) if not price_hist.empty else None,
            "features": {c: (None if pd.isna(latest[c]) else float(latest[c])) for c in FEATURE_COLS},
            "price_history": [
                {"date": d.strftime("%Y-%m-%d"), "close": round(float(c), 2)}
                for d, c in zip(price_hist["Date"], price_hist["Close"])
            ],
            "insider_trades": [
                {
                    "date": row["Trade Date"].strftime("%Y-%m-%d"),
                    "insider": row["Insider Name"],
                    "title": row["Title"],
                    "type": "Buy" if row["is_buy"] == 1 else "Sell",
                    "value": float(row["Value"]),
                }
                for _, row in insider_hist.iterrows()
            ],
        }

    results = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "horizon_days": HORIZON_DAYS,
        "tickers": tickers_payload,
        "model_performance": {
            "metrics": metrics,
            "feature_importances": feature_importances,
            "backtest": backtest,
        },
    }

    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved results to {RESULTS_PATH}")
    print("\nDone. Commit model/model.pkl and data/results.json to your repo.")


if __name__ == "__main__":
    main()
