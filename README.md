# Insider Signal — Stock Direction Predictor

A three-tab website showing a Random Forest model's stock-direction
predictions, trained on real SEC Form 4 insider filings + real price
history for 8 tickers: **BAC, SOFI, INTC, KDP, DAL, COIN, AMZN, SPG**.

- **Dashboard** — current up/down predictions per ticker, with a live
  "Refresh Prices" button.
- **Deep-Dive** — per-ticker price chart, insider filing timeline, and the
  exact feature values behind that ticker's prediction.
- **Model Performance** — accuracy/ROC-AUC, feature importances, and a
  backtest chart vs. buy-and-hold.

## Real results (already trained, already included)

Trained on 8,991 rows (pre-2024-06-01) and tested on 2,704 held-out rows
(2024-06-01 onward — a genuine out-of-time test, no leakage):

- **Accuracy: ~0.50** | **ROC-AUC: ~0.50**

This is an honest, legitimate result — mid-term stock direction is a
genuinely hard prediction problem, and a coin-flip-level result here is
expected, not a bug. Say this plainly in your write-up rather than hiding
it; it's a correct finding, not a failure. Top feature importances were
price-based (30-day and 10-day SMA, volatility) rather than the insider
features, which is also worth noting honestly.

## Project structure

```
insider-website/
├── api/
│   └── refresh.py             ← Vercel Python function (live price refresh)
├── pipeline/
│   └── generate_site_data.py   ← OFFLINE script that produced model.pkl + results.json
├── public/
│   ├── index.html
│   ├── style.css
│   └── app.js
├── data/
│   └── results.json            ← already included, trained on real data
├── model/
│   └── model.pkl                ← already included, trained on real data
├── requirements.txt             ← Python deps Vercel installs for api/refresh.py
├── vercel.json
└── README.md
```

**`model.pkl` and `results.json` are already trained and included** — you
don't need to run anything before your first deploy. `pipeline/generate_site_data.py`
is included for reference and for retraining later (e.g. with a different
ticker list or refreshed data) — it was run once via Google Colab against
the real `ticker_prices.csv` (373 tickers, 2000–2025 daily prices) and
`filings.csv` (5,479 tickers, 2020–2025 aggregated SEC Form 4 filings).
Those two source CSVs are NOT included in this repo (too large for
GitHub/Vercel, ~100MB each) — only their trained output is.

## Deploying (no local IDE needed)

1. Create a new GitHub repo and add all these files — either via GitHub's
   "Add file → Upload files" in the browser, or by pressing `.` on your
   repo page to open **github.dev** (a full code editor in the browser, no
   install) and pasting/creating each file there.
2. Go to [vercel.com](https://vercel.com), sign in with GitHub (free, no
   card required).
3. Click **Add New → Project**, pick your repo, and click **Deploy**.
   Vercel auto-detects `public/` as the static site and `api/refresh.py`
   as a serverless function.
4. That's it — you'll get a live `.vercel.app` URL.

Every time you push a new commit, Vercel redeploys automatically.

## How the "Refresh Prices" button works

- Clicking it calls `/api/refresh`, which runs `api/refresh.py` on Vercel.
- That function loads the **already-trained** model from `model/model.pkl`,
  pulls **fresh live prices** from yfinance for the 8 tracked tickers,
  recomputes the price-based features, and returns new predictions.
- It does **not** retrain the model and does **not** re-fetch insider
  filing data — insider features stay exactly as they were in
  `results.json` until you rerun `pipeline/generate_site_data.py` with
  updated source CSVs. This keeps the button fast enough to finish inside
  Vercel's 10-second free-tier function timeout.
- The button has a 45-second cooldown after each click.
- Refreshed predictions are **not saved back to the repo** — only shown in
  that browser session. Reloading the page goes back to whatever's in
  `results.json`.

## Retraining later (optional)

If you get updated price/filing data and want to retrain:

1. Get `ticker_prices.csv` (columns: `ticker_symbol, date, close`, at
   minimum) and `filings.csv` (columns: `ticker_symbol, earliest_execution_date,
   filing_date, insider_role, aggregated_signal, aggregated_value_usd,
   accession_number`) covering your date range of interest.
2. Run (Google Colab is easiest, no install):
   ```
   pip install pandas scikit-learn joblib numpy
   python pipeline/generate_site_data.py
   ```
3. This overwrites `model/model.pkl` and `data/results.json`. Commit and
   push both.
4. To change the ticker list, edit `TICKERS` near the top of both
   `pipeline/generate_site_data.py` and `api/refresh.py` — keep them
   identical, then rerun and redeploy.

## Known limitations (worth stating in a write-up)

- ROC-AUC ~0.50 means the model is not meaningfully better than a coin
  flip on this feature set/ticker list/horizon — a legitimate, reportable
  finding for a school project, not something to paper over.
- Small ticker universe (8) — results are illustrative, not statistically
  robust at scale.
- Backtest excludes transaction costs and slippage.
- `filings.csv` is aggregated per filing, not per individual insider —
  there's no true insider-name identity, only role (e.g. "CFO"). The
  "cluster buying" feature is approximated as 3+ separate buy filings in a
  window, not 3+ verified distinct people.
- Insider data update cadence is manual (via retraining), not real-time;
  only price data refreshes live via the button.
