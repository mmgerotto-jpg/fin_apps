# fin_apps
Financial apps for personal use

## Options Fair Value (`optionsapp/`)

Local web app that prices options on your portfolio and compares fair value to the market:

- **Covered calls on current holdings** — import a brokerage statement (CSV / XLSX / PDF), and for each
  stock with ≥100 shares the app lists calls in your expiry/OTM window with bid/ask/mid, **fair value**,
  edge (market − fair), market IV, delta, annualized yield, return if called away, and a verdict
  (`RICH (sell)` / `FAIR` / `CHEAP (avoid selling)`).
- **Cash-secured puts for stocks you plan to buy** — enter symbol, shares and an optional target buy
  price. The app prices puts below spot, highlights the strike nearest your target, and shows effective
  buy price (strike − premium), discount vs spot, approx. assignment probability, yield and cash needed.
  If no option chain is available it still computes theoretical fair prices.
- **Calculator** — single option: European (Black-Scholes-Merton) and American (binomial) price, Greeks,
  implied vol from a market price.

### Model
- Fair value = Cox-Ross-Rubinstein binomial tree with early exercise (US equity options are American),
  continuous dividend yield (trailing 12-month dividends / spot), risk-free rate from 13-week T-bill (^IRX).
- Fair volatility = historical (realized) vol; default blend 30% HV20 + 40% HV60 + 30% HV120. Can pick
  a single window or set a manual override in Settings. Market IV is shown next to it — the comparison is
  essentially *market implied vol vs. your realized-vol estimate*.

### Run
```bash
pip install -r requirements.txt
python -m optionsapp.app            # live data from Yahoo Finance (yfinance)
python -m optionsapp.app --demo     # offline synthetic data
```
Open http://127.0.0.1:5000. Holdings, watchlist and settings are saved to `data/portfolio.json`
(git-ignored); statements never leave your machine. Market quotes come from Yahoo Finance and may be
delayed — use for analysis, not order execution.

### Statements
The importer finds a header row with a symbol/ticker column and a quantity/shares column (Fidelity,
Schwab, IBKR, Vanguard-style exports work), sums duplicate lots, skips cash/money-market and option
rows. PDFs are read via table extraction with a text-line fallback. Try `samples/sample_statement.csv`.
If a statement doesn't parse, export positions as CSV or edit holdings manually in the Portfolio tab.

### Tests
```bash
python -m pytest -q
```
