# fin_apps
Financial apps for personal use

## Options fair value (`app.py`, `options_app/`)

A local app that prices options with a model and compares the result with market prices:

- **Covered calls on your portfolio.** Load your brokerage position exports. For each holding the app pulls the call
  chain, computes a fair value for each out-of-the-money strike and compares it with the market mid price. Existing
  short calls in the statement reduce the contracts available to sell.
- **Cash-secured puts on stocks you plan to buy.** Enter a watchlist (symbol, optional target price, shares). For each
  put strike below spot you get fair value vs market, the effective purchase price if assigned (strike − premium),
  whether it meets your target, and the cash required.
- **Single-option calculator** with Greeks and implied volatility.

### Setup

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py            # opens http://localhost:8501
```

Pick **Demo (offline synthetic)** in the sidebar to try it without a network connection, and tick
*Use the sample statements* to load `sample_data/`.

### Statements

Export positions from your broker as CSV (preferred) or XLSX; PDF statements are read best-effort from their tables.
The parser finds the header row on its own, so preamble lines and footers are fine. It recognises common column names
(`Symbol`/`Ticker`, `Quantity`/`Shares`, `Cost Basis`/`Average Cost`, …), as used by Schwab, Fidelity, IBKR, E*TRADE,
Vanguard and generic exports. Cash, money-market and total rows are skipped. Option rows are listed separately. You
can upload several files: holdings are merged with a quantity-weighted cost basis. The holdings table can be edited
before you run the analysis.

Keep real statements in `statements/`, which is git-ignored.

### How fair value is computed

| Input | Source |
|---|---|
| Spot, option chain (bid/ask) | Yahoo Finance via `yfinance` (free, ~15 min delayed) |
| Volatility | Estimated from price history: `blend` (default, mean of 60-day and 1-year realised vol), `historical`, or `ewma`. Can be overridden. |
| Dividend yield | Trailing-12-month dividends ÷ spot |
| Risk-free rate | 13-week T-bill (`^IRX`), can be overridden |
| Model | Cox-Ross-Rubinstein binomial tree, 300 steps, American exercise. Black-Scholes is used for Greeks, probabilities and implied vol. |

**Edge = market mid − fair value.** When it is positive, the market pays more than the model value. That favours
selling (covered calls, cash-secured puts). The *signal* column flags options whose edge exceeds the threshold set in
the sidebar (5% by default). Comparing market price with a realised-volatility model amounts to comparing implied vol
(`market_iv`) with realised vol (`model_vol`), and both are shown.

This is model output, not investment advice. Quotes can be stale outside market hours, and wide bid/ask spreads make
the mid price unreliable.

### Command line

```bash
python -m options_app.cli calls --statement statements/*.csv --best
python -m options_app.cli puts --watch "GOOGL:160, JPM:200" --out puts.csv
python -m options_app.cli puts --watch-file sample_data/watchlist.csv --demo
python -m options_app.cli price --spot 230 --strike 250 --days 30 --vol 0.3 --kind call --market 2.4
```

Run `python -m options_app.cli calls -h` to see the filters (DTE range, % OTM, vol method, rate, edge threshold).

### Tests

```bash
python -m pytest
```
