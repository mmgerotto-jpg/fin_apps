"""Market data providers.

`YahooProvider` pulls live quotes, price history, dividends and option chains via
yfinance (free, no API key; quotes may be delayed ~15 min). `DemoProvider`
generates a deterministic synthetic market so the app can be tried offline.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import math
from typing import Protocol

import numpy as np
import pandas as pd

from .pricing import bs_price

CHAIN_COLUMNS = ["strike", "bid", "ask", "last", "volume", "open_interest", "market_iv", "contract"]
DEFAULT_RISK_FREE = 0.04


class MarketDataProvider(Protocol):
    def spot(self, symbol: str) -> float: ...

    def history(self, symbol: str, period_days: int = 400) -> pd.Series: ...

    def dividend_yield(self, symbol: str) -> float: ...

    def expirations(self, symbol: str) -> list[dt.date]: ...

    def chain(self, symbol: str, expiry: dt.date, kind: str) -> pd.DataFrame: ...

    def risk_free_rate(self) -> float: ...


def _today() -> dt.date:
    return dt.date.today()


class YahooProvider:
    """Live data from Yahoo Finance through yfinance."""

    def __init__(self):
        import yfinance as yf  # imported lazily so the rest of the app works without it

        self._yf = yf
        self._tickers: dict[str, object] = {}

    def _t(self, symbol: str):
        sym = symbol.replace(".", "-")  # BRK.B -> BRK-B on Yahoo
        if sym not in self._tickers:
            self._tickers[sym] = self._yf.Ticker(sym)
        return self._tickers[sym]

    def spot(self, symbol: str) -> float:
        t = self._t(symbol)
        try:
            price = t.fast_info["lastPrice"]
            if price and math.isfinite(price):
                return float(price)
        except Exception:
            pass
        closes = self.history(symbol, 10)
        if closes.empty:
            raise LookupError(f"no price data for {symbol}")
        return float(closes.iloc[-1])

    def history(self, symbol: str, period_days: int = 400) -> pd.Series:
        start = _today() - dt.timedelta(days=period_days)
        hist = self._t(symbol).history(start=start.isoformat(), auto_adjust=False)
        if hist is None or hist.empty:
            return pd.Series(dtype=float)
        return hist["Close"].dropna()

    def dividend_yield(self, symbol: str) -> float:
        """Trailing-12-month dividends divided by spot (continuous-yield approximation)."""
        try:
            divs = self._t(symbol).dividends
            if divs is None or divs.empty:
                return 0.0
            cutoff = pd.Timestamp.now(tz=divs.index.tz) - pd.Timedelta(days=365)
            ttm = float(divs[divs.index >= cutoff].sum())
            spot = self.spot(symbol)
            return ttm / spot if spot > 0 else 0.0
        except Exception:
            return 0.0

    def expirations(self, symbol: str) -> list[dt.date]:
        return [dt.date.fromisoformat(d) for d in self._t(symbol).options]

    def chain(self, symbol: str, expiry: dt.date, kind: str) -> pd.DataFrame:
        oc = self._t(symbol).option_chain(expiry.isoformat())
        raw = oc.calls if kind == "call" else oc.puts
        if raw is None or raw.empty:
            return pd.DataFrame(columns=CHAIN_COLUMNS)
        df = pd.DataFrame(
            {
                "strike": raw["strike"].astype(float),
                "bid": raw.get("bid", 0.0),
                "ask": raw.get("ask", 0.0),
                "last": raw.get("lastPrice", np.nan),
                "volume": raw.get("volume", 0),
                "open_interest": raw.get("openInterest", 0),
                "market_iv": raw.get("impliedVolatility", np.nan),
                "contract": raw.get("contractSymbol", ""),
            }
        )
        return df.fillna({"bid": 0.0, "ask": 0.0, "volume": 0, "open_interest": 0}).reset_index(drop=True)

    def risk_free_rate(self) -> float:
        """13-week T-bill yield (^IRX) as the risk-free rate."""
        try:
            closes = self._t("^IRX").history(period="1mo")["Close"].dropna()
            if not closes.empty:
                return float(closes.iloc[-1]) / 100.0
        except Exception:
            pass
        return DEFAULT_RISK_FREE


class DemoProvider:
    """Deterministic synthetic market (prices, chains) for offline use and tests.

    Market option prices are Black-Scholes prices at an implied vol that differs
    from the realised vol of the generated history (with a volatility skew), so
    the comparison against fair value shows both over- and under-priced options.
    """

    BASE_PRICES = {
        "AAPL": 230.0, "MSFT": 430.0, "NVDA": 125.0, "GOOGL": 170.0, "AMZN": 190.0,
        "META": 560.0, "TSLA": 240.0, "JPM": 215.0, "KO": 68.0, "JNJ": 160.0,
        "XOM": 115.0, "COST": 880.0, "V": 280.0, "PG": 170.0, "SPY": 570.0,
    }

    def __init__(self, today: dt.date | None = None, rate: float = DEFAULT_RISK_FREE):
        self.today = today or _today()
        self.rate = rate

    def _seed(self, symbol: str) -> int:
        return int(hashlib.sha256(symbol.encode()).hexdigest()[:8], 16)

    def _params(self, symbol: str) -> tuple[float, float, float]:
        rng = np.random.default_rng(self._seed(symbol))
        base = self.BASE_PRICES.get(symbol, float(rng.uniform(20, 400)))
        vol = float(rng.uniform(0.18, 0.55))
        div = float(rng.choice([0.0, 0.0, 0.006, 0.015, 0.03]))
        return base, vol, div

    def history(self, symbol: str, period_days: int = 400) -> pd.Series:
        base, vol, _ = self._params(symbol)
        rng = np.random.default_rng(self._seed(symbol) + 1)
        n = int(period_days * 252 / 365)
        rets = rng.normal(0.0, vol / math.sqrt(252), n)
        path = np.exp(np.cumsum(rets))
        prices = base * path / path[-1]  # ends exactly at the base price
        idx = pd.bdate_range(end=pd.Timestamp(self.today), periods=n)
        return pd.Series(prices, index=idx, name="Close")

    def spot(self, symbol: str) -> float:
        return float(self.history(symbol, 30).iloc[-1])

    def dividend_yield(self, symbol: str) -> float:
        return self._params(symbol)[2]

    def expirations(self, symbol: str) -> list[dt.date]:
        out = []
        d = self.today + dt.timedelta(days=(4 - self.today.weekday()) % 7 or 7)  # next Friday
        while len(out) < 12:
            out.append(d)
            d += dt.timedelta(days=7)
        return out

    def chain(self, symbol: str, expiry: dt.date, kind: str) -> pd.DataFrame:
        spot = self.spot(symbol)
        _, vol, q = self._params(symbol)
        T = max((expiry - self.today).days, 1) / 365.0
        step = 1.0 if spot < 100 else (2.5 if spot < 300 else 5.0)
        strikes = np.arange(math.floor(spot * 0.7 / step) * step, spot * 1.3 + step, step)
        rng = np.random.default_rng(self._seed(f"{symbol}{expiry}{kind}"))
        iv_level = vol * float(rng.uniform(0.85, 1.25))
        rows = []
        for K in strikes:
            moneyness = math.log(K / spot)
            iv = max(iv_level - 0.25 * moneyness + 0.6 * moneyness**2, 0.05)  # skew + smile
            mid = bs_price(spot, K, T, self.rate, iv, kind, q)
            if mid < 0.01:
                continue
            half_spread = max(0.01, mid * 0.03)
            bid = round(max(mid - half_spread, 0.0), 2)
            ask = round(mid + half_spread, 2)
            rows.append(
                {
                    "strike": float(K),
                    "bid": bid,
                    "ask": ask,
                    "last": round(mid, 2),
                    "volume": int(rng.integers(0, 2000)),
                    "open_interest": int(rng.integers(10, 20000)),
                    "market_iv": iv,
                    "contract": f"{symbol}{expiry:%y%m%d}{kind[0].upper()}{int(K * 1000):08d}",
                }
            )
        return pd.DataFrame(rows, columns=CHAIN_COLUMNS)

    def risk_free_rate(self) -> float:
        return self.rate
