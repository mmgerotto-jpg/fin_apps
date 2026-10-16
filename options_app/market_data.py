"""Market data providers.

`YahooProvider` pulls prices, history and dividends for US and B3 stocks via
yfinance (free, delayed). Option chains (used only as a market *reference* next
to the fair price) come from Yahoo for US stocks; Yahoo has no B3 option chains,
so Brazilian strikes are generated on a round-number grid over the B3 monthly
calendar. The risk-free rate is the 13-week T-bill (US) and the Selic target
from the Banco Central SGS API (BR).

`DemoProvider` generates a deterministic synthetic market so the app can be
tried offline.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import math
import urllib.request
from typing import Protocol

import numpy as np
import pandas as pd

from .markets import BR, US, detect_market, monthly_expirations, normalize_ticker, yahoo_symbol
from .opcoes_net import OpcoesNetClient
from .pricing import bs_price

CHAIN_COLUMNS = ["strike", "bid", "ask", "last", "volume", "open_interest", "market_iv", "contract"]
DEFAULT_RATES = {US: 0.04, BR: 0.14}
BCB_SELIC_URL = "https://api.bcb.gov.br/dados/serie/bcdata.sgs.432/dados/ultimos/1?formato=json"


class MarketDataProvider(Protocol):
    def spot(self, symbol: str) -> float: ...

    def history(self, symbol: str, period_days: int = 400) -> pd.Series: ...

    def dividend_yield(self, symbol: str) -> float: ...

    def expirations(self, symbol: str) -> list[dt.date]: ...

    def chain(self, symbol: str, expiry: dt.date, kind: str) -> pd.DataFrame | None: ...

    def risk_free_rate(self, market: str) -> float: ...


def annual_to_continuous(rate: float) -> float:
    """Convert an annually compounded rate (e.g. Selic 14.25% a.a.) to continuous."""
    return math.log1p(rate)


def fetch_selic(timeout: float = 5.0) -> float | None:
    """Selic target (annual, compounded) from Banco Central SGS series 432."""
    try:
        with urllib.request.urlopen(BCB_SELIC_URL, timeout=timeout) as resp:
            data = json.loads(resp.read().decode())
        return float(str(data[-1]["valor"]).replace(",", ".")) / 100.0
    except Exception:
        return None


class YahooProvider:
    """Live data from Yahoo Finance through yfinance; B3 option quotes from opcoes.net.br."""

    def __init__(self, b3_options: OpcoesNetClient | None = None):
        import yfinance as yf  # imported lazily so the rest of the app works without it

        self._yf = yf
        self._tickers: dict[str, object] = {}
        self._rates: dict[str, float] = {}
        self.b3 = b3_options if b3_options is not None else OpcoesNetClient()
        self.notes: list[str] = []  # data-source problems to show the user
        self._b3_failed: set[str] = set()

    def reset_notes(self) -> None:
        """Clear data-source notes and retry sources that failed earlier."""
        self.notes = []
        self._b3_failed.clear()

    def _b3_note(self, ticker: str, exc: Exception) -> None:
        if ticker not in self._b3_failed:
            self._b3_failed.add(ticker)
            self.notes.append(f"{ticker}: sem cotações do opcoes.net.br ({exc}); usando strikes teóricos")

    def _t(self, symbol: str):
        ysym = symbol if symbol.startswith("^") else yahoo_symbol(symbol)
        if ysym not in self._tickers:
            self._tickers[ysym] = self._yf.Ticker(ysym)
        return self._tickers[ysym]

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
            raise LookupError(f"sem cotação para {symbol}")
        return float(closes.iloc[-1])

    def history(self, symbol: str, period_days: int = 400) -> pd.Series:
        start = dt.date.today() - dt.timedelta(days=period_days)
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
        if detect_market(symbol) == BR:
            ticker = normalize_ticker(symbol)
            if ticker not in self._b3_failed:
                try:
                    exps = [e for e in self.b3.expirations(ticker) if e > dt.date.today()]
                    if exps:
                        return exps
                    raise LookupError("nenhum vencimento listado")
                except Exception as exc:
                    self._b3_note(ticker, exc)
            return monthly_expirations(dt.date.today())
        return [dt.date.fromisoformat(d) for d in self._t(symbol).options]

    def chain(self, symbol: str, expiry: dt.date, kind: str) -> pd.DataFrame | None:
        if detect_market(symbol) == BR:
            ticker = normalize_ticker(symbol)
            if ticker in self._b3_failed:
                return None
            try:
                return self.b3.chain(ticker, expiry, kind)
            except Exception as exc:
                self._b3_note(ticker, exc)
                return None
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

    def risk_free_rate(self, market: str = US) -> float:
        """Continuously compounded risk-free rate for the market."""
        if market in self._rates:
            return self._rates[market]
        rate = None
        if market == BR:
            selic = fetch_selic()
            rate = annual_to_continuous(selic) if selic is not None else None
        else:
            try:
                closes = self._t("^IRX").history(period="1mo")["Close"].dropna()
                if not closes.empty:
                    rate = float(closes.iloc[-1]) / 100.0
            except Exception:
                pass
        if rate is None:
            rate = annual_to_continuous(DEFAULT_RATES[market]) if market == BR else DEFAULT_RATES[market]
        self._rates[market] = rate
        return rate


class DemoProvider:
    """Deterministic synthetic market (prices, chains) for offline use and tests.

    Mirrors the live provider: US stocks get a quoted option chain, B3 stocks
    don't (strikes come from a grid).
    """

    BASE_PRICES = {
        "AAPL": 230.0, "MSFT": 430.0, "NVDA": 125.0, "GOOGL": 170.0, "AMZN": 190.0, "PLTR": 45.0,
        "META": 560.0, "TSLA": 240.0, "JPM": 215.0, "KO": 68.0, "SPY": 570.0,
        "PETR4": 37.5, "VALE3": 61.0, "ITUB4": 35.0, "BBAS3": 27.0, "BOVA11": 125.0, "WEGE3": 52.0,
    }

    def __init__(self, today: dt.date | None = None, rates: dict[str, float] | None = None):
        self.today = today or dt.date.today()
        self.rates = rates or {US: 0.04, BR: annual_to_continuous(0.14)}

    def _seed(self, symbol: str) -> int:
        return int(hashlib.sha256(symbol.encode()).hexdigest()[:8], 16)

    def _params(self, symbol: str) -> tuple[float, float, float]:
        rng = np.random.default_rng(self._seed(symbol))
        base = self.BASE_PRICES.get(symbol, float(rng.uniform(20, 400)))
        vol = float(rng.uniform(0.18, 0.55))
        div = float(rng.choice([0.0, 0.0, 0.006, 0.015, 0.03]))
        if detect_market(symbol) == BR:
            div = float(rng.choice([0.04, 0.06, 0.09]))
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
        if detect_market(symbol) == BR:
            return monthly_expirations(self.today)
        out = []
        d = self.today + dt.timedelta(days=(4 - self.today.weekday()) % 7 or 7)  # next Friday
        while len(out) < 12:
            out.append(d)
            d += dt.timedelta(days=7)
        return out

    def chain(self, symbol: str, expiry: dt.date, kind: str) -> pd.DataFrame | None:
        if detect_market(symbol) == BR:
            return None
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
            mid = bs_price(spot, K, T, self.rates[US], iv, kind, q)
            if mid < 0.01:
                continue
            half_spread = max(0.01, mid * 0.03)
            rows.append(
                {
                    "strike": float(K),
                    "bid": round(max(mid - half_spread, 0.0), 2),
                    "ask": round(mid + half_spread, 2),
                    "last": round(mid, 2),
                    "volume": int(rng.integers(0, 2000)),
                    "open_interest": int(rng.integers(10, 20000)),
                    "market_iv": iv,
                    "contract": f"{symbol}{expiry:%y%m%d}{kind[0].upper()}{int(K * 1000):08d}",
                }
            )
        return pd.DataFrame(rows, columns=CHAIN_COLUMNS)

    def risk_free_rate(self, market: str = US) -> float:
        return self.rates[market]
