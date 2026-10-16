"""Market-data providers. YahooProvider uses yfinance (live); DemoProvider is synthetic/offline."""
from __future__ import annotations

import hashlib
import math
import time
from datetime import date, timedelta

import numpy as np
import pandas as pd

from .pricing import bs_price


class Quote(dict):
    """spot, div_yield, closes (list of daily closes, oldest first)."""


class YahooProvider:
    name = "yahoo"

    def __init__(self, ttl=300):
        self.ttl, self._cache = ttl, {}

    def _cached(self, key, fn):
        hit = self._cache.get(key)
        if hit and time.time() - hit[0] < self.ttl:
            return hit[1]
        val = fn()
        self._cache[key] = (time.time(), val)
        return val

    def risk_free(self):
        def go():
            import yfinance as yf
            try:
                h = yf.Ticker("^IRX").history(period="5d")["Close"].dropna()
                return float(h.iloc[-1]) / 100
            except Exception:
                return 0.043
        return self._cached("rf", go)

    def quote(self, sym):
        def go():
            import yfinance as yf
            t = yf.Ticker(sym)
            h = t.history(period="1y", auto_adjust=False)
            if h.empty:
                raise ValueError(f"No price data for {sym}")
            closes = [float(x) for x in h["Close"].dropna()]
            spot = closes[-1]
            try:
                fi = t.fast_info
                spot = float(fi["last_price"] or spot)
            except Exception:
                pass
            divs = t.dividends
            dy = 0.0
            if len(divs):
                recent = divs[divs.index > divs.index[-1] - pd.Timedelta(days=365)]
                dy = float(recent.sum()) / spot
            return Quote(spot=spot, div_yield=dy, closes=closes)
        return self._cached(("q", sym), go)

    def expiries(self, sym):
        import yfinance as yf
        return self._cached(("e", sym), lambda: list(yf.Ticker(sym).options))

    def chain(self, sym, expiry):
        """Returns {'calls': [rows], 'puts': [rows]}; row keys: strike,bid,ask,last,iv,volume,oi."""
        def go():
            import yfinance as yf
            c = yf.Ticker(sym).option_chain(expiry)
            def rows(df):
                return [dict(strike=float(r.strike), bid=_f(r.bid), ask=_f(r.ask), last=_f(r.lastPrice),
                             iv=_f(r.impliedVolatility), volume=_f(r.volume), oi=_f(r.openInterest))
                        for r in df.itertuples()]
            return {"calls": rows(c.calls), "puts": rows(c.puts)}
        return self._cached(("c", sym, expiry), go)


def _f(x):
    try:
        x = float(x)
        return None if math.isnan(x) else x
    except (TypeError, ValueError):
        return None


class DemoProvider:
    """Deterministic synthetic market so the app runs with no network."""
    name = "demo"

    def _seed(self, sym):
        return int(hashlib.md5(sym.encode()).hexdigest()[:8], 16)

    def risk_free(self):
        return 0.043

    def quote(self, sym):
        rng = np.random.default_rng(self._seed(sym))
        spot0 = 20 + self._seed(sym) % 480
        vol = 0.18 + (self._seed(sym) % 30) / 100
        rets = rng.normal(0.0004, vol / math.sqrt(252), 252)
        closes = list(spot0 * np.exp(np.cumsum(rets)))
        return Quote(spot=closes[-1], div_yield=(self._seed(sym) % 40) / 1000, closes=closes)

    def expiries(self, sym):
        d, out = date.today(), []
        for w in (1, 2, 3, 4, 6, 8, 12, 16, 26):
            out.append((d + timedelta(weeks=w)).isoformat())
        return out

    def chain(self, sym, expiry):
        q = self.quote(sym)
        S, r, dy = q["spot"], self.risk_free(), q["div_yield"]
        T = max((date.fromisoformat(expiry) - date.today()).days, 1) / 365
        rng = np.random.default_rng(self._seed(sym + expiry))
        base = 0.18 + (self._seed(sym) % 30) / 100 + ((self._seed(sym) >> 5) % 15 - 6) / 100
        step = 1 if S < 50 else 2.5 if S < 200 else 5
        ks = [round(S / step) * step + step * i for i in range(-12, 13)]
        out = {"calls": [], "puts": []}
        for kind in ("call", "put"):
            for K in ks:
                if K <= 0:
                    continue
                skew = -0.25 * math.log(K / S)  # put skew
                iv = max(0.08, base + skew + rng.normal(0, 0.005))
                mid = bs_price(S, K, T, r, dy, iv, kind)
                half = max(0.01, mid * 0.03)
                out[kind + "s"].append(dict(strike=float(K), bid=round(max(mid - half, 0), 2),
                                            ask=round(mid + half, 2), last=round(mid, 2), iv=iv,
                                            volume=int(rng.integers(0, 900)), oi=int(rng.integers(10, 5000))))
        return out


def get_provider(name="yahoo"):
    return DemoProvider() if name == "demo" else YahooProvider()
