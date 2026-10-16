"""Volatility estimators used as the 'fair' volatility input of the pricing model."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

TRADING_DAYS = 252

VOL_METHODS = ("historical", "ewma", "blend")


def _log_returns(closes: pd.Series) -> pd.Series:
    closes = pd.Series(closes, dtype=float).dropna()
    closes = closes[closes > 0]
    return np.log(closes / closes.shift(1)).dropna()


def historical_vol(closes: pd.Series, window: int = 60) -> float | None:
    """Annualised close-to-close volatility over the last `window` trading days."""
    rets = _log_returns(closes).tail(window)
    if len(rets) < 10:
        return None
    return float(rets.std(ddof=1) * math.sqrt(TRADING_DAYS))


def ewma_vol(closes: pd.Series, lam: float = 0.94) -> float | None:
    """RiskMetrics exponentially weighted volatility (annualised)."""
    rets = _log_returns(closes)
    if len(rets) < 10:
        return None
    var = float(rets.iloc[:20].var(ddof=1))
    for r in rets.iloc[20:]:
        var = lam * var + (1 - lam) * r * r
    return float(math.sqrt(var * TRADING_DAYS))


def estimate_vol(closes: pd.Series, method: str = "historical", window: int = 60) -> float | None:
    """Estimate the volatility to feed the fair-value model.

    - historical: close-to-close stdev over `window` days
    - ewma: RiskMetrics EWMA (lambda 0.94)
    - blend: average of short (window) and long (1 year) historical vol, a simple
      mean-reversion adjustment that avoids over-reacting to recent calm/stress.
    """
    method = method.lower()
    if method == "historical":
        return historical_vol(closes, window)
    if method == "ewma":
        return ewma_vol(closes)
    if method == "blend":
        short = historical_vol(closes, window)
        long = historical_vol(closes, TRADING_DAYS)
        vals = [v for v in (short, long) if v]
        return float(sum(vals) / len(vals)) if vals else None
    raise ValueError(f"unknown vol method {method!r}; expected one of {VOL_METHODS}")
