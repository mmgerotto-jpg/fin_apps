"""Market conventions for B3 (Brazil) and US listed stocks."""

from __future__ import annotations

import datetime as dt
import math
import re

BR = "BR"
US = "US"

CURRENCY = {BR: "BRL", US: "USD"}
CURRENCY_SYMBOL = {BR: "R$", US: "US$"}

# PETR4, VALE3, BOVA11, AAPL34 (BDR), PETR4F (fractional market), PETR4.SA
_B3_RE = re.compile(r"^[A-Z]{4}\d{1,2}F?$")


def normalize_ticker(symbol: str) -> str:
    s = str(symbol).strip().upper().rstrip("*")
    if s.endswith(".SA"):
        s = s[:-3]
    if _B3_RE.match(s) and s.endswith("F"):
        s = s[:-1]  # fractional lot ticker trades the same stock
    return s.replace("/", ".")


def detect_market(symbol: str) -> str:
    raw = str(symbol).strip().upper()
    if raw.endswith(".SA"):
        return BR
    return BR if _B3_RE.match(normalize_ticker(raw)) else US


def yahoo_symbol(symbol: str, market: str | None = None) -> str:
    sym = normalize_ticker(symbol)
    market = market or detect_market(sym)
    if market == BR:
        return f"{sym}.SA"
    return sym.replace(".", "-")  # BRK.B -> BRK-B


def third_friday(year: int, month: int) -> dt.date:
    first = dt.date(year, month, 1)
    offset = (4 - first.weekday()) % 7
    return first + dt.timedelta(days=offset + 14)


def monthly_expirations(today: dt.date, count: int = 12) -> list[dt.date]:
    """Standard monthly expiries (third Friday) – the B3 equity option calendar
    since 2021 and the US monthly cycle."""
    out: list[dt.date] = []
    year, month = today.year, today.month
    while len(out) < count:
        d = third_friday(year, month)
        if d > today:
            out.append(d)
        month += 1
        if month > 12:
            year, month = year + 1, 1
    return out


def strike_step(spot: float) -> float:
    if spot < 10:
        return 0.25
    if spot < 50:
        return 0.5
    if spot < 100:
        return 1.0
    if spot < 250:
        return 2.5
    if spot < 500:
        return 5.0
    return 10.0


def strike_grid(spot: float, low: float, high: float) -> list[float]:
    """Round-number strikes between low and high (used when no option chain is available)."""
    step = strike_step(spot)
    start = math.ceil(low / step - 1e-9) * step
    out = []
    k = start
    while k <= high + 1e-9:
        out.append(round(k, 2))
        k += step
    return out
