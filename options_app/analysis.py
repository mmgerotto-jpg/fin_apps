"""Fair prices for option *selling* strategies on a long-term portfolio.

- Covered calls on stocks you already hold (consolidated from your statements).
- Cash-secured puts on stocks you want to buy: you get paid to wait for a lower
  entry price; if assigned, your cost is strike - premium.

The fair price is the model value of the option (American binomial tree,
historical volatility, dividend yield, local risk-free rate). It is meant as the
reference for the limit price of your order and to compare strikes/expiries by
yield vs. probability of exercise - not to hunt mispricings. When a quoted chain
exists (US stocks via Yahoo) the market bid/ask is shown alongside for reference.
"""

from __future__ import annotations

import datetime as dt
import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .market_data import MarketDataProvider, annual_to_continuous
from .markets import CURRENCY, detect_market, normalize_ticker, strike_grid
from .portfolio import OPTIONABLE_TYPES, Holding
from .pricing import CALL, PUT, bs_greeks, fair_value, implied_vol
from .volatility import estimate_vol

LOT = 100  # standard lot: 100 shares per contract on both B3 and US exchanges


@dataclass
class Settings:
    min_dte: int = 15
    max_dte: int = 75
    max_expiries: int = 3
    call_otm_min: float = 0.0  # call strikes from spot*(1+min) ...
    call_otm_max: float = 0.20  # ... to spot*(1+max)
    put_otm_min: float = 0.0  # put strikes from spot*(1-max) to spot*(1-min)
    put_otm_max: float = 0.20
    vol_method: str = "blend"
    vol_window: int = 60
    vol_override: dict[str, float] = field(default_factory=dict)  # ticker -> sigma
    rate_override: dict[str, float] = field(default_factory=dict)  # market -> annual rate (e.g. 0.1425)
    american: bool = True
    max_prob_exercise: float = 0.30  # for suggestions
    calls_above_avg_price: bool = True  # suggestions only with strike >= preço médio


@dataclass
class StockContext:
    ticker: str
    market: str
    spot: float
    sigma: float
    q: float
    r: float


def year_fraction(expiry: dt.date, today: dt.date) -> float:
    # Options expire at the close; count the expiry day itself.
    return max((expiry - today).days + 1, 1) / 365.0


def mid_price(bid, ask, last) -> float | None:
    bid = float(bid or 0)
    ask = float(ask or 0)
    if bid > 0 and ask >= bid:
        return (bid + ask) / 2.0
    if last is not None and not (isinstance(last, float) and math.isnan(last)) and float(last) > 0:
        return float(last)
    return None


def stock_context(ticker: str, provider: MarketDataProvider, settings: Settings, market: str | None = None) -> StockContext:
    ticker = normalize_ticker(ticker)
    market = market or detect_market(ticker)
    spot = provider.spot(ticker)
    sigma = settings.vol_override.get(ticker)
    if not sigma:
        sigma = estimate_vol(provider.history(ticker, 400), settings.vol_method, settings.vol_window)
        if not sigma:
            raise LookupError("histórico de preços insuficiente para estimar a volatilidade")
    if market in settings.rate_override:
        r = annual_to_continuous(settings.rate_override[market])
    else:
        r = provider.risk_free_rate(market)
    return StockContext(ticker, market, spot, sigma, provider.dividend_yield(ticker), r)


def _expiries(provider: MarketDataProvider, ticker: str, settings: Settings, today: dt.date) -> list[dt.date]:
    exps = [e for e in provider.expirations(ticker) if settings.min_dte <= (e - today).days <= settings.max_dte]
    return sorted(exps)[: settings.max_expiries]


def price_grid(
    ctx: StockContext, kind: str, provider: MarketDataProvider, settings: Settings, today: dt.date
) -> list[dict]:
    """Fair price for each strike/expiry in the configured range."""
    if kind == CALL:
        lo, hi = ctx.spot * (1 + settings.call_otm_min), ctx.spot * (1 + settings.call_otm_max)
    else:
        lo, hi = ctx.spot * (1 - settings.put_otm_max), ctx.spot * (1 - settings.put_otm_min)
    rows = []
    for expiry in _expiries(provider, ctx.ticker, settings, today):
        T = year_fraction(expiry, today)
        chain = provider.chain(ctx.ticker, expiry, kind)
        if chain is not None and not chain.empty:
            listed = chain[(chain["strike"] >= lo) & (chain["strike"] <= hi)]
            quotes = {float(r.strike): r for r in listed.itertuples(index=False)}
            strikes, source = sorted(quotes), "listado"
        else:
            quotes, strikes, source = {}, strike_grid(ctx.spot, lo, hi), "grade"
        for K in strikes:
            fair = fair_value(ctx.spot, K, T, ctx.r, ctx.sigma, kind, ctx.q, settings.american)
            g = bs_greeks(ctx.spot, K, T, ctx.r, ctx.sigma, kind, ctx.q)
            q = quotes.get(K)
            mkt = mid_price(q.bid, q.ask, q.last) if q is not None else None
            mkt_iv = implied_vol(mkt, ctx.spot, K, T, ctx.r, kind, ctx.q) if mkt else None
            rows.append(
                {
                    "ticker": ctx.ticker,
                    "market": ctx.market,
                    "currency": CURRENCY[ctx.market],
                    "expiry": expiry,
                    "dte": (expiry - today).days,
                    "strike": K,
                    "spot": ctx.spot,
                    "fair_price": fair,
                    "fair_per_contract": fair * LOT,
                    "prob_exercise": g.prob_itm,
                    "delta": g.delta,
                    "theta_day": g.theta,
                    "model_vol": ctx.sigma,
                    "div_yield": ctx.q,
                    "rate": ctx.r,
                    "market_bid": float(q.bid) if q is not None else np.nan,
                    "market_ask": float(q.ask) if q is not None else np.nan,
                    "market_mid": mkt if mkt is not None else np.nan,
                    "market_iv": mkt_iv if mkt_iv is not None else np.nan,
                    "strike_source": source,
                }
            )
    return rows


def call_opportunities(
    holdings: list[Holding],
    provider: MarketDataProvider,
    settings: Settings | None = None,
    today: dt.date | None = None,
    committed_shares: dict[str, float] | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    """Covered-call fair prices for every optionable holding with at least one free lot.

    `committed_shares` (ticker -> shares) are already covering calls you sold and are
    excluded. Returns (table, warnings).
    """
    settings = settings or Settings()
    today = today or dt.date.today()
    committed_shares = committed_shares or {}
    rows, warnings = [], []
    for h in holdings:
        if h.asset_type not in OPTIONABLE_TYPES:
            continue
        free = h.quantity - committed_shares.get(h.ticker, 0.0) - h.lent
        contracts = int(free // LOT)
        if contracts < 1:
            lent = f" ({h.lent:g} em aluguel)" if h.lent else ""
            warnings.append(f"{h.ticker}: {free:g} ações livres{lent}, menos que 1 lote de {LOT}")
            continue
        try:
            ctx = stock_context(h.ticker, provider, settings, h.market)
            for row in price_grid(ctx, CALL, provider, settings, today):
                K, prem, dte = row["strike"], row["fair_price"], max(row["dte"], 1)
                row.update(
                    {
                        "shares": h.quantity,
                        "free_shares": free,
                        "contracts": contracts,
                        "premium_total": prem * LOT * contracts,
                        "premium_pct": prem / ctx.spot,
                        "annualized_yield": prem / ctx.spot * 365 / dte,
                        "upside_to_strike": K / ctx.spot - 1,
                        "avg_price": h.avg_price,
                        "return_if_exercised": (K + prem) / h.avg_price - 1 if h.avg_price else np.nan,
                        "strike_above_avg": (K >= h.avg_price) if h.avg_price else None,
                    }
                )
                rows.append(row)
        except Exception as exc:  # keep going for the other tickers
            warnings.append(f"{h.ticker}: {exc}")
    return _finish(rows), warnings


def put_opportunities(
    ticker: str,
    provider: MarketDataProvider,
    settings: Settings | None = None,
    today: dt.date | None = None,
    target_price: float | None = None,
    shares: float = LOT,
) -> pd.DataFrame:
    """Cash-secured put fair prices for a stock you want to buy (B3 or US ticker)."""
    settings = settings or Settings()
    today = today or dt.date.today()
    ctx = stock_context(ticker, provider, settings)
    contracts = max(int(shares // LOT), 1)
    rows = []
    for row in price_grid(ctx, PUT, provider, settings, today):
        K, prem, dte = row["strike"], row["fair_price"], max(row["dte"], 1)
        effective = K - prem
        row.update(
            {
                "contracts": contracts,
                "cash_required": K * LOT * contracts,
                "premium_total": prem * LOT * contracts,
                "premium_pct": prem / K,
                "annualized_yield": prem / K * 365 / dte,
                "effective_price": effective,
                "discount_to_spot": 1 - effective / ctx.spot,
                "target_price": target_price,
                "meets_target": (effective <= target_price) if target_price else None,
            }
        )
        rows.append(row)
    return _finish(rows)


def _finish(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return df.sort_values(["ticker", "expiry", "strike"]).reset_index(drop=True)


def suggestions(df: pd.DataFrame, settings: Settings, kind: str) -> pd.DataFrame:
    """Per ticker, the strike/expiry with the best annualized yield on the fair premium
    among options whose probability of exercise is within your limit (calls: strike at
    or above your average price, if enabled; puts: meeting your target price, if set)."""
    if df.empty:
        return df
    d = df[df["prob_exercise"] <= settings.max_prob_exercise]
    if kind == CALL and settings.calls_above_avg_price and "strike_above_avg" in d:
        d = d[d["strike_above_avg"].isna() | (d["strike_above_avg"] == True)]  # noqa: E712
    if kind == PUT and "meets_target" in d and d["meets_target"].notna().any():
        d = d[d["meets_target"] == True]  # noqa: E712
    d = d[d["fair_price"] >= 0.01]
    if d.empty:
        return d
    return d.loc[d.groupby("ticker")["annualized_yield"].idxmax()].reset_index(drop=True)
