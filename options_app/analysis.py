"""Compare model fair values with market prices.

- Covered calls: calls on stocks you hold (from statements). As the seller you
  want market premium ABOVE fair value (edge > 0).
- Cash-secured puts: puts on stocks you plan to buy. Selling a put pays you
  premium to wait for your entry price; again, edge > 0 means the market pays
  more than the model says the option is worth.
"""

from __future__ import annotations

import datetime as dt
import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .market_data import MarketDataProvider
from .pricing import CALL, PUT, bs_greeks, bs_price, fair_value, implied_vol
from .statements import Position, WatchItem
from .volatility import estimate_vol


@dataclass
class Settings:
    min_dte: int = 7
    max_dte: int = 60
    max_expiries: int = 4
    call_otm_min: float = 0.0  # strikes from spot*(1+min) ...
    call_otm_max: float = 0.15  # ... to spot*(1+max)
    put_otm_min: float = 0.0  # strikes from spot*(1-max) to spot*(1-min)
    put_otm_max: float = 0.15
    vol_method: str = "blend"
    vol_window: int = 60
    vol_override: float | None = None  # use this sigma for every stock if set
    risk_free: float | None = None  # None -> provider's rate
    american: bool = True
    edge_threshold: float = 0.05  # |edge| / fair above this is flagged
    min_contracts_only: bool = False  # only analyse holdings with >= 100 shares
    min_bid: float = 0.05  # skip illiquid quotes


@dataclass
class StockContext:
    symbol: str
    spot: float
    sigma: float
    q: float
    r: float


def mid_price(bid: float, ask: float, last: float | None) -> float | None:
    if bid and ask and bid > 0 and ask >= bid:
        return (bid + ask) / 2.0
    if last and not (isinstance(last, float) and math.isnan(last)) and last > 0:
        return float(last)
    return None


def year_fraction(expiry: dt.date, today: dt.date) -> float:
    # Options expire at the close; count the expiry day itself.
    return max((expiry - today).days + 1, 1) / 365.0


def signal(edge_pct: float | None, threshold: float) -> str:
    if edge_pct is None or (isinstance(edge_pct, float) and math.isnan(edge_pct)):
        return "n/a"
    if edge_pct > threshold:
        return "Rich (good to sell)"
    if edge_pct < -threshold:
        return "Cheap (avoid selling)"
    return "Fair"


def stock_context(symbol: str, provider: MarketDataProvider, settings: Settings, rate: float) -> StockContext:
    spot = provider.spot(symbol)
    if settings.vol_override:
        sigma = settings.vol_override
    else:
        sigma = estimate_vol(provider.history(symbol, 400), settings.vol_method, settings.vol_window)
        if not sigma:
            raise LookupError(f"not enough price history to estimate volatility for {symbol}")
    return StockContext(symbol, spot, sigma, provider.dividend_yield(symbol), rate)


def _select_expiries(provider: MarketDataProvider, symbol: str, settings: Settings, today: dt.date) -> list[dt.date]:
    exps = [e for e in provider.expirations(symbol) if settings.min_dte <= (e - today).days <= settings.max_dte]
    return sorted(exps)[: settings.max_expiries]


def _price_rows(
    ctx: StockContext, expiry: dt.date, chain: pd.DataFrame, kind: str, settings: Settings, today: dt.date
) -> list[dict]:
    T = year_fraction(expiry, today)
    dte = (expiry - today).days
    rows = []
    for opt in chain.itertuples(index=False):
        K = float(opt.strike)
        mkt = mid_price(float(opt.bid or 0), float(opt.ask or 0), opt.last)
        if mkt is None or (opt.bid or 0) < settings.min_bid:
            continue
        fair = fair_value(ctx.spot, K, T, ctx.r, ctx.sigma, kind, ctx.q, settings.american)
        euro = bs_price(ctx.spot, K, T, ctx.r, ctx.sigma, kind, ctx.q)
        g = bs_greeks(ctx.spot, K, T, ctx.r, ctx.sigma, kind, ctx.q)
        iv = implied_vol(mkt, ctx.spot, K, T, ctx.r, kind, ctx.q)
        if iv is None and opt.market_iv and not math.isnan(opt.market_iv):
            iv = float(opt.market_iv)
        edge = mkt - fair
        edge_pct = edge / fair if fair > 0.005 else np.nan
        rows.append(
            {
                "symbol": ctx.symbol,
                "expiry": expiry,
                "dte": dte,
                "strike": K,
                "spot": ctx.spot,
                "bid": float(opt.bid or 0),
                "ask": float(opt.ask or 0),
                "market_mid": mkt,
                "fair_value": fair,
                "bs_european": euro,
                "edge": edge,
                "edge_pct": edge_pct,
                "signal": signal(edge_pct, settings.edge_threshold),
                "market_iv": iv,
                "model_vol": ctx.sigma,
                "delta": g.delta,
                "theta_day": g.theta,
                "prob_itm": g.prob_itm,
                "div_yield": ctx.q,
                "volume": opt.volume,
                "open_interest": opt.open_interest,
                "contract": opt.contract,
            }
        )
    return rows


def analyze_covered_calls(
    positions: list[Position],
    provider: MarketDataProvider,
    settings: Settings | None = None,
    today: dt.date | None = None,
    short_calls: dict[str, int] | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    """Fair value vs market for OTM calls on each held stock. Returns (table, errors).

    `short_calls` maps symbol -> call contracts already written; those shares are
    excluded from the contracts available to sell.
    """
    short_calls = short_calls or {}
    settings = settings or Settings()
    today = today or dt.date.today()
    rate = settings.risk_free if settings.risk_free is not None else provider.risk_free_rate()
    rows, errors = [], []
    for pos in positions:
        if pos.quantity <= 0:
            continue
        contracts = max(int(pos.quantity // 100) - short_calls.get(pos.symbol, 0), 0)
        if settings.min_contracts_only and contracts < 1:
            continue
        try:
            ctx = stock_context(pos.symbol, provider, settings, rate)
            lo, hi = ctx.spot * (1 + settings.call_otm_min), ctx.spot * (1 + settings.call_otm_max)
            for expiry in _select_expiries(provider, pos.symbol, settings, today):
                chain = provider.chain(pos.symbol, expiry, CALL)
                chain = chain[(chain["strike"] >= lo) & (chain["strike"] <= hi)]
                for row in _price_rows(ctx, expiry, chain, CALL, settings, today):
                    K, prem, dte = row["strike"], row["market_mid"], max(row["dte"], 1)
                    row.update(
                        {
                            "shares_held": pos.quantity,
                            "contracts_available": contracts,
                            "premium_total": prem * 100 * contracts,
                            "premium_yield": prem / ctx.spot,
                            "annualized_yield": prem / ctx.spot * 365 / dte,
                            "upside_cap_pct": K / ctx.spot - 1,
                            "cost_basis": pos.cost_basis_per_share,
                            "return_if_called": (
                                (K - pos.cost_basis_per_share + prem) / pos.cost_basis_per_share
                                if pos.cost_basis_per_share
                                else np.nan
                            ),
                        }
                    )
                    rows.append(row)
        except Exception as exc:  # keep going for the other symbols
            errors.append(f"{pos.symbol}: {exc}")
    return _finish(rows), errors


def analyze_cash_secured_puts(
    watchlist: list[WatchItem],
    provider: MarketDataProvider,
    settings: Settings | None = None,
    today: dt.date | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    """Fair value vs market for OTM puts on stocks you plan to buy. Returns (table, errors)."""
    settings = settings or Settings()
    today = today or dt.date.today()
    rate = settings.risk_free if settings.risk_free is not None else provider.risk_free_rate()
    rows, errors = [], []
    for item in watchlist:
        try:
            ctx = stock_context(item.symbol, provider, settings, rate)
            lo, hi = ctx.spot * (1 - settings.put_otm_max), ctx.spot * (1 - settings.put_otm_min)
            contracts = max(int(item.shares // 100), 1)
            for expiry in _select_expiries(provider, item.symbol, settings, today):
                chain = provider.chain(item.symbol, expiry, PUT)
                chain = chain[(chain["strike"] >= lo) & (chain["strike"] <= hi)]
                for row in _price_rows(ctx, expiry, chain, PUT, settings, today):
                    K, prem, dte = row["strike"], row["market_mid"], max(row["dte"], 1)
                    effective = K - prem
                    row.update(
                        {
                            "target_price": item.target_price,
                            "contracts": contracts,
                            "cash_required": K * 100 * contracts,
                            "premium_total": prem * 100 * contracts,
                            "effective_buy_price": effective,
                            "discount_to_spot": 1 - effective / ctx.spot,
                            "annualized_yield": prem / K * 365 / dte,
                            "meets_target": (effective <= item.target_price) if item.target_price else None,
                        }
                    )
                    rows.append(row)
        except Exception as exc:
            errors.append(f"{item.symbol}: {exc}")
    return _finish(rows), errors


def _finish(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return df.sort_values(["symbol", "expiry", "strike"]).reset_index(drop=True)


def best_per_symbol(df: pd.DataFrame, min_prob_otm: float = 0.0, by: str = "edge") -> pd.DataFrame:
    """For each symbol, the option where the market pays most above fair value.

    Ranked by dollar edge per share by default (percentage edge favours far-OTM
    options worth a few cents), optionally restricted to options with at least
    `min_prob_otm` risk-neutral chance of expiring worthless.
    """
    if df.empty:
        return df
    d = df.dropna(subset=[by])
    if min_prob_otm:
        d = d[(1 - d["prob_itm"]) >= min_prob_otm]
    if d.empty:
        return d
    return d.loc[d.groupby("symbol")[by].idxmax()].reset_index(drop=True)
