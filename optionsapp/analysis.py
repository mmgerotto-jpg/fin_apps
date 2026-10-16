"""Fair-value vs market comparison for covered calls (holdings) and cash-secured puts (watchlist)."""
from __future__ import annotations

import math
from datetime import date

import numpy as np

from .pricing import binomial_price, bs_greeks, implied_vol


def historical_vol(closes, window):
    c = np.asarray(closes[-(window + 1):], dtype=float)
    if len(c) < 3:
        return None
    r = np.diff(np.log(c))
    return float(np.std(r, ddof=1) * math.sqrt(252))


def vol_estimates(closes):
    est = {f"hv{w}": historical_vol(closes, w) for w in (20, 60, 120, 250)}
    vals = [v for v in (est["hv20"], est["hv60"], est["hv120"]) if v]
    # Blend: weights recent regime and longer-run level
    est["blend"] = (0.3 * (est["hv20"] or 0) + 0.4 * (est["hv60"] or 0) + 0.3 * (est["hv120"] or 0)) \
        if len(vals) == 3 else (vals[0] if vals else 0.3)
    return est


def pick_expiries(expiries, min_dte, max_dte, limit=3):
    today = date.today()
    out = []
    for e in expiries:
        dte = (date.fromisoformat(e) - today).days
        if min_dte <= dte <= max_dte:
            out.append((e, dte))
    if not out and expiries:  # nearest to the window
        mid = (min_dte + max_dte) / 2
        e = min(expiries, key=lambda e: abs((date.fromisoformat(e) - today).days - mid))
        out = [(e, (date.fromisoformat(e) - today).days)]
    return out[:limit]


def _mid(row):
    b, a = row.get("bid"), row.get("ask")
    if b and a and a >= b > 0:
        return (b + a) / 2
    return row.get("last") or None


def price_contract(kind, S, K, dte, r, q, fair_vol, row=None, threshold=0.10):
    T = max(dte, 0.5) / 365
    fair = binomial_price(S, K, T, r, q, fair_vol, kind, american=True, steps=150)
    g = bs_greeks(S, K, T, r, q, fair_vol, kind)
    res = dict(strike=K, dte=dte, fair=fair, delta=g.delta, theta=g.theta, vega=g.vega,
               moneyness=K / S - 1)
    if row:
        mkt = _mid(row)
        iv = implied_vol(mkt, S, K, T, r, q, kind) if mkt else None
        edge = (mkt - fair) if mkt is not None else None
        rel = edge / fair if (edge is not None and fair > 0.01) else None
        # Seller's perspective: market above fair means the option is rich -> attractive to sell.
        verdict = "n/a" if rel is None else "RICH (sell)" if rel > threshold else \
            "CHEAP (avoid selling)" if rel < -threshold else "FAIR"
        res.update(bid=row.get("bid"), ask=row.get("ask"), market=mkt, iv=iv or row.get("iv"),
                   edge=edge, edge_pct=rel, verdict=verdict, volume=row.get("volume"), oi=row.get("oi"))
    return res


def _settings(s):
    d = dict(min_dte=20, max_dte=60, n_expiries=2, vol_source="blend", vol_override=None,
             call_otm=(0.0, 0.10), put_otm=(-0.15, 0.0), threshold=0.10)
    d.update({k: v for k, v in (s or {}).items() if v is not None})
    return d


def _fair_vol(est, st):
    if st["vol_override"]:
        return float(st["vol_override"])
    return est.get(st["vol_source"]) or est["blend"]


def analyze_calls(provider, holding, settings=None):
    """Covered-call candidates for a stock holding."""
    st = _settings(settings)
    sym = holding["symbol"]
    q = provider.quote(sym)
    S, dy, r = q["spot"], q["div_yield"], provider.risk_free()
    est = vol_estimates(q["closes"])
    fv = _fair_vol(est, st)
    contracts = int(holding.get("quantity", 0) // 100)
    out = dict(symbol=sym, spot=S, div_yield=dy, rate=r, vols=est, fair_vol=fv,
               quantity=holding.get("quantity"), contracts=contracts, rows=[], error=None)
    lo, hi = st["call_otm"]
    try:
        exps = pick_expiries(provider.expiries(sym), st["min_dte"], st["max_dte"], st["n_expiries"])
    except Exception as e:
        exps, out["error"] = [], f"no option chain: {e}"
    for exp, dte in exps:
        try:
            chain = provider.chain(sym, exp)["calls"]
        except Exception as e:
            out["error"] = str(e)
            continue
        for row in chain:
            m = row["strike"] / S - 1
            if lo <= m <= hi:
                p = price_contract("call", S, row["strike"], dte, r, dy, fv, row, st["threshold"])
                p["expiry"] = exp
                if p.get("market"):
                    p["premium_total"] = p["market"] * 100 * contracts
                    p["yield_annual"] = p["market"] / S * 365 / max(dte, 1)
                    p["called_away_return"] = (row["strike"] - S + p["market"]) / S
                out["rows"].append(p)
    if not exps and not out["error"]:
        out["error"] = "no listed expiries"
    return out


def analyze_puts(provider, plan, settings=None):
    """Cash-secured put candidates for a stock the user plans to buy.

    plan: {symbol, shares (optional), target_price (optional)}
    """
    st = _settings(settings)
    sym = plan["symbol"]
    q = provider.quote(sym)
    S, dy, r = q["spot"], q["div_yield"], provider.risk_free()
    est = vol_estimates(q["closes"])
    fv = _fair_vol(est, st)
    target = plan.get("target_price")
    shares = plan.get("shares") or 100
    contracts = max(int(shares // 100), 1)
    out = dict(symbol=sym, spot=S, div_yield=dy, rate=r, vols=est, fair_vol=fv, target_price=target,
               shares=shares, contracts=contracts, rows=[], error=None)
    lo, hi = st["put_otm"]
    if target:
        lo = min(lo, target / S - 1 - 0.03)
    try:
        exps = pick_expiries(provider.expiries(sym), st["min_dte"], st["max_dte"], st["n_expiries"])
    except Exception as e:
        exps, out["error"] = [], f"no option chain: {e}"
    for exp, dte in exps:
        try:
            chain = provider.chain(sym, exp)["puts"]
        except Exception as e:
            out["error"] = str(e)
            continue
        best = None
        for row in chain:
            m = row["strike"] / S - 1
            if lo <= m <= hi:
                p = price_contract("put", S, row["strike"], dte, r, dy, fv, row, st["threshold"])
                p["expiry"] = exp
                prem = p.get("market") or p["fair"]
                p["effective_buy"] = row["strike"] - prem
                p["discount_vs_spot"] = p["effective_buy"] / S - 1
                p["yield_annual"] = prem / row["strike"] * 365 / max(dte, 1)
                p["cash_required"] = row["strike"] * 100 * contracts
                p["prob_assign"] = -p["delta"]
                out["rows"].append(p)
                if target and (best is None or abs(row["strike"] - target) < abs(best["strike"] - target)):
                    best = p
        if best and abs(best["strike"] / target - 1) <= 0.05:
            best["target_match"] = True
    # No chain available: still compute theoretical fair puts on a synthetic strike grid
    if not out["rows"]:
        dte = (st["min_dte"] + st["max_dte"]) // 2
        strikes = sorted({round(float(S * (1 + m)), 2) for m in np.linspace(lo, hi, 6)} | ({target} if target else set()))
        for K in strikes:
            p = price_contract("put", S, K, dte, r, dy, fv)
            p.update(expiry=f"~{dte}d (theoretical)", effective_buy=K - p["fair"],
                     discount_vs_spot=(K - p["fair"]) / S - 1, yield_annual=p["fair"] / K * 365 / dte,
                     cash_required=K * 100 * contracts, prob_assign=-p["delta"], target_match=bool(K == target))
            out["rows"].append(p)
    return out


def quick_price(kind, S, K, dte, r, q, vol, market=None):
    """Single-contract calculator."""
    T = max(dte, 0.5) / 365
    from .pricing import bs_price
    g = bs_greeks(S, K, T, r, q, vol, kind)
    res = dict(bsm=bs_price(S, K, T, r, q, vol, kind),
               american=binomial_price(S, K, T, r, q, vol, kind, True, 300),
               delta=g.delta, gamma=g.gamma, vega=g.vega, theta=g.theta, rho=g.rho)
    if market:
        res["implied_vol"] = implied_vol(market, S, K, T, r, q, kind)
        res["edge"] = market - res["american"]
    return res
