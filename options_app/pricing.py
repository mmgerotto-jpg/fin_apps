"""Option pricing models.

US equity options are American-style, so the fair value used throughout the app
is a Cox-Ross-Rubinstein binomial tree with early exercise and a continuous
dividend yield. Black-Scholes-Merton (European) is provided for Greeks,
probabilities and implied-volatility inversion.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

CALL = "call"
PUT = "put"


def _check_kind(kind: str) -> str:
    kind = kind.lower()
    if kind not in (CALL, PUT):
        raise ValueError(f"option kind must be 'call' or 'put', got {kind!r}")
    return kind


def norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def norm_pdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def _d1_d2(S: float, K: float, T: float, r: float, q: float, sigma: float) -> tuple[float, float]:
    vol_sqrt_t = sigma * math.sqrt(T)
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / vol_sqrt_t
    return d1, d1 - vol_sqrt_t


def _intrinsic(S: float, K: float, kind: str) -> float:
    return max(S - K, 0.0) if kind == CALL else max(K - S, 0.0)


def bs_price(S: float, K: float, T: float, r: float, sigma: float, kind: str, q: float = 0.0) -> float:
    """Black-Scholes-Merton price of a European option with continuous dividend yield q."""
    kind = _check_kind(kind)
    if T <= 0 or sigma <= 0:
        # Degenerate case: discounted forward intrinsic value.
        fwd = S * math.exp(-q * max(T, 0)) - K * math.exp(-r * max(T, 0))
        return max(fwd, 0.0) if kind == CALL else max(-fwd, 0.0)
    d1, d2 = _d1_d2(S, K, T, r, q, sigma)
    if kind == CALL:
        return S * math.exp(-q * T) * norm_cdf(d1) - K * math.exp(-r * T) * norm_cdf(d2)
    return K * math.exp(-r * T) * norm_cdf(-d2) - S * math.exp(-q * T) * norm_cdf(-d1)


@dataclass(frozen=True)
class Greeks:
    delta: float
    gamma: float
    theta: float  # per calendar day
    vega: float  # per 1 vol point (0.01)
    rho: float  # per 1% rate move
    prob_itm: float  # risk-neutral probability of finishing in the money


def bs_greeks(S: float, K: float, T: float, r: float, sigma: float, kind: str, q: float = 0.0) -> Greeks:
    kind = _check_kind(kind)
    if T <= 0 or sigma <= 0:
        itm = (S > K) if kind == CALL else (S < K)
        sign = 1.0 if kind == CALL else -1.0
        return Greeks(sign * float(itm), 0.0, 0.0, 0.0, 0.0, float(itm))
    d1, d2 = _d1_d2(S, K, T, r, q, sigma)
    sqrt_t = math.sqrt(T)
    disc_q = math.exp(-q * T)
    disc_r = math.exp(-r * T)
    gamma = disc_q * norm_pdf(d1) / (S * sigma * sqrt_t)
    vega = S * disc_q * norm_pdf(d1) * sqrt_t / 100.0
    common_theta = -S * disc_q * norm_pdf(d1) * sigma / (2 * sqrt_t)
    if kind == CALL:
        delta = disc_q * norm_cdf(d1)
        theta = common_theta - r * K * disc_r * norm_cdf(d2) + q * S * disc_q * norm_cdf(d1)
        rho = K * T * disc_r * norm_cdf(d2) / 100.0
        prob_itm = norm_cdf(d2)
    else:
        delta = -disc_q * norm_cdf(-d1)
        theta = common_theta + r * K * disc_r * norm_cdf(-d2) - q * S * disc_q * norm_cdf(-d1)
        rho = -K * T * disc_r * norm_cdf(-d2) / 100.0
        prob_itm = norm_cdf(-d2)
    return Greeks(delta, gamma, theta / 365.0, vega, rho, prob_itm)


def binomial_price(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    kind: str,
    q: float = 0.0,
    steps: int = 300,
    american: bool = True,
) -> float:
    """Cox-Ross-Rubinstein binomial tree price (American exercise by default)."""
    kind = _check_kind(kind)
    if T <= 0 or sigma <= 0:
        euro = bs_price(S, K, T, r, sigma, kind, q)
        return max(euro, _intrinsic(S, K, kind)) if american else euro
    steps = max(int(steps), 10)
    dt = T / steps
    u = math.exp(sigma * math.sqrt(dt))
    d = 1.0 / u
    growth = math.exp((r - q) * dt)
    p = (growth - d) / (u - d)
    if not 0.0 < p < 1.0:
        # Time step too coarse for the parameters; fall back to Black-Scholes.
        return bs_price(S, K, T, r, sigma, kind, q)
    disc = math.exp(-r * dt)

    j = np.arange(steps + 1)
    prices = S * u ** (steps - 2 * j)
    if kind == CALL:
        values = np.maximum(prices - K, 0.0)
    else:
        values = np.maximum(K - prices, 0.0)

    for step in range(steps - 1, -1, -1):
        values = disc * (p * values[:-1] + (1.0 - p) * values[1:])
        if american:
            prices = prices[:-1] * d  # node prices one step earlier: S*u^(step-2j)
            exercise = prices - K if kind == CALL else K - prices
            values = np.maximum(values, exercise)
    return float(values[0])


def fair_value(
    S: float, K: float, T: float, r: float, sigma: float, kind: str, q: float = 0.0, american: bool = True
) -> float:
    """Model fair value used by the app (American binomial unless american=False)."""
    if american:
        return binomial_price(S, K, T, r, sigma, kind, q)
    return bs_price(S, K, T, r, sigma, kind, q)


def implied_vol(
    price: float,
    S: float,
    K: float,
    T: float,
    r: float,
    kind: str,
    q: float = 0.0,
    lo: float = 1e-4,
    hi: float = 5.0,
    tol: float = 1e-6,
    max_iter: int = 200,
) -> float | None:
    """Black-Scholes implied volatility by bisection. Returns None if no solution exists."""
    kind = _check_kind(kind)
    if price is None or not math.isfinite(price) or price <= 0 or T <= 0:
        return None
    p_lo = bs_price(S, K, T, r, lo, kind, q)
    p_hi = bs_price(S, K, T, r, hi, kind, q)
    if price < p_lo - tol or price > p_hi + tol:
        return None
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        p_mid = bs_price(S, K, T, r, mid, kind, q)
        if abs(p_mid - price) < tol:
            return mid
        if p_mid < price:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)
