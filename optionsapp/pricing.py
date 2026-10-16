"""Option pricing: Black-Scholes-Merton (European), CRR binomial (American), Greeks, implied vol."""
from __future__ import annotations

import math
from dataclasses import dataclass

from scipy.optimize import brentq
from scipy.stats import norm


def _d1d2(S, K, T, r, q, sigma):
    vs = sigma * math.sqrt(T)
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma**2) * T) / vs
    return d1, d1 - vs


def bs_price(S, K, T, r, q, sigma, kind="call"):
    """Black-Scholes-Merton with continuous dividend yield q."""
    if T <= 0 or sigma <= 0:
        intrinsic = max(S - K, 0) if kind == "call" else max(K - S, 0)
        return intrinsic
    d1, d2 = _d1d2(S, K, T, r, q, sigma)
    if kind == "call":
        return S * math.exp(-q * T) * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)
    return K * math.exp(-r * T) * norm.cdf(-d2) - S * math.exp(-q * T) * norm.cdf(-d1)


@dataclass
class Greeks:
    delta: float
    gamma: float
    vega: float   # per 1 vol point (1%)
    theta: float  # per calendar day
    rho: float    # per 1% rate move


def bs_greeks(S, K, T, r, q, sigma, kind="call") -> Greeks:
    if T <= 0 or sigma <= 0:
        d = (1.0 if S > K else 0.0) if kind == "call" else (-1.0 if S < K else 0.0)
        return Greeks(d, 0, 0, 0, 0)
    d1, d2 = _d1d2(S, K, T, r, q, sigma)
    eq, er = math.exp(-q * T), math.exp(-r * T)
    pdf = norm.pdf(d1)
    gamma = eq * pdf / (S * sigma * math.sqrt(T))
    vega = S * eq * pdf * math.sqrt(T) / 100
    if kind == "call":
        delta = eq * norm.cdf(d1)
        theta = (-S * eq * pdf * sigma / (2 * math.sqrt(T))
                 - r * K * er * norm.cdf(d2) + q * S * eq * norm.cdf(d1)) / 365
        rho = K * T * er * norm.cdf(d2) / 100
    else:
        delta = -eq * norm.cdf(-d1)
        theta = (-S * eq * pdf * sigma / (2 * math.sqrt(T))
                 + r * K * er * norm.cdf(-d2) - q * S * eq * norm.cdf(-d1)) / 365
        rho = -K * T * er * norm.cdf(-d2) / 100
    return Greeks(delta, gamma, vega, theta, rho)


def binomial_price(S, K, T, r, q, sigma, kind="call", american=True, steps=200):
    """Cox-Ross-Rubinstein tree; American early exercise optional."""
    if T <= 0 or sigma <= 0:
        return max(S - K, 0) if kind == "call" else max(K - S, 0)
    dt = T / steps
    u = math.exp(sigma * math.sqrt(dt))
    d = 1 / u
    p = (math.exp((r - q) * dt) - d) / (u - d)
    disc = math.exp(-r * dt)
    sign = 1 if kind == "call" else -1
    vals = [max(sign * (S * u**j * d ** (steps - j) - K), 0) for j in range(steps + 1)]
    for i in range(steps - 1, -1, -1):
        for j in range(i + 1):
            v = disc * (p * vals[j + 1] + (1 - p) * vals[j])
            if american:
                v = max(v, sign * (S * u**j * d ** (i - j) - K))
            vals[j] = v
    return vals[0]


def implied_vol(price, S, K, T, r, q, kind="call"):
    """Solve BSM implied volatility; returns None when no solution exists."""
    if T <= 0 or price <= 0:
        return None
    try:
        f = lambda s: bs_price(S, K, T, r, q, s, kind) - price
        if f(1e-4) > 0 or f(5.0) < 0:
            return None
        return brentq(f, 1e-4, 5.0, xtol=1e-8)
    except Exception:
        return None
