import math

import pytest

from options_app.pricing import bs_greeks, bs_price, binomial_price, implied_vol


def test_black_scholes_reference_value():
    # Hull, Options Futures & Other Derivatives: S=42, K=40, r=10%, sigma=20%, T=0.5
    assert bs_price(42, 40, 0.5, 0.10, 0.20, "call") == pytest.approx(4.76, abs=0.01)
    assert bs_price(42, 40, 0.5, 0.10, 0.20, "put") == pytest.approx(0.81, abs=0.01)


def test_put_call_parity_with_dividends():
    S, K, T, r, q, v = 100, 95, 0.75, 0.04, 0.02, 0.3
    c, p = bs_price(S, K, T, r, v, "call", q), bs_price(S, K, T, r, v, "put", q)
    assert c - p == pytest.approx(S * math.exp(-q * T) - K * math.exp(-r * T), abs=1e-9)


def test_binomial_converges_to_black_scholes_for_european():
    euro = bs_price(100, 105, 0.5, 0.05, 0.25, "call", 0.01)
    assert binomial_price(100, 105, 0.5, 0.05, 0.25, "call", 0.01, steps=800, american=False) == pytest.approx(euro, abs=0.01)


def test_american_put_early_exercise_premium():
    # Hull example: American put S=50 K=50 r=10% sigma=40% T=5 months ~ 4.28 (vs European ~4.08)
    am = binomial_price(50, 50, 5 / 12, 0.10, 0.40, "put", steps=1000)
    eu = bs_price(50, 50, 5 / 12, 0.10, 0.40, "put")
    assert am == pytest.approx(4.28, abs=0.02)
    assert am > eu


def test_american_call_no_dividend_equals_european():
    am = binomial_price(100, 100, 1.0, 0.05, 0.2, "call", 0.0, steps=800)
    assert am == pytest.approx(bs_price(100, 100, 1.0, 0.05, 0.2, "call"), abs=0.01)


def test_implied_vol_round_trip():
    price = bs_price(230, 250, 30 / 365, 0.04, 0.33, "call", 0.005)
    assert implied_vol(price, 230, 250, 30 / 365, 0.04, "call", 0.005) == pytest.approx(0.33, abs=1e-4)
    assert implied_vol(0.0, 230, 250, 0.1, 0.04, "call") is None
    assert implied_vol(500.0, 230, 250, 0.1, 0.04, "call") is None  # above max possible price


def test_greeks_signs():
    c = bs_greeks(100, 100, 0.25, 0.04, 0.3, "call")
    p = bs_greeks(100, 100, 0.25, 0.04, 0.3, "put")
    assert 0 < c.delta < 1 and -1 < p.delta < 0
    assert c.gamma == pytest.approx(p.gamma)
    assert c.theta < 0 and c.vega > 0
    assert c.prob_itm + p.prob_itm == pytest.approx(1.0)


def test_expired_option_is_intrinsic():
    assert bs_price(110, 100, 0, 0.04, 0.3, "call") == pytest.approx(10)
    assert binomial_price(90, 100, 0, 0.04, 0.3, "put") == pytest.approx(10)
