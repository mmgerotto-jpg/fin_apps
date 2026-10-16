import math

from optionsapp.pricing import binomial_price, bs_greeks, bs_price, implied_vol


def test_bs_reference_value():
    # Hull textbook: S=42 K=40 r=10% sigma=20% T=0.5 -> call 4.76, put 0.81
    assert abs(bs_price(42, 40, 0.5, 0.1, 0, 0.2, "call") - 4.76) < 0.01
    assert abs(bs_price(42, 40, 0.5, 0.1, 0, 0.2, "put") - 0.81) < 0.01


def test_put_call_parity_with_dividends():
    S, K, T, r, q, v = 100, 95, 0.75, 0.04, 0.02, 0.3
    c, p = bs_price(S, K, T, r, q, v, "call"), bs_price(S, K, T, r, q, v, "put")
    assert abs((c - p) - (S * math.exp(-q * T) - K * math.exp(-r * T))) < 1e-9


def test_binomial_converges_and_american_premium():
    S, K, T, r, v = 100, 110, 1, 0.05, 0.25
    assert abs(binomial_price(S, K, T, r, 0, v, "call", american=False, steps=500) - bs_price(S, K, T, r, 0, v)) < 0.02
    # American call on non-dividend stock == European
    assert abs(binomial_price(S, K, T, r, 0, v, "call", True, 500) - binomial_price(S, K, T, r, 0, v, "call", False, 500)) < 1e-9
    # American put worth more than European
    assert binomial_price(S, K, T, r, 0, v, "put", True) > bs_price(S, K, T, r, 0, v, "put")


def test_implied_vol_roundtrip():
    for kind in ("call", "put"):
        px = bs_price(150, 160, 0.2, 0.04, 0.01, 0.37, kind)
        assert abs(implied_vol(px, 150, 160, 0.2, 0.04, 0.01, kind) - 0.37) < 1e-6
    assert implied_vol(0.0001, 100, 50, 0.1, 0.04, 0, "call") is None  # below intrinsic


def test_greeks_signs():
    g = bs_greeks(100, 100, 0.25, 0.04, 0, 0.2, "call")
    assert 0.5 < g.delta < 0.6 and g.gamma > 0 and g.vega > 0 and g.theta < 0
    assert bs_greeks(100, 100, 0.25, 0.04, 0, 0.2, "put").delta < 0
