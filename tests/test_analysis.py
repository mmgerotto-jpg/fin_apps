import datetime as dt

import pandas as pd
import pytest

from options_app.analysis import Settings, analyze_cash_secured_puts, analyze_covered_calls, best_per_symbol, signal
from options_app.market_data import DemoProvider
from options_app.pricing import binomial_price
from options_app.statements import Position, WatchItem
from options_app.volatility import estimate_vol, historical_vol

TODAY = dt.date(2026, 10, 16)


class FixedProvider:
    """Flat-vol market where the market prices options at a known vol."""

    def __init__(self, market_vol, spot=100.0):
        self.market_vol, self._spot = market_vol, spot

    def spot(self, s):
        return self._spot

    def history(self, s, period_days=400):
        import numpy as np

        rng = np.random.default_rng(0)
        rets = rng.normal(0, 0.25 / np.sqrt(252), 300)
        return pd.Series(self._spot * np.exp(np.cumsum(rets) - np.cumsum(rets)[-1]))

    def dividend_yield(self, s):
        return 0.0

    def expirations(self, s):
        return [TODAY + dt.timedelta(days=30)]

    def risk_free_rate(self):
        return 0.04

    def chain(self, s, expiry, kind):
        from options_app.analysis import year_fraction

        T = year_fraction(expiry, TODAY)
        rows = []
        for K in (95.0, 100.0, 105.0, 110.0):
            p = binomial_price(self._spot, K, T, 0.04, self.market_vol, kind)
            rows.append({"strike": K, "bid": p - 0.02, "ask": p + 0.02, "last": p, "volume": 1,
                         "open_interest": 1, "market_iv": self.market_vol, "contract": ""})
        return pd.DataFrame(rows)


def test_rich_market_flagged_for_calls():
    settings = Settings(vol_override=0.25, min_bid=0.0)
    df, errors = analyze_covered_calls([Position("XYZ", 250, 80.0)], FixedProvider(0.35), settings, TODAY)
    assert not errors
    assert set(df["strike"]) == {100.0, 105.0, 110.0}  # only spot..+15% strikes
    assert (df["edge"] > 0).all() and (df["signal"] == "Rich (good to sell)").all()
    assert (df["contracts_available"] == 2).all()
    assert df["market_iv"].iloc[0] == pytest.approx(0.35, abs=0.01)


def test_fair_market_and_short_calls_reduce_contracts():
    settings = Settings(vol_override=0.30, min_bid=0.0)
    df, _ = analyze_covered_calls([Position("XYZ", 250)], FixedProvider(0.30), settings, TODAY, short_calls={"XYZ": 1})
    assert (df["signal"] == "Fair").all()
    assert (df["contracts_available"] == 1).all()


def test_cheap_puts_and_effective_price():
    settings = Settings(vol_override=0.40, min_bid=0.0)
    df, errors = analyze_cash_secured_puts([WatchItem("XYZ", 97.0, 200)], FixedProvider(0.25), settings, TODAY)
    assert not errors
    assert set(df["strike"]) == {95.0, 100.0}  # spot-15%..spot
    assert (df["signal"] == "Cheap (avoid selling)").all()
    row = df[df["strike"] == 95.0].iloc[0]
    assert row["effective_buy_price"] == pytest.approx(95.0 - row["market_mid"])
    assert row["cash_required"] == 95.0 * 200 and bool(row["meets_target"])


def test_demo_provider_end_to_end():
    p = DemoProvider(today=TODAY)
    calls, errs = analyze_covered_calls([Position("AAPL", 300, 150), Position("KO", 50)], p, Settings(), TODAY)
    assert not errs and set(calls["symbol"]) == {"AAPL", "KO"}
    puts, errs = analyze_cash_secured_puts([WatchItem("GOOGL", 160)], p, Settings(), TODAY)
    assert not errs and (puts["strike"] <= puts["spot"]).all()
    best = best_per_symbol(calls)
    assert len(best) == 2


def test_errors_are_collected_not_raised():
    class Broken(FixedProvider):
        def spot(self, s):
            raise LookupError("no data")

    df, errors = analyze_covered_calls([Position("BAD", 100)], Broken(0.3), Settings(), TODAY)
    assert df.empty and errors == ["BAD: no data"]


def test_vol_estimators():
    import numpy as np

    rng = np.random.default_rng(1)
    closes = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.3 / np.sqrt(252), 2000))))
    assert historical_vol(closes, 1000) == pytest.approx(0.30, abs=0.03)
    for m in ("historical", "ewma", "blend"):
        assert 0.15 < estimate_vol(closes, m) < 0.5
    assert historical_vol(closes.head(5)) is None


def test_signal_thresholds():
    assert signal(0.10, 0.05).startswith("Rich")
    assert signal(-0.10, 0.05).startswith("Cheap")
    assert signal(0.01, 0.05) == "Fair"
    assert signal(float("nan"), 0.05) == "n/a"
