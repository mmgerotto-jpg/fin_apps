import datetime as dt

import numpy as np
import pandas as pd
import pytest

from options_app.analysis import Settings, call_opportunities, put_opportunities, suggestions
from options_app.market_data import DemoProvider
from options_app.portfolio import Holding
from options_app.pricing import binomial_price
from options_app.volatility import estimate_vol, historical_vol

TODAY = dt.date(2026, 10, 16)


class FlatProvider:
    """US-style market with a quoted chain priced at a known vol, or no chain (B3-style)."""

    def __init__(self, spot=100.0, with_chain=True):
        self._spot, self.with_chain = spot, with_chain

    def spot(self, s):
        return self._spot

    def history(self, s, period_days=400):
        rng = np.random.default_rng(0)
        rets = rng.normal(0, 0.25 / np.sqrt(252), 300)
        return pd.Series(self._spot * np.exp(np.cumsum(rets) - np.cumsum(rets)[-1]))

    def dividend_yield(self, s):
        return 0.0

    def expirations(self, s):
        return [TODAY + dt.timedelta(days=30), TODAY + dt.timedelta(days=200)]

    def risk_free_rate(self, market):
        return 0.04 if market == "US" else 0.13

    def chain(self, s, expiry, kind):
        if not self.with_chain:
            return None
        from options_app.analysis import year_fraction

        T = year_fraction(expiry, TODAY)
        rows = []
        for K in (90.0, 95.0, 100.0, 105.0, 110.0, 130.0):
            p = binomial_price(self._spot, K, T, 0.04, 0.30, kind)
            rows.append({"strike": K, "bid": p - 0.02, "ask": p + 0.02, "last": p, "volume": 1,
                         "open_interest": 1, "market_iv": 0.3, "contract": ""})
        return pd.DataFrame(rows)


def test_calls_use_listed_strikes_and_free_lots():
    s = Settings(vol_override={"XYZ": 0.30})
    df, warns = call_opportunities([Holding("XYZ", 350, 80.0, market="US")], FlatProvider(), s, TODAY,
                                   committed_shares={"XYZ": 100})
    assert not warns
    assert set(df["strike"]) == {100.0, 105.0, 110.0}  # spot .. +20%, listed only, 1 expiry within 15-75d
    assert (df["contracts"] == 2).all()  # 350 - 100 committed = 250 -> 2 lots
    assert (df["strike_source"] == "listado").all()
    # model priced at the same vol as the market -> fair ~ market mid, IV recovered
    assert (df["fair_price"] - df["market_mid"]).abs().max() < 0.03
    assert df["market_iv"].iloc[0] == pytest.approx(0.30, abs=0.01)
    row = df[df["strike"] == 110].iloc[0]
    assert row["return_if_exercised"] == pytest.approx((110 + row["fair_price"]) / 80 - 1)


def test_br_calls_use_strike_grid_and_selic():
    s = Settings(vol_override={"PETR4": 0.30})
    df, _ = call_opportunities([Holding("PETR4", 200, 30.0)], FlatProvider(37.5, with_chain=False), s, TODAY)
    assert (df["strike_source"] == "grade").all()
    assert (df["currency"] == "BRL").all()
    assert df["strike"].min() >= 37.5 and df["strike"].max() <= 45.0
    assert np.allclose(np.diff(sorted(df["strike"])), 0.5)  # R$0.50 steps for a R$37 stock
    assert (df["rate"] == 0.13).all()


def test_rate_override_is_converted_to_continuous():
    s = Settings(vol_override={"PETR4": 0.3}, rate_override={"BR": 0.15})
    df, _ = call_opportunities([Holding("PETR4", 100)], FlatProvider(37.5, False), s, TODAY)
    assert df["rate"].iloc[0] == pytest.approx(np.log1p(0.15))


def test_holdings_without_full_lot_or_not_optionable_are_skipped():
    df, warns = call_opportunities(
        [Holding("XYZ", 60, market="US"), Holding("XPML11", 500, asset_type="fii")], FlatProvider(), Settings(), TODAY
    )
    assert df.empty and len(warns) == 1 and "XYZ" in warns[0]


def test_puts_effective_price_and_target():
    s = Settings(vol_override={"XYZ": 0.30})
    df = put_opportunities("XYZ", FlatProvider(), s, TODAY, target_price=94.0, shares=250)
    assert set(df["strike"]) == {90.0, 95.0, 100.0}
    r = df[df["strike"] == 95.0].iloc[0]
    assert r["effective_price"] == pytest.approx(95.0 - r["fair_price"])
    assert r["contracts"] == 2 and r["cash_required"] == 95.0 * 200
    assert bool(r["meets_target"]) == (r["effective_price"] <= 94.0)


def test_suggestions_respect_probability_and_avg_price():
    s = Settings(vol_override={"XYZ": 0.30}, max_prob_exercise=0.30)
    df, _ = call_opportunities([Holding("XYZ", 100, 107.0, market="US")], FlatProvider(), s, TODAY)
    best = suggestions(df, s, "call")
    assert len(best) == 1
    assert best.iloc[0]["prob_exercise"] <= 0.30 and best.iloc[0]["strike"] >= 107.0
    s.max_prob_exercise = 0.01
    assert suggestions(df, s, "call").empty


def test_demo_provider_end_to_end_br_and_us():
    p = DemoProvider(today=TODAY)
    df, warns = call_opportunities([Holding("PETR4", 300, 30), Holding("AAPL", 200, 150)], p, Settings(), TODAY)
    assert not warns and set(df["ticker"]) == {"PETR4", "AAPL"}
    assert set(df.loc[df["ticker"] == "PETR4", "strike_source"]) == {"grade"}
    assert set(df.loc[df["ticker"] == "AAPL", "strike_source"]) == {"listado"}
    # B3 expiries are third Fridays
    assert all(e.weekday() == 4 and 15 <= e.day <= 21 for e in df.loc[df["ticker"] == "PETR4", "expiry"])
    puts = put_opportunities("PLTR", p, Settings(), TODAY)
    assert (puts["strike"] <= puts["spot"]).all()


def test_errors_are_collected_not_raised():
    class Broken(FlatProvider):
        def spot(self, s):
            raise LookupError("sem dados")

    df, warns = call_opportunities([Holding("BAD", 100, market="US")], Broken(), Settings(), TODAY)
    assert df.empty and warns == ["BAD: sem dados"]


def test_vol_estimators():
    rng = np.random.default_rng(1)
    closes = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.3 / np.sqrt(252), 2000))))
    assert historical_vol(closes, 1000) == pytest.approx(0.30, abs=0.03)
    for m in ("historical", "ewma", "blend"):
        assert 0.15 < estimate_vol(closes, m) < 0.5
    assert historical_vol(closes.head(5)) is None
