import datetime as dt

from options_app.markets import (
    BR, US, detect_market, monthly_expirations, normalize_ticker, strike_grid, third_friday, yahoo_symbol,
)


def test_market_detection_and_yahoo_symbols():
    for t in ("PETR4", "VALE3", "BOVA11", "AAPL34", "PETR4F", "petr4.sa"):
        assert detect_market(t) == BR
    for t in ("PLTR", "AAPL", "BRK.B", "SPY"):
        assert detect_market(t) == US
    assert normalize_ticker("PETR4F") == "PETR4"
    assert yahoo_symbol("PETR4") == "PETR4.SA"
    assert yahoo_symbol("BRK.B") == "BRK-B"


def test_b3_calendar():
    assert third_friday(2026, 11) == dt.date(2026, 11, 20)
    exps = monthly_expirations(dt.date(2026, 11, 20), 3)
    assert exps == [dt.date(2026, 12, 18), dt.date(2027, 1, 15), dt.date(2027, 2, 19)]


def test_strike_grid():
    assert strike_grid(37.5, 37.5, 40.0) == [37.5, 38.0, 38.5, 39.0, 39.5, 40.0]
    assert strike_grid(230, 230, 240) == [230.0, 232.5, 235.0, 237.5, 240.0]
