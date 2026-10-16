import datetime as dt
from types import SimpleNamespace

import pandas as pd
import pytest

from options_app.market_data import CHAIN_COLUMNS, YahooProvider


class FakeTicker:
    def __init__(self, sym):
        self.sym = sym
        self.fast_info = {"lastPrice": 200.0}
        self.options = ("2026-11-20", "2026-12-18")
        idx = pd.date_range(end=pd.Timestamp.now(tz="America/New_York"), periods=4, freq="90D")
        self.dividends = pd.Series([0.5, 0.5, 0.5, 0.5], index=idx)

    def history(self, **kw):
        if self.sym == "^IRX":
            return pd.DataFrame({"Close": [4.25, 4.3]})
        return pd.DataFrame({"Close": [198.0, 199.0, 200.0]})

    def option_chain(self, date):
        calls = pd.DataFrame({"contractSymbol": ["X1"], "strike": [210.0], "lastPrice": [3.1], "bid": [3.0],
                              "ask": [3.2], "volume": [None], "openInterest": [55], "impliedVolatility": [0.3]})
        return SimpleNamespace(calls=calls, puts=calls.iloc[0:0])


@pytest.fixture
def provider(monkeypatch):
    p = YahooProvider()
    monkeypatch.setattr(p._yf, "Ticker", FakeTicker)
    return p


def test_yahoo_mapping(provider):
    assert provider.spot("BRK.B") == 200.0
    assert provider._tickers.keys() == {"BRK-B"}
    assert provider.expirations("AAPL") == [dt.date(2026, 11, 20), dt.date(2026, 12, 18)]
    calls = provider.chain("AAPL", dt.date(2026, 11, 20), "call")
    assert list(calls.columns) == CHAIN_COLUMNS
    assert calls.iloc[0]["volume"] == 0 and calls.iloc[0]["ask"] == 3.2
    assert provider.chain("AAPL", dt.date(2026, 11, 20), "put").empty
    assert provider.risk_free_rate() == pytest.approx(0.043)
    assert 0.005 < provider.dividend_yield("AAPL") <= 0.01  # ~1.5-2.0 / 200
    assert len(provider.history("AAPL")) == 3
