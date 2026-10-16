import datetime as dt
import io
import json

import pytest

from options_app.analysis import Settings, call_opportunities, put_opportunities
from options_app.market_data import YahooProvider
from options_app.opcoes_net import OpcoesNetClient, option_kind_from_ticker, parse_chain, parse_expirations
from options_app.portfolio import Holding

EXP = {"success": True, "data": {"vencimentos": [
    {"value": "2026-11-20", "text": "20/11/2026 (25 dias úteis)"}, {"value": "2026-12-18"}, "15/01/2027"]}}
CHAIN = {"success": True, "data": {"cotacoesOpcoes": [
    # ticker, ?, type, model, ?, strike, ?, ?, last, trades, volume
    ["PETRK400_2026", "x", "CALL", "A", "OTM", 40.0, None, None, 0.62, 120, 85000.0],
    ["PETRK410_2026", "x", "CALL", "A", "OTM", 41.0, None, None, 0.35, 40, 12000.0],
    ["PETRK41W4_2026", "x", "CALL", "A", "OTM", 41.0, None, None, 0.30, 2, 100.0],  # duplicate strike, less traded
    ["PETRW350_2026", "x", "PUT", "E", "OTM", "35,00", None, None, "0,48", 80, 40000.0],
    ["IBOVK130", "x", "CALL", "E", "OTM", None, None, None, 1.0, 1, 1.0],  # no strike -> skipped
]}}


def test_series_letters():
    assert option_kind_from_ticker("PETRK400") == "call"
    assert option_kind_from_ticker("VALEW650") == "put"
    assert option_kind_from_ticker("PETR4") is None


def test_parsers():
    assert parse_expirations(EXP) == [dt.date(2026, 11, 20), dt.date(2026, 12, 18), dt.date(2027, 1, 15)]
    calls = parse_chain(CHAIN, "call")
    assert list(calls["strike"]) == [40.0, 41.0]
    assert calls.loc[calls["strike"] == 41.0, "contract"].item() == "PETRK410"
    assert calls.loc[calls["strike"] == 40.0, "last"].item() == 0.62
    puts = parse_chain(CHAIN, "put")
    assert puts.iloc[0]["strike"] == 35.0 and puts.iloc[0]["last"] == 0.48
    dict_rows = {"data": {"cotacoesOpcoes": [{"ticker": "VALEK700", "strike": 70, "ultimo": 1.2, "volume": 5}]}}
    assert parse_chain(dict_rows, "call").iloc[0]["strike"] == 70.0


class FakeOpener:
    def __init__(self, fail=False):
        self.urls, self.fail = [], fail

    def __call__(self, req, timeout):
        self.urls.append(req.full_url)
        if self.fail:
            raise OSError("bloqueado")
        payload = CHAIN if "cotacoes=true" in req.full_url else EXP
        return io.BytesIO(json.dumps(payload).encode())


def test_client_urls_and_cache():
    op = FakeOpener()
    c = OpcoesNetClient(opener=op, cache_dir=None, sleep=lambda s: None)
    c.expirations("PETR4")
    c.chain("PETR4", dt.date(2026, 11, 20), "call")
    c.chain("PETR4", dt.date(2026, 11, 20), "put")  # cached
    assert len(op.urls) == 2
    assert "idAcao=PETR4" in op.urls[1] and "vencimentos=2026-11-20" in op.urls[1]


class FakeTicker:
    def __init__(self, sym):
        self.fast_info = {"lastPrice": 37.5}
        import pandas as pd
        import numpy as np

        rets = np.random.default_rng(0).normal(0, 0.02, 300)
        self._h = pd.DataFrame({"Close": 37.5 * np.exp(np.cumsum(rets) - rets.sum())})
        self.dividends = pd.Series(dtype=float)

    def history(self, **kw):
        return self._h


@pytest.fixture
def provider(monkeypatch):
    def make(fail=False):
        p = YahooProvider(OpcoesNetClient(opener=FakeOpener(fail), cache_dir=None, sleep=lambda s: None))
        monkeypatch.setattr(p._yf, "Ticker", FakeTicker)
        p._rates["BR"] = 0.13
        return p
    return make


def test_br_calls_use_opcoes_net_quotes(provider):
    p = provider()
    today = dt.date(2026, 10, 16)
    s = Settings(min_dte=0, max_dte=40, max_expiries=1, call_otm_min=0.0, call_otm_max=0.15)
    df, warns = call_opportunities([Holding("PETR4", 300, 30.0)], p, s, today)
    assert not warns and not p.notes
    assert set(df["strike_source"]) == {"listado"} and set(df["strike"]) == {40.0, 41.0}
    assert df.loc[df["strike"] == 40.0, "market_mid"].item() == 0.62
    assert df.loc[df["strike"] == 40.0, "contract"].item() == "PETRK400"
    puts = put_opportunities("PETR4", p, Settings(min_dte=0, max_dte=40, max_expiries=1), today)
    assert list(puts["strike"]) == [35.0]


def test_falls_back_to_grid_when_site_unreachable(provider):
    p = provider(fail=True)
    df, _ = call_opportunities([Holding("PETR4", 300)], p, Settings(), dt.date.today())
    assert set(df["strike_source"]) == {"grade"}
    assert len(p.notes) == 1 and "opcoes.net.br" in p.notes[0]
    p.reset_notes()
    assert p.notes == []


class RateLimitedOpener(FakeOpener):
    """Returns 429 for the first `n` requests."""

    def __init__(self, n, retry_after="2"):
        super().__init__()
        self.n, self.retry_after = n, retry_after

    def __call__(self, req, timeout):
        import urllib.error
        from email.message import Message

        if self.n > 0:
            self.n -= 1
            self.urls.append(req.full_url)
            h = Message()
            if self.retry_after:
                h["Retry-After"] = self.retry_after
            raise urllib.error.HTTPError(req.full_url, 429, "Too Many Requests", h, None)
        return super().__call__(req, timeout)


def test_retries_429_with_retry_after_and_throttles():
    sleeps, now = [], [0.0]
    op = RateLimitedOpener(2)
    c = OpcoesNetClient(opener=op, cache_dir=None, sleep=lambda s: (sleeps.append(s), now.__setitem__(0, now[0] + s)),
                        clock=lambda: now[0], min_interval=1.5)
    assert c.expirations("BBSE3")
    assert len(op.urls) == 3
    assert sleeps.count(2.0) == 2  # honoured Retry-After twice


def test_gives_up_after_max_retries():
    import urllib.error

    c = OpcoesNetClient(opener=RateLimitedOpener(10, retry_after=None), cache_dir=None, sleep=lambda s: None,
                        max_retries=2)
    with pytest.raises(urllib.error.HTTPError):
        c.expirations("BBSE3")


def test_disk_cache_survives_new_client(tmp_path):
    op = FakeOpener()
    OpcoesNetClient(opener=op, cache_dir=tmp_path, sleep=lambda s: None).expirations("WEGE3")
    op2 = FakeOpener(fail=True)
    assert OpcoesNetClient(opener=op2, cache_dir=tmp_path, sleep=lambda s: None).expirations("WEGE3")
    assert op2.urls == []
    stale = OpcoesNetClient(opener=FakeOpener(), cache_dir=tmp_path, cache_ttl=-1, sleep=lambda s: None)
    stale.expirations("WEGE3")  # expired cache -> refetch
