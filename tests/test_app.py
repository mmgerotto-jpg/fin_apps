import io
from pathlib import Path

import pytest

from optionsapp.app import create_app

SAMPLE = Path(__file__).parent.parent / "samples" / "sample_statement.csv"


@pytest.fixture
def client(tmp_path):
    return create_app("demo", str(tmp_path / "p.json")).test_client()


def test_end_to_end(client):
    r = client.post("/api/statement", data={"file": (io.BytesIO(SAMPLE.read_bytes()), "s.csv")},
                    content_type="multipart/form-data")
    assert r.status_code == 200 and len(r.json["holdings"]) == 3

    res = client.post("/api/analyze/calls", json={}).json["results"]
    aapl = next(x for x in res if x["symbol"] == "AAPL")
    assert aapl["contracts"] == 3 and aapl["rows"] and not aapl["error"]
    for row in aapl["rows"]:
        assert row["fair"] > 0 and row["market"] > 0 and row["verdict"] in ("RICH (sell)", "FAIR", "CHEAP (avoid selling)")
        assert 0 <= row["moneyness"] <= 0.10

    client.put("/api/watchlist", json=[{"symbol": "nvda", "shares": 200, "target_price": ""}])
    spot = client.post("/api/analyze/puts", json={}).json["results"][0]["spot"]
    client.put("/api/watchlist", json=[{"symbol": "nvda", "shares": 200, "target_price": round(spot * 0.95, 2)}])
    p = client.post("/api/analyze/puts", json={}).json["results"][0]
    assert p["symbol"] == "NVDA" and p["contracts"] == 2
    assert any(x.get("target_match") for x in p["rows"])
    assert all(x["strike"] <= spot * 1.0001 for x in p["rows"])
    assert all(x["effective_buy"] < x["strike"] for x in p["rows"])


def test_puts_without_chain_fall_back_to_theoretical(client, monkeypatch):
    prov = client.application.config["provider"]
    monkeypatch.setattr(prov, "expiries", lambda s: [])
    client.put("/api/watchlist", json=[{"symbol": "XYZ", "shares": 100, "target_price": 50}])
    p = client.post("/api/analyze/puts", json={}).json["results"][0]
    assert p["rows"] and all("theoretical" in x["expiry"] for x in p["rows"])


def test_calc_and_index(client):
    r = client.post("/api/calc", json={"kind": "call", "spot": 42, "strike": 40, "dte": 182.5, "vol": 20,
                                      "rate": 10, "div": 0, "market": 5}).json
    assert abs(r["bsm"] - 4.76) < 0.01 and r["implied_vol"] > 0.2
    assert client.get("/").status_code == 200
