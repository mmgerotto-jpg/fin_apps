from options_app.portfolio import Holding, OptionHolding, Portfolio, frame_to_holdings


def test_replace_broker_consolidate_and_persist(tmp_path):
    p = Portfolio(
        [Holding("PETR4", 100, 30.0, broker="BTG Pactual"), Holding("PETR4", 300, 34.0, broker="XP Investimentos"),
         Holding("AAPL", 50, 150.0, broker="Avenue")],
        [OptionHolding("PETR4", "call", -200, 40.0, "2026-11-20", "XP Investimentos")],
    )
    cons = {h.ticker: h for h in p.consolidated()}
    assert cons["PETR4"].quantity == 400 and cons["PETR4"].avg_price == 33.0
    assert cons["PETR4"].broker == "BTG Pactual, XP Investimentos"
    assert cons["AAPL"].market == "US" and cons["PETR4"].currency == "BRL"
    assert p.committed_shares() == {"PETR4": 200}

    p.replace_broker("xp investimentos", [Holding("VALE3", 100, 60.0, broker="XP Investimentos")], [])
    assert {h.ticker for h in p.holdings} == {"PETR4", "AAPL", "VALE3"}
    assert p.committed_shares() == {}

    path = tmp_path / "c.json"
    p.save(path)
    loaded = Portfolio.load(path)
    assert [h.ticker for h in loaded.holdings] == [h.ticker for h in p.holdings] and loaded.updated_at


def test_frame_round_trip_normalizes():
    p = Portfolio([Holding("petr4f", 100, None, broker="BTG"), Holding("PLTR", 10, 20.0, broker="Nomad")])
    back = frame_to_holdings(p.holdings_frame())
    assert [(h.ticker, h.market) for h in back] == [("PETR4", "BR"), ("PLTR", "US")]
