from pathlib import Path

import pandas as pd
import pytest

from options_app.statements import (
    merge_positions,
    parse_number,
    parse_statement,
    parse_watchlist,
    short_calls_by_symbol,
)

DATA = Path(__file__).resolve().parent.parent / "sample_data"


@pytest.mark.parametrize(
    "raw,expected",
    [("$1,234.50", 1234.5), ("(12.5)", -12.5), ("-$3", -3.0), ("--", None), ("+2.5%", 2.5), (7, 7.0), ("", None)],
)
def test_parse_number(raw, expected):
    assert parse_number(raw) == expected


def test_schwab_export_with_preamble_cash_totals_and_options():
    res = parse_statement(DATA / "schwab_positions.csv")
    by_sym = {p.symbol: p for p in res.positions}
    assert set(by_sym) == {"AAPL", "MSFT", "KO"}
    assert by_sym["AAPL"].quantity == 300
    assert by_sym["AAPL"].cost_basis_per_share == pytest.approx(150.0)  # $45,000 / 300
    assert len(res.options) == 1
    assert res.options[0].underlying == "AAPL" and res.options[0].kind == "call" and res.options[0].quantity == -1
    assert short_calls_by_symbol(res.options) == {"AAPL": 1}


def test_fidelity_export_skips_money_market_and_footer():
    res = parse_statement(DATA / "fidelity_positions.csv")
    assert {p.symbol: p.quantity for p in res.positions} == {"AAPL": 100, "NVDA": 250}
    nvda = next(p for p in res.positions if p.symbol == "NVDA")
    assert nvda.cost_basis_per_share == pytest.approx(40.0)


def test_merge_across_statements_weights_cost_basis():
    pos = parse_statement(DATA / "schwab_positions.csv").positions + parse_statement(DATA / "fidelity_positions.csv").positions
    merged = {p.symbol: p for p in merge_positions(pos)}
    assert merged["AAPL"].quantity == 400
    assert merged["AAPL"].cost_basis_per_share == pytest.approx(150.0)


def test_excel_and_occ_symbols(tmp_path):
    df = pd.DataFrame(
        [["My broker statement", "", ""], ["Ticker", "Shares", "Avg Cost"], ["brk.b", "120", "300"],
         ["TSLA261120C00300000", "-1", "5"], ["SPY", "50", "400"]]
    )
    path = tmp_path / "pos.xlsx"
    df.to_excel(path, header=False, index=False)
    res = parse_statement(path)
    assert {p.symbol for p in res.positions} == {"BRK.B", "SPY"}
    assert res.options[0].underlying == "TSLA" and res.options[0].kind == "call"


def test_missing_columns_raises():
    with pytest.raises(ValueError):
        parse_statement(b"a,b\n1,2\n", "x.csv")


def test_watchlist_text_and_file():
    items = parse_watchlist(text="aapl:180, MSFT, KO:60:300")
    assert [(i.symbol, i.target_price, i.shares) for i in items] == [("AAPL", 180, 100), ("MSFT", None, 100), ("KO", 60, 300)]
    file_items = parse_watchlist(DATA / "watchlist.csv")
    assert [i.symbol for i in file_items] == ["GOOGL", "JPM", "COST"]
