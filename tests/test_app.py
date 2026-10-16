"""Headless run of the Streamlit UI against the offline demo provider."""

import shutil
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parent.parent
APP = str(ROOT / "app.py")


@pytest.fixture
def app(tmp_path, monkeypatch):
    monkeypatch.setenv("CARTEIRA_PATH", str(tmp_path / "carteira.json"))
    at = AppTest.from_file(APP, default_timeout=60)
    at.run()
    at.sidebar.radio[0].set_value("Demo (dados sintéticos, offline)").run()
    return at, tmp_path


def _button(at, label):
    return next(b for b in at.button if b.label == label)


def test_example_portfolio_then_calls(app):
    at, tmp = app
    _button(at, "Carregar carteira de exemplo").click().run()
    assert not at.exception and (tmp / "carteira.json").exists()
    _button(at, "Calcular calls").click().run()
    assert not at.exception
    tables = [d.value for d in at.dataframe if "Ativo" in d.value.columns and "Preço justo" in d.value.columns]
    assert tables, "no results table"
    tickers = set(tables[-1]["Ativo"])
    assert {"PETR4", "VALE3", "AAPL", "NVDA", "BOVA11"} <= tickers and "XPML11" not in tickers


def test_puts_for_typed_tickers(app):
    at, _ = app
    at.text_input(key="puts_text").input("PLTR:40, PETR4").run()
    _button(at, "Calcular puts").click().run()
    assert not at.exception
    tables = [d.value for d in at.dataframe if "Preço efetivo se exercido" in d.value.columns]
    assert {"PLTR", "PETR4"} <= set(tables[-1]["Ativo"])
    assert any(m.label == "Preço justo" for m in at.metric)
