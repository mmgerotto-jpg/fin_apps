"""Headless run of the Streamlit UI against the offline demo provider."""

from pathlib import Path

from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parent.parent / "app.py")


def test_app_runs_puts_and_calculator_in_demo_mode():
    at = AppTest.from_file(APP, default_timeout=60)
    at.run()
    assert not at.exception
    at.sidebar.radio[0].set_value("Demo (offline synthetic)").run()
    puts_button = next(b for b in at.button if b.label == "Analyse puts")
    puts_button.click().run()
    assert not at.exception
    assert any(m.label == "Options analysed" and int(m.value) > 0 for m in at.metric)
    assert any(m.label.startswith("Fair value") for m in at.metric)


def test_app_covered_calls_from_sample_statements():
    at = AppTest.from_file(APP, default_timeout=60)
    at.run()
    at.sidebar.radio[0].set_value("Demo (offline synthetic)").run()
    next(c for c in at.checkbox if c.label.startswith("Use the sample statements")).check().run()
    assert not at.exception
    next(b for b in at.button if b.label == "Analyse covered calls").click().run()
    assert not at.exception
    assert not at.warning, [w.value for w in at.warning]
    analysed = next(m for m in at.metric if m.label == "Options analysed")
    assert int(analysed.value) > 0
    results = [d.value for d in at.dataframe if "expiry" in d.value.columns]
    symbols = set(results[-1]["symbol"])  # the "All options" table
    assert {"AAPL", "MSFT", "NVDA", "KO"} <= symbols
