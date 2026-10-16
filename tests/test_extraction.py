from types import SimpleNamespace

import pandas as pd
import pytest

from options_app.extraction import (
    ExtractedHolding, ExtractedOption, ExtractionError, StatementExtraction, extract_statement, file_to_content,
    to_portfolio_items,
)


class FakeClient:
    def __init__(self, parsed, stop_reason="end_turn"):
        self.calls = []
        self.beta = SimpleNamespace(messages=SimpleNamespace(parse=self._parse))
        self.parsed, self.stop_reason = parsed, stop_reason

    def _parse(self, **kw):
        self.calls.append(kw)
        return SimpleNamespace(parsed_output=self.parsed, stop_reason=self.stop_reason)


SAMPLE = StatementExtraction(
    broker="BTG Pactual",
    statement_date="2026-10-15",
    holdings=[ExtractedHolding(ticker="PETR4", name="PETROBRAS PN", asset_type="acao", market="BR", quantity=800, avg_price=31.2),
              ExtractedHolding(ticker="BOVA11", name="ISHARES BOVA", asset_type="etf", market="BR", quantity=100, avg_price=None)],
    options=[ExtractedOption(underlying="PETR4", kind="call", strike=40.0, expiry="2026-11-20", quantity_shares=-500, description="PETRK400")],
    notes=["CDB ignorado"],
)


def test_pdf_request_shape_and_conversion():
    client = FakeClient(SAMPLE)
    ex = extract_statement("btg.pdf", b"%PDF-1.4 fake", "BTG Pactual", client=client)
    kw = client.calls[0]
    assert kw["model"] == "claude-opus-5-5" and kw["output_format"] is StatementExtraction
    doc = kw["messages"][0]["content"][0]
    assert doc["type"] == "document" and doc["source"]["media_type"] == "application/pdf"
    assert "BTG Pactual" in kw["messages"][0]["content"][-1]["text"]
    holdings, options = to_portfolio_items(ex, "BTG Pactual")
    assert [(h.ticker, h.quantity, h.broker) for h in holdings] == [("PETR4", 800, "BTG Pactual"), ("BOVA11", 100, "BTG Pactual")]
    assert options[0].quantity == -500 and options[0].kind == "call"


def test_refusal_and_truncation_raise():
    with pytest.raises(ExtractionError):
        extract_statement("x.pdf", b"x", client=FakeClient(None, "refusal"))
    with pytest.raises(ExtractionError):
        extract_statement("x.pdf", b"x", client=FakeClient(None, "max_tokens"))


def test_file_to_content_kinds(tmp_path):
    assert file_to_content("print.png", b"\x89PNG")[0]["type"] == "image"
    assert "PETR4" in file_to_content("pos.csv", "Ativo;Qtd\nPETR4;100\n".encode())[0]["text"]
    xlsx = tmp_path / "x.xlsx"
    pd.DataFrame([["Ativo", "Qtd"], ["VALE3", 10]]).to_excel(xlsx, header=False, index=False)
    assert "VALE3" in file_to_content("x.xlsx", xlsx.read_bytes())[0]["text"]
