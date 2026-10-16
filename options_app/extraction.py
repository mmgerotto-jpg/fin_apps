"""Read brokerage statements (BTG, XP, Avenue, Nomad, ...) with Claude.

Statements arrive as PDFs, screenshots or spreadsheet exports in very different
layouts, so instead of one parser per broker we send the file to Claude and get
back a validated, structured list of positions.

Needs an Anthropic API key (ANTHROPIC_API_KEY, or an `ant auth login` profile).
"""

from __future__ import annotations

import base64
import io
from pathlib import Path
from typing import Literal

import pandas as pd
from pydantic import BaseModel, Field

from .portfolio import Holding, OptionHolding

MODEL = "claude-opus-5-5"

SYSTEM_PROMPT = """\
You extract investment positions from brokerage statements for a personal options tool.
The user holds accounts at Brazilian brokers (BTG Pactual, XP Investimentos) that custody
B3-listed assets, and at US-account brokers (Avenue, Nomad) that custody US-listed assets
in USD. Statements may be in Portuguese or English and may be PDFs, screenshots or
spreadsheet exports ("posição consolidada", "extrato de custódia", "portfolio statement").

Report every equity-like position you can see:
- B3 stocks (PETR4, VALE3), ETFs (BOVA11, IVVB11), BDRs (AAPL34) and FIIs (XPML11).
  Fractional tickers ending in F (PETR4F) are the same stock: report the base ticker and
  add the quantity to it. Market "BR".
- US stocks and ETFs (AAPL, PLTR, VOO). Market "US".
- Listed options as `options`: underlying ticker, call/put, strike, expiry and quantity in
  underlying SHARES, negative when sold/written (lançada/vendida). US option quantities are
  in contracts of 100 shares, so multiply by 100; B3 option quantities are already in
  shares. In B3 option tickers (e.g. PETRK380) the fifth letter gives type and month:
  A-L are calls (A=Jan ... L=Dec), M-X are puts (M=Jan ... X=Dec). The underlying is the
  4-letter root plus the share class held in the statement (PETR -> PETR4 if the account
  holds PETR4).

Do not report cash, fixed income (CDB, LCI, LCA, Tesouro Direto, debentures, CRI/CRA),
investment funds, COE, crypto or pension plans; mention them in `notes` only if relevant.

quantity is the number of shares/units held (positive). avg_price is the average purchase
price per share (preço médio / average cost / cost basis per share) in the asset's own
currency; if only a total cost is shown divide it by quantity; if no cost information
exists, use null - never guess. Use the broker name shown on the statement. If the same
ticker appears in several sub-accounts of the same statement, add the quantities.
If shares are lent out (aluguel de ações / BTC como doador / fully paid securities lending),
report the lent quantity in `lent`; `quantity` stays the total owned including lent shares.
Put anything ambiguous (illegible lines, conflicting numbers) in `notes`.
"""


class ExtractedHolding(BaseModel):
    ticker: str = Field(description="Ticker as traded, e.g. PETR4, BOVA11, AAPL34, PLTR")
    name: str = Field(description="Security name as shown")
    asset_type: Literal["acao", "etf", "bdr", "fii", "outro"]
    market: Literal["BR", "US"]
    quantity: float
    avg_price: float | None = Field(description="Average cost per share in the asset currency, or null")
    lent: float = Field(description="Shares currently lent out (aluguel / BTC doador / securities lending), 0 if none")


class ExtractedOption(BaseModel):
    underlying: str
    kind: Literal["call", "put"]
    strike: float | None
    expiry: str | None = Field(description="ISO date YYYY-MM-DD, or null")
    quantity_shares: float = Field(description="Underlying shares; negative when sold/written")
    description: str = Field(description="Option line as shown on the statement")


class StatementExtraction(BaseModel):
    broker: str
    statement_date: str | None = Field(description="Position date, ISO YYYY-MM-DD, or null")
    holdings: list[ExtractedHolding]
    options: list[ExtractedOption]
    notes: list[str]


class ExtractionError(RuntimeError):
    pass


_IMAGE_TYPES = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".webp": "image/webp", ".gif": "image/gif"}


def file_to_content(name: str, data: bytes) -> list[dict]:
    """Turn an uploaded file into Messages API content blocks."""
    suffix = Path(name).suffix.lower()
    if suffix == ".pdf":
        b64 = base64.standard_b64encode(data).decode()
        return [{"type": "document", "source": {"type": "base64", "media_type": "application/pdf", "data": b64}, "title": name}]
    if suffix in _IMAGE_TYPES:
        b64 = base64.standard_b64encode(data).decode()
        return [{"type": "image", "source": {"type": "base64", "media_type": _IMAGE_TYPES[suffix], "data": b64}}]
    if suffix in (".xlsx", ".xls", ".xlsm"):
        sheets = pd.read_excel(io.BytesIO(data), sheet_name=None, header=None, dtype=str)
        text = "\n\n".join(f"# sheet: {sn}\n{df.fillna('').to_csv(index=False, header=False)}" for sn, df in sheets.items())
    else:
        text = data.decode("utf-8-sig", errors="replace")
    return [{"type": "text", "text": f"<file name=\"{name}\">\n{text}\n</file>"}]


def extract_statement(name: str, data: bytes, broker_hint: str | None = None, client=None, model: str = MODEL) -> StatementExtraction:
    """Send one statement to Claude and return the structured positions."""
    import anthropic

    client = client or anthropic.Anthropic()
    hint = f"The user says this statement is from {broker_hint}. " if broker_hint else ""
    content = file_to_content(name, data) + [
        {"type": "text", "text": f"{hint}Extract the positions from this statement ({name})."}
    ]
    response = client.beta.messages.parse(
        model=model,
        max_tokens=16000,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": content}],
        output_config={"effort": "medium"},
        output_format=StatementExtraction,
        betas=["server-side-fallback-2026-07-01"],
        fallbacks="default",
    )
    if response.stop_reason == "refusal":
        raise ExtractionError(f"{name}: Claude recusou a leitura do arquivo")
    if response.stop_reason == "max_tokens":
        raise ExtractionError(f"{name}: resposta truncada (extrato muito grande); divida o arquivo")
    if response.parsed_output is None:
        raise ExtractionError(f"{name}: não foi possível interpretar a resposta")
    return response.parsed_output


def to_portfolio_items(ex: StatementExtraction, broker: str | None = None) -> tuple[list[Holding], list[OptionHolding]]:
    broker = (broker or ex.broker or "").strip()
    holdings = [
        Holding(h.ticker, h.quantity, h.avg_price, h.market, broker, h.asset_type, h.name, h.lent)
        for h in ex.holdings
        if h.quantity
    ]
    options = [
        OptionHolding(o.underlying, o.kind, o.quantity_shares, o.strike, o.expiry, broker, o.description)
        for o in ex.options
    ]
    return holdings, options
