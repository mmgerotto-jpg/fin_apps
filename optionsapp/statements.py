"""Parse brokerage statements (CSV / PDF / XLSX-as-CSV) into stock holdings.

Handles many broker layouts heuristically: finds a symbol column and a quantity column by header
name; for PDFs, extracts tables, falling back to text-line regex.
Option positions (e.g. OCC symbols "AAPL 250117C00150000") are separated from stocks.
"""
from __future__ import annotations

import csv
import io
import re
from dataclasses import dataclass, asdict

SYMBOL_HDR = ("symbol", "ticker", "security", "instrument", "description")
QTY_HDR = ("quantity", "qty", "shares", "units", "position")
COST_HDR = ("cost basis", "cost", "average cost", "avg cost", "avg price", "average price")
PRICE_HDR = ("price", "last price", "market price", "last")

OCC_RE = re.compile(r"^([A-Z.]{1,6})\s*(\d{6})([CP])(\d{8})$")
TICKER_RE = re.compile(r"^[A-Z]{1,5}(?:[.-][A-Z])?$")
STOP_WORDS = {"USD", "TOTAL", "CASH", "NET", "BUY", "SELL", "ETF", "FUND", "THE", "AND", "INC", "CORP",
              "QTY", "PAGE", "DATE", "ACCOUNT", "STOCK", "STOCKS", "EQUITY", "N", "A", "SPAXX", "FDRXX"}


@dataclass
class Holding:
    symbol: str
    quantity: float
    cost_basis: float | None = None
    last_price: float | None = None

    def to_dict(self):
        return asdict(self)


def _num(s):
    if s is None:
        return None
    s = str(s).strip().replace("$", "").replace(",", "")
    neg = s.startswith("(") and s.endswith(")")
    s = s.strip("()")
    try:
        v = float(s)
    except ValueError:
        return None
    return -v if neg else v


def _find(headers, candidates):
    low = [h.strip().lower() for h in headers]
    for c in candidates:  # exact first, then substring
        for i, h in enumerate(low):
            if h == c:
                return i
    for c in candidates:
        for i, h in enumerate(low):
            if c in h:
                return i
    return None


def _clean_symbol(raw: str):
    raw = (raw or "").strip().upper()
    m = OCC_RE.match(raw)
    if m:
        return None  # option contract, skip for stock holdings
    tok = raw.split()[0] if raw.split() else ""
    tok = tok.strip("()*")
    if TICKER_RE.match(tok) and tok not in STOP_WORDS:
        return tok.replace(".", "-")
    return None


def _rows_to_holdings(rows):
    """rows: list of lists of strings. Locates a header row then reads positions."""
    for hi, row in enumerate(rows):
        si, qi = _find(row, SYMBOL_HDR[:2] + SYMBOL_HDR[2:]), _find(row, QTY_HDR)
        if si is not None and qi is not None and si != qi:
            ci, pi = _find(row, COST_HDR), _find(row, PRICE_HDR)
            out = {}
            for r in rows[hi + 1:]:
                if len(r) <= max(si, qi):
                    continue
                sym, qty = _clean_symbol(r[si]), _num(r[qi])
                if not sym or not qty or qty <= 0:
                    continue
                h = out.setdefault(sym, Holding(sym, 0.0))
                h.quantity += qty
                if ci is not None and ci < len(r):
                    v = _num(r[ci])
                    h.cost_basis = v if v else h.cost_basis
                if pi is not None and pi < len(r) and pi != ci:
                    h.last_price = _num(r[pi]) or h.last_price
            if out:
                return list(out.values())
    return []


def parse_csv(text: str):
    text = text.lstrip("﻿")
    try:
        dialect = csv.Sniffer().sniff(text[:4096], delimiters=",;\t|")
    except csv.Error:
        dialect = csv.excel
    rows = [r for r in csv.reader(io.StringIO(text), dialect)]
    return _rows_to_holdings(rows)


LINE_RE = re.compile(r"^\s*([A-Z]{1,5}(?:[.-][A-Z])?)\s+(?:.*?\s)?(-?[\d,]+(?:\.\d+)?)\s+\$?[\d,]+\.\d{2,4}\b")


def parse_pdf(data: bytes):
    import pdfplumber
    holdings, text_lines = {}, []
    with pdfplumber.open(io.BytesIO(data)) as pdf:
        for page in pdf.pages:
            for table in page.extract_tables() or []:
                for h in _rows_to_holdings([[c or "" for c in row] for row in table]):
                    cur = holdings.setdefault(h.symbol, Holding(h.symbol, 0.0))
                    cur.quantity += h.quantity
                    cur.cost_basis = h.cost_basis or cur.cost_basis
                    cur.last_price = h.last_price or cur.last_price
            text_lines += (page.extract_text() or "").splitlines()
    if holdings:
        return list(holdings.values())
    # Text fallback: "AAPL  Apple Inc  100  190.50  19,050.00"
    for line in text_lines:
        m = LINE_RE.match(line)
        if m:
            sym, qty = _clean_symbol(m.group(1)), _num(m.group(2))
            if sym and qty and qty > 0:
                holdings.setdefault(sym, Holding(sym, 0.0)).quantity += qty
    return list(holdings.values())


def parse_statement(filename: str, data: bytes):
    name = filename.lower()
    if name.endswith(".pdf"):
        return parse_pdf(data)
    if name.endswith((".xlsx", ".xls")):
        import pandas as pd
        df = pd.read_excel(io.BytesIO(data), header=None, dtype=str).fillna("")
        return _rows_to_holdings(df.values.tolist())
    return parse_csv(data.decode("utf-8", errors="replace"))
