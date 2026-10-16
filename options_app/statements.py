"""Parse brokerage statements / position exports into stock holdings.

Supports CSV and Excel exports (Schwab, Fidelity, IBKR, E*TRADE, Vanguard,
Robinhood-style and generic files) by locating the header row and matching
column names against known aliases, plus a best-effort PDF table reader.
Cash lines, totals and option positions are skipped (option positions are
reported separately so existing covered calls are visible).
"""

from __future__ import annotations

import csv
import io
import re
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

SYMBOL_ALIASES = ["symbol", "ticker", "security symbol", "symbol/cusip", "instrument", "financial instrument"]
QTY_ALIASES = ["quantity", "qty", "shares", "qty (quantity)", "quantity held", "position", "share quantity", "units"]
COST_TOTAL_ALIASES = ["cost basis", "cost basis total", "total cost basis", "total cost", "cost basis ($)", "cost"]
COST_AVG_ALIASES = [
    "average cost",
    "avg cost",
    "average cost basis",
    "cost per share",
    "cost/share",
    "unit cost",
    "avg price",
    "average price",
    "cost price",
]
PRICE_ALIASES = ["price", "last price", "last", "market price", "close price", "current price", "mark"]
DESC_ALIASES = ["description", "name", "security description", "security name", "security"]

CALL_KIND, PUT_KIND = "call", "put"

NON_EQUITY_HINTS = (
    "cash",
    "money market",
    "total",
    "pending",
    "account",
    "sweep",
    "fdic",
)

TICKER_RE = re.compile(r"^[A-Z][A-Z0-9]{0,5}([.\-/][A-Z0-9]{1,2})?$")
# e.g. "AAPL 01/17/2025 150.00 C", "AAPL 250117C00150000", "-AAPL250117C150"
OPTION_RE = re.compile(
    r"(\d{1,2}/\d{1,2}/\d{2,4}.*\b[CP](ALL|UT)?\b)|(\d{6}[CP]\d{3,8})|(\b(CALL|PUT)\b)", re.IGNORECASE
)


@dataclass
class Position:
    symbol: str
    quantity: float
    cost_basis_per_share: float | None = None
    last_price: float | None = None
    description: str = ""
    source: str = ""


@dataclass
class OptionPosition:
    underlying: str
    kind: str | None  # "call", "put" or None if it could not be determined
    quantity: float  # negative = short (written)
    description: str = ""


@dataclass
class ParseResult:
    positions: list[Position] = field(default_factory=list)
    options: list[OptionPosition] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)

    @property
    def option_lines(self) -> list[str]:
        return [o.description for o in self.options]
    skipped: list[str] = field(default_factory=list)

    def to_frame(self) -> pd.DataFrame:
        return positions_to_frame(self.positions)


def parse_number(value) -> float | None:
    """Parse '$1,234.56', '(12.5)', '1.2K', '--' etc. into a float."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return None if pd.isna(value) else float(value)
    s = str(value).strip()
    if not s or s in {"-", "--", "N/A", "n/a", "NA"}:
        return None
    neg = s.startswith("(") and s.endswith(")")
    s = s.strip("()").replace("$", "").replace(",", "").replace("%", "").strip()
    if s.startswith("-"):
        neg, s = True, s[1:]
    s = s.lstrip("+")
    mult = 1.0
    if s[-1:].upper() == "K":
        mult, s = 1e3, s[:-1]
    try:
        num = float(s) * mult
    except ValueError:
        return None
    return -num if neg else num


def _norm(col) -> str:
    return re.sub(r"\s+", " ", str(col).strip().lower().replace("﻿", "").replace('"', ""))


def _find_col(columns: list[str], aliases: list[str]) -> str | None:
    normed = {_norm(c): c for c in columns}
    for alias in aliases:
        if alias in normed:
            return normed[alias]
    for alias in aliases:  # looser: alias as prefix ("Quantity (shares)")
        for n, original in normed.items():
            if n.startswith(alias):
                return original
    return None


def _clean_symbol(raw) -> str:
    s = str(raw).strip().upper()
    s = s.rstrip("*").strip()  # Fidelity marks money-market funds with **
    return s.replace("/", ".")


def is_option_text(text: str) -> bool:
    return bool(OPTION_RE.search(text or ""))


def parse_option_line(line: str, quantity: float | None) -> OptionPosition:
    """Extract underlying and call/put from broker option descriptions."""
    text = line.strip().lstrip("-")
    m = re.match(r"^([A-Z][A-Z0-9.]{0,5}?)(\d{6})([CP])\d+", text.replace(" ", ""))
    if m:
        return OptionPosition(m.group(1), CALL_KIND if m.group(3) == "C" else PUT_KIND, quantity or 0.0, line)
    underlying = re.split(r"[\s]+", text)[0].upper()
    upper = f" {text.upper()} "
    if re.search(r"\bCALL\b|\s[\d.]+\s?C\s", upper):
        kind = CALL_KIND
    elif re.search(r"\bPUT\b|\s[\d.]+\s?P\s", upper):
        kind = PUT_KIND
    else:
        kind = None
    return OptionPosition(underlying, kind, quantity or 0.0, line)


def short_calls_by_symbol(options: list[OptionPosition]) -> dict[str, int]:
    """Number of call contracts already written per underlying (these shares are not free to cover)."""
    out: dict[str, int] = {}
    for o in options:
        if o.kind == CALL_KIND and o.quantity < 0:
            out[o.underlying] = out.get(o.underlying, 0) + int(abs(o.quantity))
    return out


def frame_to_result(df: pd.DataFrame, source: str = "") -> ParseResult:
    """Convert a positions table with arbitrary broker column names into a ParseResult."""
    result = ParseResult()
    cols = [str(c) for c in df.columns]
    df = df.copy()
    df.columns = cols
    sym_col = _find_col(cols, SYMBOL_ALIASES)
    qty_col = _find_col(cols, QTY_ALIASES)
    if not sym_col or not qty_col:
        raise ValueError(f"could not find symbol/quantity columns in {source or 'statement'}: {cols}")
    total_cost_col = _find_col(cols, COST_TOTAL_ALIASES)
    avg_cost_col = _find_col(cols, COST_AVG_ALIASES)
    price_col = _find_col(cols, PRICE_ALIASES)
    desc_col = _find_col(cols, DESC_ALIASES)

    for _, row in df.iterrows():
        raw_sym = row.get(sym_col)
        if raw_sym is None or (isinstance(raw_sym, float) and pd.isna(raw_sym)):
            continue
        raw_sym_s = str(raw_sym).strip()
        desc = str(row.get(desc_col, "") or "") if desc_col else ""
        if desc == "nan":
            desc = ""
        line = f"{raw_sym_s} {desc}".strip()
        if is_option_text(raw_sym_s) or (" " in raw_sym_s and is_option_text(line)):
            result.options.append(parse_option_line(line, parse_number(row.get(qty_col))))
            continue
        sym = _clean_symbol(raw_sym_s)
        is_cash_like = any(h in raw_sym_s.lower() for h in NON_EQUITY_HINTS) or any(
            h in desc.lower() for h in ("money market", "cash sweep", "cash & cash")
        )
        if not TICKER_RE.match(sym) or is_cash_like or raw_sym_s.endswith("**"):
            result.skipped.append(line)
            continue
        qty = parse_number(row.get(qty_col))
        if qty is None or qty == 0:
            result.skipped.append(line)
            continue
        cost = None
        if avg_cost_col:
            cost = parse_number(row.get(avg_cost_col))
        if cost is None and total_cost_col:
            total = parse_number(row.get(total_cost_col))
            if total is not None and qty:
                cost = abs(total / qty)
        price = parse_number(row.get(price_col)) if price_col else None
        result.positions.append(
            Position(sym, qty, cost, price, desc if desc else "", source)
        )
    return result


def _find_header_index(lines: list[list[str]]) -> int | None:
    for i, cells in enumerate(lines[:200]):
        normed = [_norm(c) for c in cells]
        if _find_col(normed, SYMBOL_ALIASES) and _find_col(normed, QTY_ALIASES):
            return i
    return None


def parse_csv_text(text: str, source: str = "") -> ParseResult:
    """Parse CSV text, skipping any preamble lines above the header row."""
    rows = list(csv.reader(io.StringIO(text.lstrip("﻿"))))
    header_idx = _find_header_index(rows)
    if header_idx is None:
        raise ValueError(f"no header row with symbol and quantity columns found in {source or 'CSV'}")
    header = [c.strip() for c in rows[header_idx]]
    body = []
    for r in rows[header_idx + 1 :]:
        if not any(c.strip() for c in r):
            if body:  # blank line ends the positions section (IBKR/Schwab multi-section files)
                break
            continue
        if _find_header_index([r]) == 0:  # repeated header (multi-account files)
            continue
        body.append((r + [""] * len(header))[: len(header)])
    df = pd.DataFrame(body, columns=header)
    return frame_to_result(df, source)


def parse_excel(data: bytes | str | Path, source: str = "") -> ParseResult:
    raw = pd.read_excel(io.BytesIO(data) if isinstance(data, bytes) else data, header=None, dtype=str)
    rows = raw.fillna("").astype(str).values.tolist()
    header_idx = _find_header_index(rows)
    if header_idx is None:
        raise ValueError(f"no header row with symbol and quantity columns found in {source or 'Excel file'}")
    df = pd.DataFrame(rows[header_idx + 1 :], columns=[c.strip() for c in rows[header_idx]])
    df = df[df.apply(lambda r: any(str(v).strip() for v in r), axis=1)]
    return frame_to_result(df, source)


def parse_pdf(data: bytes | str | Path, source: str = "") -> ParseResult:
    """Best-effort: extract tables from a PDF statement and parse any positions table found."""
    import pdfplumber

    combined = ParseResult()
    fh = io.BytesIO(data) if isinstance(data, bytes) else data
    with pdfplumber.open(fh) as pdf:
        for page in pdf.pages:
            for table in page.extract_tables() or []:
                rows = [[(c or "").replace("\n", " ") for c in row] for row in table if row]
                idx = _find_header_index(rows)
                if idx is None:
                    continue
                df = pd.DataFrame(rows[idx + 1 :], columns=rows[idx])
                try:
                    part = frame_to_result(df, source)
                except ValueError:
                    continue
                combined.positions += part.positions
                combined.options += part.options
                combined.skipped += part.skipped
    if not combined.positions and not combined.options:
        raise ValueError(f"no positions table found in PDF {source}")
    return combined


def parse_statement(data: bytes | str | Path, filename: str | None = None) -> ParseResult:
    """Parse a statement given raw bytes (with filename) or a path."""
    if isinstance(data, (str, Path)) and filename is None:
        path = Path(data)
        filename = path.name
        data = path.read_bytes()
    name = (filename or "").lower()
    if name.endswith((".xlsx", ".xls", ".xlsm")):
        return parse_excel(data, filename or "")
    if name.endswith(".pdf"):
        return parse_pdf(data, filename or "")
    text = data.decode("utf-8-sig", errors="replace") if isinstance(data, bytes) else str(data)
    return parse_csv_text(text, filename or "")


def merge_positions(positions: list[Position]) -> list[Position]:
    """Combine the same symbol across accounts/statements (quantity-weighted cost basis)."""
    merged: dict[str, Position] = {}
    for p in positions:
        if p.symbol not in merged:
            merged[p.symbol] = Position(p.symbol, p.quantity, p.cost_basis_per_share, p.last_price, p.description, p.source)
            continue
        m = merged[p.symbol]
        if m.cost_basis_per_share is not None and p.cost_basis_per_share is not None:
            total_qty = m.quantity + p.quantity
            if total_qty:
                m.cost_basis_per_share = (
                    m.cost_basis_per_share * m.quantity + p.cost_basis_per_share * p.quantity
                ) / total_qty
        elif m.cost_basis_per_share is None:
            m.cost_basis_per_share = p.cost_basis_per_share
        m.quantity += p.quantity
        m.last_price = m.last_price or p.last_price
        m.description = m.description or p.description
        if p.source and p.source not in m.source:
            m.source = f"{m.source}, {p.source}" if m.source else p.source
    return list(merged.values())


def positions_to_frame(positions: list[Position]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "symbol": p.symbol,
                "quantity": p.quantity,
                "cost_basis": p.cost_basis_per_share,
                "description": p.description,
                "source": p.source,
            }
            for p in positions
        ],
        columns=["symbol", "quantity", "cost_basis", "description", "source"],
    )


def frame_to_positions(df: pd.DataFrame) -> list[Position]:
    out = []
    for _, r in df.iterrows():
        sym = str(r.get("symbol") or "").strip().upper()
        qty = parse_number(r.get("quantity"))
        if not sym or not qty:
            continue
        out.append(Position(sym, qty, parse_number(r.get("cost_basis")), None, str(r.get("description") or ""), str(r.get("source") or "")))
    return out


@dataclass
class WatchItem:
    symbol: str
    target_price: float | None = None  # max price you'd be happy paying
    shares: float = 100


def parse_watchlist(data: bytes | str | Path | None = None, text: str | None = None) -> list[WatchItem]:
    """Watchlist from a CSV (symbol[,target_price][,shares]) or free text 'AAPL:180, MSFT'."""
    items: list[WatchItem] = []
    if text is not None:
        for token in re.split(r"[,\n;]+", text):
            token = token.strip()
            if not token:
                continue
            parts = re.split(r"[:@=\s]+", token)
            sym = _clean_symbol(parts[0])
            target = parse_number(parts[1]) if len(parts) > 1 else None
            shares = parse_number(parts[2]) if len(parts) > 2 else None
            items.append(WatchItem(sym, target, shares or 100))
        return items
    if isinstance(data, (str, Path)):
        data = Path(data).read_bytes()
    df = pd.read_csv(io.BytesIO(data))
    return watchlist_from_frame(df)


def watchlist_from_frame(df: pd.DataFrame) -> list[WatchItem]:
    cols = [str(c) for c in df.columns]
    df = df.copy()
    df.columns = cols
    sym_col = _find_col(cols, SYMBOL_ALIASES)
    if not sym_col:
        raise ValueError("watchlist needs a 'symbol' column")
    target_col = _find_col(cols, ["target_price", "target price", "target", "limit", "buy price", "max price"])
    shares_col = _find_col(cols, ["shares", "quantity", "qty"])
    items = []
    for _, r in df.iterrows():
        sym = _clean_symbol(r.get(sym_col) or "")
        if not sym or sym == "NAN" or not TICKER_RE.match(sym):
            continue
        target = parse_number(r.get(target_col)) if target_col else None
        shares = parse_number(r.get(shares_col)) if shares_col else None
        items.append(WatchItem(sym, target, shares or 100))
    return items
