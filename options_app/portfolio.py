"""The consolidated portfolio across brokers, saved locally as JSON."""

from __future__ import annotations

import datetime as dt
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import pandas as pd

from .markets import CURRENCY, detect_market, normalize_ticker
from .statements import parse_number

DEFAULT_PATH = Path(__file__).resolve().parent.parent / "data" / "carteira.json"

# Asset types that can have listed options worth analysing.
OPTIONABLE_TYPES = {"acao", "etf", "bdr"}
ASSET_TYPES = ["acao", "etf", "bdr", "fii", "outro"]


@dataclass
class Holding:
    ticker: str
    quantity: float
    avg_price: float | None = None  # preço médio, in the asset's currency
    market: str = ""  # "BR" or "US"
    broker: str = ""
    asset_type: str = "acao"
    name: str = ""

    def __post_init__(self):
        self.ticker = normalize_ticker(self.ticker)
        self.market = self.market or detect_market(self.ticker)

    @property
    def currency(self) -> str:
        return CURRENCY[self.market]


@dataclass
class OptionHolding:
    underlying: str
    kind: str | None  # "call" / "put"
    quantity: float  # in underlying SHARES (US contracts x 100); negative = sold (lançada)
    strike: float | None = None
    expiry: str | None = None  # ISO date
    broker: str = ""
    description: str = ""

    def __post_init__(self):
        self.underlying = normalize_ticker(self.underlying)


@dataclass
class Portfolio:
    holdings: list[Holding] = field(default_factory=list)
    options: list[OptionHolding] = field(default_factory=list)
    updated_at: str = ""

    # ---------- persistence ----------
    def save(self, path: Path | str = DEFAULT_PATH) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.updated_at = dt.datetime.now().isoformat(timespec="seconds")
        path.write_text(json.dumps(asdict(self), ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path | str = DEFAULT_PATH) -> "Portfolio":
        path = Path(path)
        if not path.exists():
            return cls()
        data = json.loads(path.read_text(encoding="utf-8"))
        return cls(
            [Holding(**h) for h in data.get("holdings", [])],
            [OptionHolding(**o) for o in data.get("options", [])],
            data.get("updated_at", ""),
        )

    # ---------- updates ----------
    def replace_broker(self, broker: str, holdings: list[Holding], options: list[OptionHolding]) -> None:
        """Swap in a fresh statement for one broker, keeping the other brokers' positions."""
        key = broker.strip().lower()
        self.holdings = [h for h in self.holdings if h.broker.strip().lower() != key] + holdings
        self.options = [o for o in self.options if o.broker.strip().lower() != key] + options

    # ---------- views ----------
    def consolidated(self) -> list[Holding]:
        """One line per ticker across brokers, with quantity-weighted average price."""
        merged: dict[tuple[str, str], Holding] = {}
        for h in self.holdings:
            k = (h.ticker, h.market)
            if k not in merged:
                merged[k] = Holding(h.ticker, h.quantity, h.avg_price, h.market, h.broker, h.asset_type, h.name)
                continue
            m = merged[k]
            if m.avg_price is not None and h.avg_price is not None and (m.quantity + h.quantity):
                m.avg_price = (m.avg_price * m.quantity + h.avg_price * h.quantity) / (m.quantity + h.quantity)
            elif m.avg_price is None:
                m.avg_price = h.avg_price
            m.quantity += h.quantity
            if h.broker and h.broker not in m.broker.split(", "):
                m.broker = f"{m.broker}, {h.broker}" if m.broker else h.broker
            m.name = m.name or h.name
        return list(merged.values())

    def committed_shares(self) -> dict[str, float]:
        """Shares already covering sold calls, per underlying (not free for new calls)."""
        out: dict[str, float] = {}
        for o in self.options:
            if o.kind == "call" and o.quantity < 0:
                out[o.underlying] = out.get(o.underlying, 0.0) + abs(o.quantity)
        return out

    def holdings_frame(self) -> pd.DataFrame:
        return holdings_to_frame(self.holdings)

    def options_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            [asdict(o) for o in self.options],
            columns=["underlying", "kind", "quantity", "strike", "expiry", "broker", "description"],
        )


HOLDING_COLUMNS = ["ticker", "quantity", "avg_price", "market", "broker", "asset_type", "name"]


def holdings_to_frame(holdings: list[Holding]) -> pd.DataFrame:
    return pd.DataFrame([asdict(h) for h in holdings], columns=HOLDING_COLUMNS)


def frame_to_holdings(df: pd.DataFrame) -> list[Holding]:
    out = []
    for _, r in df.iterrows():
        ticker = str(r.get("ticker") or "").strip()
        qty = parse_number(r.get("quantity"))
        if not ticker or ticker.lower() == "nan" or not qty:
            continue
        market = str(r.get("market") or "").strip().upper()
        asset_type = str(r.get("asset_type") or "").strip().lower()
        out.append(
            Holding(
                ticker,
                qty,
                parse_number(r.get("avg_price")),
                market if market in CURRENCY else "",
                str(r.get("broker") or "").replace("nan", ""),
                asset_type if asset_type in ASSET_TYPES else "acao",
                str(r.get("name") or "").replace("nan", ""),
            )
        )
    return out


def frame_to_options(df: pd.DataFrame) -> list[OptionHolding]:
    out = []
    for _, r in df.iterrows():
        und = str(r.get("underlying") or "").strip()
        if not und or und.lower() == "nan":
            continue
        kind = str(r.get("kind") or "").strip().lower()
        out.append(
            OptionHolding(
                und,
                kind if kind in ("call", "put") else None,
                parse_number(r.get("quantity")) or 0.0,
                parse_number(r.get("strike")),
                str(r.get("expiry") or "") or None,
                str(r.get("broker") or ""),
                str(r.get("description") or ""),
            )
        )
    return out
