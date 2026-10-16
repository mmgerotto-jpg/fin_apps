"""B3 option quotes from opcoes.net.br (reference prices for Brazilian options).

Uses the JSON endpoint behind https://opcoes.net.br/opcoes/bovespa/<TICKER>:

    /listaopcoes/completa?idAcao=PETR4&listarVencimentos=true&cotacoes=false
        -> data.vencimentos: [{"value": "2026-11-20", ...}, ...]
    /listaopcoes/completa?idAcao=PETR4&listarVencimentos=false&cotacoes=true&vencimentos=2026-11-20
        -> data.cotacoesOpcoes: [[ "PETRK400_2026", <...>, "CALL", "A", <...>, strike, <...>, <...>, last, trades, volume, ...], ...]

The site has no documented API, so parsing is defensive: option type comes from
the series letter of the ticker (A-L call, M-X put), and rows that don't look
like options are skipped. Run `python -m options_app.cli opcoes-net PETR4` to see
what the site returns if quotes stop showing up.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from http.cookiejar import CookieJar
from pathlib import Path

import numpy as np
import pandas as pd

BASE_URL = "https://opcoes.net.br/listaopcoes/completa"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126 Safari/537.36",
    "Accept": "application/json, text/javascript, */*; q=0.01",
    "X-Requested-With": "XMLHttpRequest",
}
DEFAULT_CACHE_DIR = Path(__file__).resolve().parent.parent / "data" / "cache" / "opcoes_net"
CHAIN_COLUMNS = ["strike", "bid", "ask", "last", "volume", "open_interest", "market_iv", "contract"]

_OPTION_TICKER = re.compile(r"^([A-Z]{4})([A-X])(\d{2,4}[A-Z]?)\b")
# Positions observed in data.cotacoesOpcoes rows.
STRIKE_IDX, LAST_IDX, TRADES_IDX, VOLUME_IDX = 5, 8, 9, 10


def option_kind_from_ticker(ticker: str) -> str | None:
    """B3 series letter: A-L calls (Jan-Dec), M-X puts (Jan-Dec)."""
    m = _OPTION_TICKER.match(ticker.upper())
    if not m:
        return None
    return "call" if m.group(2) <= "L" else "put"


def _num(v) -> float | None:
    if v is None or isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        return float(v)
    s = str(v).strip().replace(".", "").replace(",", ".") if "," in str(v) else str(v).strip()
    try:
        return float(s)
    except ValueError:
        return None


def _parse_date(v) -> dt.date | None:
    if isinstance(v, dict):
        v = v.get("value") or v.get("dataVencimento") or v.get("data") or v.get("text")
    s = str(v or "").strip()
    m = re.search(r"(\d{4})-(\d{2})-(\d{2})", s)
    if m:
        return dt.date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
    m = re.search(r"(\d{2})/(\d{2})/(\d{4})", s)
    if m:
        return dt.date(int(m.group(3)), int(m.group(2)), int(m.group(1)))
    return None


def parse_expirations(payload: dict) -> list[dt.date]:
    data = payload.get("data", payload)
    out = {d for d in (_parse_date(v) for v in data.get("vencimentos") or []) if d}
    return sorted(out)


def _row_fields(row) -> tuple[str, float | None, float | None, float | None, float | None]:
    """(ticker, strike, last, trades, volume) from a list row or a dict row."""
    if isinstance(row, dict):
        lower = {str(k).lower(): v for k, v in row.items()}

        def pick(*names):
            return next((lower[n] for n in names if n in lower), None)

        return (
            str(pick("ticker", "codigo", "ativo", "symbol") or ""),
            _num(pick("strike", "exercicio", "precoexercicio")),
            _num(pick("ultimo", "last", "preco", "fechamento")),
            _num(pick("negocios", "numneg", "trades")),
            _num(pick("volume", "volfinanceiro", "vol")),
        )
    row = list(row)

    def at(i):
        return _num(row[i]) if i < len(row) else None

    return str(row[0] if row else ""), at(STRIKE_IDX), at(LAST_IDX), at(TRADES_IDX), at(VOLUME_IDX)


def parse_chain(payload: dict, kind: str) -> pd.DataFrame:
    data = payload.get("data", payload)
    rows = []
    for row in data.get("cotacoesOpcoes") or []:
        ticker, strike, last, trades, volume = _row_fields(row)
        ticker = ticker.split("_")[0].strip().upper()
        if option_kind_from_ticker(ticker) != kind or not strike or strike <= 0:
            continue
        rows.append(
            {
                "strike": strike,
                "bid": 0.0,  # opcoes.net.br list shows last trade, not the book
                "ask": 0.0,
                "last": last if last and last > 0 else np.nan,
                "volume": volume or 0.0,
                "open_interest": trades or 0.0,  # number of trades today (liquidity hint)
                "market_iv": np.nan,
                "contract": ticker,
            }
        )
    df = pd.DataFrame(rows, columns=CHAIN_COLUMNS)
    # Several series can share a strike (weekly/adjusted); keep the most traded one.
    if not df.empty:
        df = df.sort_values("volume", ascending=False).drop_duplicates("strike").sort_values("strike")
    return df.reset_index(drop=True)


class OpcoesNetClient:
    """HTTP client with throttling, retry on 429 and a short-lived disk cache.

    The site rate-limits bursts (HTTP 429), so requests are spaced by
    `min_interval` seconds, a 429/503 is retried after the delay the site asks for
    (Retry-After) or an exponential backoff, and responses are cached on disk for
    `cache_ttl` seconds so recalculating doesn't hit the site again.
    """

    def __init__(
        self,
        timeout: float = 15.0,
        opener=None,
        min_interval: float = 1.5,
        max_retries: int = 3,
        cache_dir: Path | str | None = DEFAULT_CACHE_DIR,
        cache_ttl: float = 15 * 60,
        sleep=time.sleep,
        clock=time.monotonic,
    ):
        self.timeout = timeout
        if opener is None:
            cookie_opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(CookieJar()))
            opener = cookie_opener.open
        self._open = opener
        self.min_interval = min_interval
        self.max_retries = max_retries
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.cache_ttl = cache_ttl
        self._sleep, self._clock = sleep, clock
        self._last_request = None
        self._cache: dict[str, dict] = {}

    # ---------- cache ----------
    def _cache_path(self, url: str) -> Path | None:
        if not self.cache_dir:
            return None
        return self.cache_dir / (hashlib.sha1(url.encode()).hexdigest() + ".json")

    def _read_cache(self, url: str) -> dict | None:
        if url in self._cache:
            return self._cache[url]
        path = self._cache_path(url)
        try:
            if path and path.exists() and time.time() - path.stat().st_mtime < self.cache_ttl:
                payload = json.loads(path.read_text(encoding="utf-8"))
                self._cache[url] = payload
                return payload
        except (OSError, ValueError):
            pass
        return None

    def _write_cache(self, url: str, payload: dict) -> None:
        self._cache[url] = payload
        path = self._cache_path(url)
        if path:
            try:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(payload), encoding="utf-8")
            except OSError:
                pass

    # ---------- HTTP ----------
    def _throttle(self) -> None:
        if self._last_request is not None:
            wait = self.min_interval - (self._clock() - self._last_request)
            if wait > 0:
                self._sleep(wait)
        self._last_request = self._clock()

    def _get_json(self, url: str, ticker: str) -> dict:
        headers = dict(HEADERS, Referer=f"https://opcoes.net.br/opcoes/bovespa/{ticker}")
        for attempt in range(self.max_retries + 1):
            self._throttle()
            try:
                with self._open(urllib.request.Request(url, headers=headers), timeout=self.timeout) as resp:
                    return json.loads(resp.read().decode("utf-8"))
            except urllib.error.HTTPError as exc:
                if exc.code not in (429, 503) or attempt == self.max_retries:
                    raise
                retry_after = _num(exc.headers.get("Retry-After") if exc.headers else None)
                self._sleep(min(retry_after if retry_after else 3.0 * 2**attempt, 30.0))
        raise RuntimeError("unreachable")

    def fetch(self, ticker: str, expiry: dt.date | None = None) -> dict:
        params = {"idAcao": ticker, "listarVencimentos": "true" if expiry is None else "false",
                  "cotacoes": "false" if expiry is None else "true"}
        if expiry is not None:
            params["vencimentos"] = expiry.isoformat()
        url = f"{BASE_URL}?{urllib.parse.urlencode(params)}"
        cached = self._read_cache(url)
        if cached is not None:
            return cached
        payload = self._get_json(url, ticker)
        if isinstance(payload, dict) and payload.get("success") is False:
            raise LookupError(f"opcoes.net.br: {payload.get('message') or 'sem dados'} para {ticker}")
        self._write_cache(url, payload)
        return payload

    def expirations(self, ticker: str) -> list[dt.date]:
        return parse_expirations(self.fetch(ticker))

    def chain(self, ticker: str, expiry: dt.date, kind: str) -> pd.DataFrame:
        return parse_chain(self.fetch(ticker, expiry), kind)
