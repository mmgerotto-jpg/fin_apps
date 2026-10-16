"""Command-line interface.

Examples:
    python -m options_app.cli calls --statement statements/positions.csv
    python -m options_app.cli puts --watch "AAPL:200, MSFT:400"
    python -m options_app.cli price --spot 100 --strike 105 --days 30 --vol 0.25 --kind call
    python -m options_app.cli calls --statement sample_data/schwab_positions.csv --demo
"""

from __future__ import annotations

import argparse
import sys

import pandas as pd

from .analysis import Settings, analyze_cash_secured_puts, analyze_covered_calls, best_per_symbol
from .market_data import DemoProvider, YahooProvider
from .pricing import bs_greeks, bs_price, fair_value, implied_vol
from .statements import merge_positions, parse_statement, parse_watchlist, short_calls_by_symbol

CALL_COLS = ["symbol", "expiry", "dte", "strike", "spot", "market_mid", "fair_value", "edge", "edge_pct",
             "signal", "market_iv", "model_vol", "delta", "prob_itm", "annualized_yield", "contracts_available", "premium_total"]
PUT_COLS = ["symbol", "expiry", "dte", "strike", "spot", "market_mid", "fair_value", "edge", "edge_pct",
            "signal", "market_iv", "model_vol", "prob_itm", "effective_buy_price", "annualized_yield", "cash_required"]


def _settings(args) -> Settings:
    return Settings(
        min_dte=args.min_dte,
        max_dte=args.max_dte,
        max_expiries=args.max_expiries,
        call_otm_max=args.otm,
        put_otm_max=args.otm,
        vol_method=args.vol_method,
        vol_window=args.vol_window,
        vol_override=args.vol,
        risk_free=args.rate,
        edge_threshold=args.edge,
    )


def _provider(args):
    return DemoProvider() if args.demo else YahooProvider()


def _print(df: pd.DataFrame, cols: list[str], args) -> None:
    if df.empty:
        print("No options matched the filters.")
        return
    view = best_per_symbol(df) if args.best else df
    with pd.option_context("display.max_rows", 500, "display.width", 250, "display.float_format", "{:,.3f}".format):
        print(view[[c for c in cols if c in view.columns]].to_string(index=False))
    if args.out:
        df.to_csv(args.out, index=False)
        print(f"\nFull results written to {args.out}")


def cmd_calls(args) -> int:
    positions, options = [], []
    for path in args.statement:
        res = parse_statement(path)
        positions += res.positions
        options += res.options
        for o in res.options:
            print(f"[{path}] existing option position: {o.description} (qty {o.quantity:g})", file=sys.stderr)
    positions = merge_positions(positions)
    print(f"Holdings: {', '.join(f'{p.symbol} x{p.quantity:g}' for p in positions)}\n")
    df, errors = analyze_covered_calls(
        positions, _provider(args), _settings(args), short_calls=short_calls_by_symbol(options)
    )
    for e in errors:
        print(f"warning: {e}", file=sys.stderr)
    _print(df, CALL_COLS, args)
    return 0


def cmd_puts(args) -> int:
    items = parse_watchlist(args.watch_file) if args.watch_file else parse_watchlist(text=args.watch or "")
    if not items:
        print("Provide --watch 'AAPL:180, MSFT' or --watch-file watchlist.csv", file=sys.stderr)
        return 2
    df, errors = analyze_cash_secured_puts(items, _provider(args), _settings(args))
    for e in errors:
        print(f"warning: {e}", file=sys.stderr)
    _print(df, PUT_COLS, args)
    return 0


def cmd_price(args) -> int:
    T = args.days / 365.0
    fair = fair_value(args.spot, args.strike, T, args.rate_value, args.vol, args.kind, args.div, not args.european)
    euro = bs_price(args.spot, args.strike, T, args.rate_value, args.vol, args.kind, args.div)
    g = bs_greeks(args.spot, args.strike, T, args.rate_value, args.vol, args.kind, args.div)
    print(f"Fair value ({'European' if args.european else 'American'}): {fair:.4f}")
    print(f"Black-Scholes European:  {euro:.4f}")
    print(f"Delta {g.delta:.4f}  Gamma {g.gamma:.4f}  Theta/day {g.theta:.4f}  Vega {g.vega:.4f}  P(ITM) {g.prob_itm:.2%}")
    if args.market is not None:
        iv = implied_vol(args.market, args.spot, args.strike, T, args.rate_value, args.kind, args.div)
        edge = args.market - fair
        print(f"Market {args.market:.4f}  edge {edge:+.4f} ({edge / fair:+.1%})  implied vol {iv:.2%}" if iv else
              f"Market {args.market:.4f}  edge {edge:+.4f}  implied vol: n/a")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="options_app", description="Option fair value vs market price")
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp):
        sp.add_argument("--demo", action="store_true", help="use synthetic offline market data")
        sp.add_argument("--min-dte", type=int, default=7)
        sp.add_argument("--max-dte", type=int, default=60)
        sp.add_argument("--max-expiries", type=int, default=4)
        sp.add_argument("--otm", type=float, default=0.15, help="max %% out of the money (0.15 = 15%%)")
        sp.add_argument("--vol-method", choices=["historical", "ewma", "blend"], default="blend")
        sp.add_argument("--vol-window", type=int, default=60)
        sp.add_argument("--vol", type=float, default=None, help="override volatility for all stocks")
        sp.add_argument("--rate", type=float, default=None, help="risk-free rate (default: 13w T-bill)")
        sp.add_argument("--edge", type=float, default=0.05, help="edge threshold for flags")
        sp.add_argument("--best", action="store_true", help="show only the richest option per stock")
        sp.add_argument("--out", help="write full results to CSV")

    c = sub.add_parser("calls", help="covered calls on holdings from statements")
    c.add_argument("--statement", nargs="+", required=True, help="CSV/XLSX/PDF statement(s)")
    common(c)
    c.set_defaults(func=cmd_calls)

    u = sub.add_parser("puts", help="cash-secured puts on stocks you plan to buy")
    u.add_argument("--watch", help="e.g. 'AAPL:200, MSFT:400' (symbol[:target price[:shares]])")
    u.add_argument("--watch-file", help="CSV with symbol[,target_price][,shares]")
    common(u)
    u.set_defaults(func=cmd_puts)

    pr = sub.add_parser("price", help="price a single option")
    pr.add_argument("--spot", type=float, required=True)
    pr.add_argument("--strike", type=float, required=True)
    pr.add_argument("--days", type=float, required=True)
    pr.add_argument("--vol", type=float, required=True)
    pr.add_argument("--kind", choices=["call", "put"], required=True)
    pr.add_argument("--rate", dest="rate_value", type=float, default=0.04)
    pr.add_argument("--div", type=float, default=0.0)
    pr.add_argument("--market", type=float, help="market price to compare against")
    pr.add_argument("--european", action="store_true")
    pr.set_defaults(func=cmd_price)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
