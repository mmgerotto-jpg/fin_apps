"""Linha de comando.

Exemplos:
    python -m options_app.cli ler-extratos extratos/btg.pdf extratos/avenue.pdf --salvar
    python -m options_app.cli calls
    python -m options_app.cli puts "PLTR, PETR4:34"
    python -m options_app.cli preco --spot 37.5 --strike 40 --dias 30 --vol 0.3 --tipo call --juro 0.13
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from .analysis import Settings, call_opportunities, put_opportunities, suggestions
from .market_data import DemoProvider, YahooProvider
from .portfolio import DEFAULT_PATH, Portfolio, holdings_to_frame
from .pricing import bs_greeks, fair_value
from .statements import parse_watchlist

CALL_COLS = ["ticker", "currency", "expiry", "dte", "strike", "spot", "fair_price", "premium_pct", "annualized_yield",
             "prob_exercise", "avg_price", "return_if_exercised", "contracts", "premium_total", "market_bid", "market_ask"]
PUT_COLS = ["ticker", "currency", "expiry", "dte", "strike", "spot", "fair_price", "premium_pct", "annualized_yield",
            "prob_exercise", "effective_price", "discount_to_spot", "cash_required", "market_bid", "market_ask"]


def _settings(args, **kw) -> Settings:
    return Settings(min_dte=args.min_dte, max_dte=args.max_dte, max_expiries=args.vencimentos,
                    vol_method=args.vol, max_prob_exercise=args.prob_max, **kw)


def _provider(args):
    return DemoProvider() if args.demo else YahooProvider()


def _print(df: pd.DataFrame, cols: list[str], args, kind: str) -> None:
    if df.empty:
        print("Nenhuma opção dentro dos filtros.")
        return
    view = df if args.todos else suggestions(df, _settings(args), kind)
    with pd.option_context("display.max_rows", 500, "display.width", 250, "display.float_format", "{:,.3f}".format):
        print(view[[c for c in cols if c in view.columns]].to_string(index=False))
    if args.csv:
        df.to_csv(args.csv, index=False)
        print(f"\nResultado completo em {args.csv}")


def cmd_read(args) -> int:
    from .extraction import extract_statement, to_portfolio_items

    p = Portfolio.load(args.carteira)
    for path in args.arquivos:
        ex = extract_statement(Path(path).name, Path(path).read_bytes(), args.corretora)
        holdings, options = to_portfolio_items(ex, args.corretora)
        print(f"\n== {path}: {ex.broker} ({ex.statement_date or 's/ data'})")
        print(holdings_to_frame(holdings).to_string(index=False))
        for o in options:
            print(f"  opção: {o.description} ({o.quantity:g} ações)")
        for n in ex.notes:
            print(f"  nota: {n}")
        p.replace_broker(args.corretora or ex.broker, holdings, options)
    if args.salvar:
        p.save(args.carteira)
        print(f"\nCarteira salva em {args.carteira}")
    return 0


def cmd_calls(args) -> int:
    p = Portfolio.load(args.carteira)
    if not p.holdings:
        print(f"Carteira vazia ({args.carteira}). Rode 'ler-extratos' antes.", file=sys.stderr)
        return 2
    settings = _settings(args, call_otm_max=args.otm)
    df, warns = call_opportunities(p.consolidated(), _provider(args), settings, committed_shares=p.committed_shares())
    for w in warns:
        print(f"aviso: {w}", file=sys.stderr)
    _print(df, CALL_COLS, args, "call")
    return 0


def cmd_puts(args) -> int:
    settings = _settings(args, put_otm_max=args.otm)
    frames = []
    for item in parse_watchlist(text=args.ativos):
        try:
            frames.append(put_opportunities(item.symbol, _provider(args), settings, target_price=item.target_price,
                                            shares=item.shares))
        except Exception as exc:
            print(f"aviso: {item.symbol}: {exc}", file=sys.stderr)
    _print(pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(), PUT_COLS, args, "put")
    return 0


def cmd_price(args) -> int:
    T = args.dias / 365.0
    fv = fair_value(args.spot, args.strike, T, args.juro, args.vol, args.tipo, args.div)
    g = bs_greeks(args.spot, args.strike, T, args.juro, args.vol, args.tipo, args.div)
    print(f"Preço justo: {fv:.4f}  (por contrato: {fv * 100:,.2f})")
    print(f"Delta {g.delta:.3f}  Theta/dia {g.theta:.4f}  Vega {g.vega:.4f}  Prob. exercício {g.prob_itm:.1%}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="options_app", description="Preço justo de opções (calls cobertas e puts)")
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp):
        sp.add_argument("--carteira", default=str(DEFAULT_PATH))
        sp.add_argument("--demo", action="store_true", help="dados sintéticos offline")
        sp.add_argument("--min-dte", type=int, default=15)
        sp.add_argument("--max-dte", type=int, default=75)
        sp.add_argument("--vencimentos", type=int, default=3)
        sp.add_argument("--otm", type=float, default=0.20, help="distância máx. do strike (0.20 = 20%%)")
        sp.add_argument("--vol", choices=["blend", "historical", "ewma"], default="blend")
        sp.add_argument("--prob-max", type=float, default=0.30, help="prob. máx. de exercício nas sugestões")
        sp.add_argument("--todos", action="store_true", help="mostrar todos os strikes, não só a sugestão")
        sp.add_argument("--csv", help="salvar resultado completo em CSV")

    r = sub.add_parser("ler-extratos", help="ler extratos com o Claude e atualizar a carteira")
    r.add_argument("arquivos", nargs="+")
    r.add_argument("--corretora", help="BTG Pactual, XP Investimentos, Avenue, Nomad…")
    r.add_argument("--carteira", default=str(DEFAULT_PATH))
    r.add_argument("--salvar", action="store_true")
    r.set_defaults(func=cmd_read)

    c = sub.add_parser("calls", help="calls cobertas sobre a carteira")
    common(c)
    c.set_defaults(func=cmd_calls)

    u = sub.add_parser("puts", help="puts para comprar ações (B3 ou EUA)")
    u.add_argument("ativos", help="ex.: 'PLTR, PETR4:34:200' (ticker[:preço alvo[:quantidade]])")
    common(u)
    u.set_defaults(func=cmd_puts)

    pr = sub.add_parser("preco", help="preço justo de uma opção")
    pr.add_argument("--spot", type=float, required=True)
    pr.add_argument("--strike", type=float, required=True)
    pr.add_argument("--dias", type=float, required=True)
    pr.add_argument("--vol", type=float, required=True)
    pr.add_argument("--tipo", choices=["call", "put"], required=True)
    pr.add_argument("--juro", type=float, default=0.04, help="juro contínuo a.a.")
    pr.add_argument("--div", type=float, default=0.0)
    pr.set_defaults(func=cmd_price)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
