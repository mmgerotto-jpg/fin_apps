"""App local: preço justo de opções para lançamento coberto e puts de compra.

Rodar com:  streamlit run app.py
"""

from __future__ import annotations

import datetime as dt
import os
from pathlib import Path

import pandas as pd
import streamlit as st

from options_app.analysis import Settings, call_opportunities, put_opportunities, suggestions
from options_app.market_data import DemoProvider, YahooProvider, annual_to_continuous
from options_app.markets import BR, CURRENCY_SYMBOL, US, detect_market, normalize_ticker
from options_app.portfolio import (
    ASSET_TYPES,
    DEFAULT_PATH,
    Holding,
    OptionHolding,
    Portfolio,
    frame_to_holdings,
    frame_to_options,
    holdings_to_frame,
)
from options_app.pricing import bs_greeks, fair_value, implied_vol
from options_app.statements import parse_statement, parse_watchlist
from options_app.volatility import estimate_vol

SAMPLE_DIR = Path(__file__).resolve().parent / "sample_data"
PORTFOLIO_PATH = Path(os.environ.get("CARTEIRA_PATH", DEFAULT_PATH))
BROKERS = ["Detectar", "BTG Pactual", "XP Investimentos", "Avenue", "Nomad", "Outra"]

st.set_page_config(page_title="Opções – preço justo", layout="wide")
st.title("Preço justo de opções")
st.caption(
    "Referência de preço para **lançar calls cobertas** sobre a sua carteira e **vender puts** de ações que você "
    "quer comprar. Não é ferramenta de arbitragem: o preço justo serve para definir o preço da sua ordem e comparar "
    "strikes e vencimentos por rendimento × probabilidade de exercício."
)


# ---------- barra lateral ----------
with st.sidebar:
    st.header("Dados de mercado")
    source = st.radio("Fonte", ["Yahoo Finance (atraso ~15 min)", "Demo (dados sintéticos, offline)"], index=0)
    use_demo = source.startswith("Demo")

    st.header("Leitura de extratos")
    api_key = st.text_input(
        "Chave da API Anthropic", type="password",
        help="Usada para o Claude ler os extratos. Se vazio, usa a variável ANTHROPIC_API_KEY. Não é salva.",
    )

    st.header("Modelo")
    vol_method = st.selectbox(
        "Volatilidade usada no preço justo", ["blend", "historical", "ewma"],
        format_func={"blend": "Mista (60d + 1 ano)", "historical": "Histórica (janela)", "ewma": "EWMA (λ=0,94)"}.get,
    )
    vol_window = st.slider("Janela da vol. histórica (pregões)", 20, 252, 60, step=5)
    american = st.checkbox("Exercício americano (árvore binomial)", value=True)
    rate_override = {}
    if st.checkbox("Definir Selic manualmente (padrão: Banco Central)"):
        rate_override[BR] = st.number_input("Selic (% a.a.)", 0.0, 40.0, 14.25, 0.25) / 100
    if st.checkbox("Definir juro dos EUA manualmente (padrão: T-bill 13 semanas)"):
        rate_override[US] = st.number_input("Juro EUA (% a.a.)", 0.0, 15.0, 4.0, 0.25) / 100

    st.header("Vencimentos")
    dte_range = st.slider("Dias até o vencimento", 0, 365, (15, 75))
    max_expiries = st.slider("Máx. de vencimentos por ativo", 1, 8, 3)


@st.cache_resource
def get_provider(demo: bool):
    return DemoProvider() if demo else YahooProvider()


provider = get_provider(use_demo)


def base_settings(**kw) -> Settings:
    return Settings(
        min_dte=dte_range[0], max_dte=dte_range[1], max_expiries=max_expiries, vol_method=vol_method,
        vol_window=vol_window, american=american, rate_override=rate_override, **kw,
    )


def load_portfolio() -> Portfolio:
    if "portfolio" not in st.session_state:
        st.session_state["portfolio"] = Portfolio.load(PORTFOLIO_PATH)
    return st.session_state["portfolio"]


def save_portfolio(p: Portfolio) -> None:
    p.save(PORTFOLIO_PATH)
    st.session_state["portfolio"] = p


COLUMNS = {
    "ticker": ("Ativo", None), "currency": ("Moeda", None), "expiry": ("Vencimento", None), "dte": ("Dias", None),
    "strike": ("Strike", "{:,.2f}"), "spot": ("Cotação", "{:,.2f}"), "fair_price": ("Preço justo", "{:,.2f}"),
    "fair_per_contract": ("Justo / contrato", "{:,.0f}"), "premium_pct": ("Prêmio %", "{:.2%}"),
    "annualized_yield": ("Rend. anualizado", "{:.1%}"), "prob_exercise": ("Prob. exercício", "{:.0%}"),
    "delta": ("Delta", "{:.2f}"), "theta_day": ("Theta/dia", "{:.3f}"), "model_vol": ("Vol. usada", "{:.0%}"),
    "upside_to_strike": ("Alta até strike", "{:.1%}"), "avg_price": ("Preço médio", "{:,.2f}"),
    "return_if_exercised": ("Retorno se exercido", "{:.1%}"), "contracts": ("Contratos", None),
    "free_shares": ("Ações livres", "{:,.0f}"), "premium_total": ("Prêmio total (justo)", "{:,.0f}"),
    "effective_price": ("Preço efetivo se exercido", "{:,.2f}"), "discount_to_spot": ("Desconto vs cotação", "{:.1%}"),
    "target_price": ("Preço alvo", "{:,.2f}"), "meets_target": ("Atinge alvo", None),
    "cash_required": ("Caixa necessário", "{:,.0f}"), "market_bid": ("Compra (mercado)", "{:,.2f}"),
    "market_ask": ("Venda (mercado)", "{:,.2f}"), "market_iv": ("Vol. implícita mercado", "{:.0%}"),
    "strike_source": ("Origem do strike", None),
}


def show(df: pd.DataFrame, cols: list[str], key: str) -> None:
    view = df[[c for c in cols if c in df.columns]].copy()
    if "strike_source" in view:
        view["strike_source"] = view["strike_source"].map({"listado": "listado", "grade": "teórico"})
    fmt = {COLUMNS[c][0]: COLUMNS[c][1] for c in view.columns if COLUMNS.get(c, (None, None))[1]}
    view = view.rename(columns={c: COLUMNS[c][0] for c in view.columns if c in COLUMNS})
    st.dataframe(view.style.format(fmt, na_rep="–"), width="stretch", hide_index=True)
    st.download_button("Baixar CSV", df.to_csv(index=False).encode(), f"{key}.csv", "text/csv", key=f"dl_{key}")


def chart(df: pd.DataFrame, key: str) -> None:
    tickers = sorted(df["ticker"].unique())
    c1, c2 = st.columns(2)
    t = c1.selectbox("Ativo", tickers, key=f"ct_{key}")
    sub = df[df["ticker"] == t]
    exp = c2.selectbox("Vencimento", sorted(sub["expiry"].unique()), key=f"ce_{key}")
    d = sub[sub["expiry"] == exp].set_index("strike")
    cols = ["fair_price"] + (["market_bid", "market_ask"] if d["market_bid"].notna().any() else [])
    st.line_chart(d[cols].rename(columns={c: COLUMNS[c][0] for c in cols}), x_label="strike", y_label="prêmio por ação")


tab_wallet, tab_calls, tab_puts, tab_calc = st.tabs(
    ["Carteira", "Calls cobertas", "Puts para comprar", "Calculadora"]
)

# ======================= CARTEIRA =======================
with tab_wallet:
    st.subheader("1. Ler extratos")
    st.markdown(
        "Envie os extratos de posição (PDF, print da tela, CSV ou Excel) da **BTG, XP, Avenue e Nomad**. "
        "O Claude lê cada arquivo e devolve ações, ETFs, BDRs, FIIs e opções. Renda fixa, fundos e caixa são ignorados. "
        "Ao salvar, as posições de cada corretora **substituem** as anteriores da mesma corretora."
    )
    files = st.file_uploader(
        "Extratos", type=["pdf", "png", "jpg", "jpeg", "webp", "csv", "xlsx", "xls", "txt"], accept_multiple_files=True
    )
    brokers_for = {}
    for f in files or []:
        guess = next((b for b in BROKERS[1:5] if b.split()[0].lower() in f.name.lower()), "Detectar")
        brokers_for[f.name] = st.selectbox(f"Corretora de **{f.name}**", BROKERS, BROKERS.index(guess), key=f"b_{f.name}")

    c1, c2 = st.columns([1, 3])
    if c1.button("Ler extratos com Claude", type="primary", disabled=not files):
        from options_app.extraction import ExtractionError, extract_statement, to_portfolio_items

        import anthropic

        client = anthropic.Anthropic(api_key=api_key) if api_key else None
        results = []
        for f in files:
            hint = brokers_for[f.name]
            hint = None if hint in ("Detectar", "Outra") else hint
            with st.spinner(f"Claude lendo {f.name}…"):
                try:
                    ex = extract_statement(f.name, f.getvalue(), hint, client=client)
                    h, o = to_portfolio_items(ex, hint)
                    results.append({"file": f.name, "broker": hint or ex.broker, "date": ex.statement_date,
                                    "holdings": h, "options": o, "notes": ex.notes})
                except (ExtractionError, anthropic.APIError, anthropic.AnthropicError) as exc:
                    results.append({"file": f.name, "error": str(exc)})
                except Exception as exc:  # unreadable file etc.
                    results.append({"file": f.name, "error": f"{type(exc).__name__}: {exc}"})
        st.session_state["extracted"] = results
    if c2.button("Leitor simples (CSV/Excel, sem Claude)", disabled=not files):
        results = []
        for f in files:
            broker = brokers_for[f.name] if brokers_for[f.name] not in ("Detectar", "Outra") else Path(f.name).stem
            try:
                res = parse_statement(f.getvalue(), f.name)
                h = [Holding(p.symbol, p.quantity, p.cost_basis_per_share, "", broker, "acao", p.description)
                     for p in res.positions]
                o = [OptionHolding(x.underlying, x.kind, x.quantity * (100 if detect_market(x.underlying) == US else 1),
                                   None, None, broker, x.description) for x in res.options]
                results.append({"file": f.name, "broker": broker, "date": None, "holdings": h, "options": o,
                                "notes": ["Leitor simples: confira tipo de ativo e quantidades."]})
            except Exception as exc:
                results.append({"file": f.name, "error": str(exc)})
        st.session_state["extracted"] = results

    if st.session_state.get("extracted"):
        ok = [r for r in st.session_state["extracted"] if "error" not in r]
        for r in st.session_state["extracted"]:
            if "error" in r:
                st.error(f"{r['file']}: {r['error']}")
                continue
            with st.expander(f"{r['file']} → {r['broker']} ({len(r['holdings'])} ativos, {len(r['options'])} opções)"
                             + (f" · posição de {r['date']}" if r["date"] else ""), expanded=True):
                st.dataframe(holdings_to_frame(r["holdings"]), hide_index=True, width="stretch")
                if r["options"]:
                    st.dataframe(Portfolio(options=r["options"]).options_frame(), hide_index=True, width="stretch")
                for n in r["notes"]:
                    st.info(n)
        if ok and st.button("Salvar na carteira", type="primary"):
            p = load_portfolio()
            for r in ok:
                p.replace_broker(r["broker"], r["holdings"], r["options"])
            save_portfolio(p)
            st.session_state["extracted"] = []
            st.success("Carteira atualizada.")
            st.rerun()

    st.subheader("2. Minha carteira")
    p = load_portfolio()
    st.caption(f"Salva em `{PORTFOLIO_PATH}`" + (f" · atualizada em {p.updated_at}" if p.updated_at else ""))
    hold_edit = st.data_editor(
        p.holdings_frame(), num_rows="dynamic", width="stretch", key=f"hold_{p.updated_at}",
        column_config={
            "ticker": st.column_config.TextColumn("Ativo"),
            "quantity": st.column_config.NumberColumn("Quantidade", format="%.0f"),
            "avg_price": st.column_config.NumberColumn("Preço médio", format="%.2f"),
            "lent": st.column_config.NumberColumn("Em aluguel", format="%.0f",
                                                  help="Ações emprestadas (BTC doador): não cobrem calls"),
            "market": st.column_config.SelectboxColumn("Mercado", options=[BR, US]),
            "broker": st.column_config.TextColumn("Corretora"),
            "asset_type": st.column_config.SelectboxColumn("Tipo", options=ASSET_TYPES),
            "name": st.column_config.TextColumn("Nome"),
        },
    )
    st.markdown("**Opções em aberto** (quantidade em ações; negativa = lançada)")
    opt_edit = st.data_editor(
        p.options_frame(), num_rows="dynamic", width="stretch", key=f"opt_{p.updated_at}",
        column_config={
            "underlying": st.column_config.TextColumn("Ativo-objeto"),
            "kind": st.column_config.SelectboxColumn("Tipo", options=["call", "put"]),
            "quantity": st.column_config.NumberColumn("Qtd (ações)", format="%.0f"),
            "strike": st.column_config.NumberColumn("Strike", format="%.2f"),
            "expiry": st.column_config.TextColumn("Vencimento"),
            "broker": st.column_config.TextColumn("Corretora"),
            "description": st.column_config.TextColumn("Descrição"),
        },
    )
    if st.button("Salvar alterações"):
        save_portfolio(Portfolio(frame_to_holdings(hold_edit), frame_to_options(opt_edit)))
        st.success("Carteira salva.")
        st.rerun()
    if not p.holdings and st.button("Carregar carteira de exemplo"):
        save_portfolio(Portfolio.load(SAMPLE_DIR / "carteira_exemplo.json"))
        st.rerun()
    if p.holdings:
        st.markdown("**Consolidado por ativo**")
        cons = holdings_to_frame(p.consolidated()).rename(columns={
            "ticker": "Ativo", "quantity": "Quantidade", "avg_price": "Preço médio", "lent": "Em aluguel", "market": "Mercado",
            "broker": "Corretoras", "asset_type": "Tipo", "name": "Nome"})
        st.dataframe(cons, hide_index=True, width="stretch")

# ======================= CALLS COBERTAS =======================
with tab_calls:
    p = load_portfolio()
    st.markdown(
        "Preço justo das calls sobre ações que você já tem (lote de 100). Ações já comprometidas com calls "
        "lançadas não entram. A **sugestão** é o strike/vencimento com maior rendimento anualizado dentro do "
        "limite de probabilidade de exercício."
    )
    if not p.holdings:
        st.info("Carteira vazia: leia seus extratos na aba **Carteira**.")
    c1, c2, c3 = st.columns(3)
    max_prob = c1.slider("Prob. máx. de exercício (sugestão)", 5, 60, 25, 5, key="cp") / 100
    otm = c2.slider("Strikes: % acima da cotação", 0, 50, (2, 20), key="co")
    above_pm = c3.checkbox("Só strikes ≥ preço médio", value=True, help="Evita ser exercido com prejuízo.")
    tickers = sorted({h.ticker for h in p.consolidated() if h.asset_type in ("acao", "etf", "bdr")})
    chosen = st.multiselect("Ativos", tickers, default=tickers)
    if st.button("Calcular calls", type="primary", disabled=not chosen):
        settings = base_settings(call_otm_min=otm[0] / 100, call_otm_max=otm[1] / 100, max_prob_exercise=max_prob,
                                 calls_above_avg_price=above_pm)
        hold = [h for h in p.consolidated() if h.ticker in chosen]
        with st.spinner("Calculando preços justos…"):
            df, warns = call_opportunities(hold, provider, settings, committed_shares=p.committed_shares())
        st.session_state["calls"] = (df, warns, settings)
    if "calls" in st.session_state:
        df, warns, settings = st.session_state["calls"]
        for w in warns:
            st.warning(w)
        if not df.empty:
            st.subheader("Sugestões")
            sug = suggestions(df, settings, "call")
            if sug.empty:
                st.info("Nenhum strike dentro dos limites. Aumente a probabilidade máxima ou o intervalo de strikes.")
            else:
                show(sug, ["ticker", "currency", "expiry", "dte", "strike", "spot", "fair_price", "premium_pct",
                           "annualized_yield", "prob_exercise", "upside_to_strike", "avg_price",
                           "return_if_exercised", "contracts", "premium_total", "market_bid", "market_ask"], "calls_sug")
            st.subheader("Todos os strikes")
            show(df, ["ticker", "currency", "expiry", "dte", "strike", "strike_source", "spot", "fair_price",
                      "fair_per_contract", "premium_pct", "annualized_yield", "prob_exercise", "delta", "theta_day",
                      "model_vol", "upside_to_strike", "return_if_exercised", "contracts", "free_shares",
                      "premium_total", "market_bid", "market_ask", "market_iv"], "calls_all")
            chart(df, "calls")

# ======================= PUTS =======================
with tab_puts:
    st.markdown(
        "Digite ações da B3 ou dos EUA que você quer comprar. Vendendo a put você recebe o prêmio enquanto espera; "
        "se for exercido, compra a ação pelo **strike − prêmio**. Opcional: `TICKER:preço alvo:quantidade` "
        "(ex.: `PLTR:40:200, PETR4:34`)."
    )
    c1, c2 = st.columns([3, 1])
    text = c1.text_input("Ativos que quero comprar", placeholder="PLTR, PETR4:34, VALE3:55:300", key="puts_text")
    max_prob_p = c2.slider("Prob. máx. de exercício (sugestão)", 5, 80, 35, 5, key="pp") / 100
    otm_p = st.slider("Strikes: % abaixo da cotação", 0, 50, (0, 20), key="po")
    if st.button("Calcular puts", type="primary", disabled=not text.strip()):
        settings = base_settings(put_otm_min=otm_p[0] / 100, put_otm_max=otm_p[1] / 100, max_prob_exercise=max_prob_p)
        frames, errors = [], []
        with st.spinner("Calculando preços justos…"):
            for item in parse_watchlist(text=text):
                try:
                    frames.append(put_opportunities(normalize_ticker(item.symbol), provider, settings,
                                                    target_price=item.target_price, shares=item.shares))
                except Exception as exc:
                    errors.append(f"{item.symbol}: {exc}")
        st.session_state["puts"] = (pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(), errors, settings)
    if "puts" in st.session_state:
        df, errors, settings = st.session_state["puts"]
        for e in errors:
            st.warning(e)
        if not df.empty:
            st.subheader("Sugestões")
            sug = suggestions(df, settings, "put")
            if sug.empty:
                st.info("Nenhum strike dentro dos limites. Ajuste a probabilidade máxima, o alvo ou o intervalo.")
            else:
                show(sug, ["ticker", "currency", "expiry", "dte", "strike", "spot", "fair_price", "premium_pct",
                           "annualized_yield", "prob_exercise", "effective_price", "discount_to_spot", "target_price",
                           "contracts", "cash_required", "premium_total", "market_bid", "market_ask"], "puts_sug")
            st.subheader("Todos os strikes")
            show(df, ["ticker", "currency", "expiry", "dte", "strike", "strike_source", "spot", "fair_price",
                      "fair_per_contract", "premium_pct", "annualized_yield", "prob_exercise", "delta", "model_vol",
                      "effective_price", "discount_to_spot", "meets_target", "cash_required", "market_bid",
                      "market_ask", "market_iv"], "puts_all")
            chart(df, "puts")

# ======================= CALCULADORA =======================
with tab_calc:
    st.markdown("Preço justo de uma opção específica. Informe um ativo para preencher cotação, volatilidade e dividendos.")
    c1, c2, c3 = st.columns(3)
    sym = normalize_ticker(c1.text_input("Ativo (opcional)"))
    defaults = {"spot": 100.0, "vol": 0.30, "q": 0.0, "r": 0.04}
    if sym:
        try:
            mkt = detect_market(sym)
            defaults["spot"] = round(provider.spot(sym), 2)
            defaults["vol"] = round(estimate_vol(provider.history(sym, 400), vol_method, vol_window) or 0.3, 4)
            defaults["q"] = round(provider.dividend_yield(sym), 4)
            defaults["r"] = round(annual_to_continuous(rate_override[mkt]) if mkt in rate_override
                                  else provider.risk_free_rate(mkt), 4)
            c1.caption(f"Mercado: {mkt} ({CURRENCY_SYMBOL[mkt]})")
        except Exception as exc:
            st.warning(f"Não foi possível carregar {sym}: {exc}")
    kind = c1.radio("Tipo", ["call", "put"], horizontal=True)
    S = c2.number_input("Cotação", 0.01, value=float(defaults["spot"]), key=f"S_{sym}")
    K = c2.number_input("Strike", 0.01, value=float(defaults["spot"]), key=f"K_{sym}")
    expiry = c2.date_input("Vencimento", dt.date.today() + dt.timedelta(days=30))
    sigma = c3.number_input("Volatilidade (a.a.)", 0.01, 5.0, float(defaults["vol"]), 0.01, key=f"v_{sym}")
    r = c3.number_input("Juro (contínuo, a.a.)", 0.0, 0.5, float(defaults["r"]), 0.0025, format="%.4f", key=f"r_{sym}")
    q = c3.number_input("Dividend yield", 0.0, 0.3, float(defaults["q"]), 0.001, format="%.4f", key=f"q_{sym}")
    market_px = c1.number_input("Preço de mercado (opcional)", 0.0, value=0.0, step=0.05)

    T = max((expiry - dt.date.today()).days + 1, 1) / 365.0
    fv = fair_value(S, K, T, r, sigma, kind, q, american)
    g = bs_greeks(S, K, T, r, sigma, kind, q)
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Preço justo", f"{fv:,.2f}")
    m2.metric("Por contrato (100)", f"{fv * 100:,.0f}")
    m3.metric("Delta", f"{g.delta:.2f}")
    m4.metric("Prob. de exercício", f"{g.prob_itm:.0%}")
    st.caption(f"Gamma {g.gamma:.4f} · Theta/dia {g.theta:.4f} · Vega (por ponto de vol) {g.vega:.4f}")
    if market_px > 0:
        iv = implied_vol(market_px, S, K, T, r, kind, q)
        st.write(f"Volatilidade implícita no preço de mercado: **{iv:.1%}**" if iv else "Volatilidade implícita: n/d")

st.divider()
st.caption(
    "Preço justo: árvore binomial (exercício americano) com volatilidade histórica, dividend yield dos últimos 12 meses "
    "e juro livre de risco (Selic do Banco Central para B3; T-bill de 13 semanas para EUA). Para a B3 não há cotação "
    "gratuita de opções no Yahoo: os strikes são teóricos (grade de valores redondos) nos vencimentos mensais "
    "(3ª sexta-feira); confira a série existente no home broker. Não é recomendação de investimento."
)
