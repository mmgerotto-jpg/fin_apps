"""Local web app: option fair value vs market price.

Run with:  streamlit run app.py
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import pandas as pd
import streamlit as st

from options_app.analysis import Settings, analyze_cash_secured_puts, analyze_covered_calls, best_per_symbol
from options_app.market_data import DemoProvider, YahooProvider
from options_app.pricing import bs_greeks, bs_price, fair_value, implied_vol
from options_app.statements import (
    WatchItem,
    frame_to_positions,
    merge_positions,
    parse_statement,
    positions_to_frame,
    short_calls_by_symbol,
    watchlist_from_frame,
)

SAMPLE_DIR = Path(__file__).resolve().parent / "sample_data"

st.set_page_config(page_title="Options Fair Value", layout="wide")
st.title("Options fair value vs market")


# ---------- sidebar: data source and model settings ----------
with st.sidebar:
    st.header("Market data")
    source = st.radio("Source", ["Yahoo Finance (live, ~15 min delay)", "Demo (offline synthetic)"], index=0)
    use_demo = source.startswith("Demo")

    st.header("Model")
    vol_method = st.selectbox(
        "Fair volatility estimate",
        ["blend", "historical", "ewma"],
        help="blend = average of the short-window and 1-year historical vol; "
        "historical = close-to-close stdev over the window; ewma = RiskMetrics (λ=0.94).",
    )
    vol_window = st.slider("Historical vol window (trading days)", 20, 252, 60, step=5)
    override_vol = st.checkbox("Override volatility for all stocks")
    vol_override = st.number_input("Volatility (annual, e.g. 0.30)", 0.01, 3.0, 0.30, 0.01) if override_vol else None
    override_rate = st.checkbox("Override risk-free rate (default: 13-week T-bill)")
    rate = st.number_input("Risk-free rate", 0.0, 0.2, 0.04, 0.0025, format="%.4f") if override_rate else None
    american = st.checkbox("American exercise (binomial tree)", value=True)

    st.header("Option filters")
    dte_range = st.slider("Days to expiry", 0, 365, (7, 60))
    max_expiries = st.slider("Max expiries per stock", 1, 12, 4)
    call_otm = st.slider("Call strikes: % above spot", 0, 50, (0, 15))
    put_otm = st.slider("Put strikes: % below spot", 0, 50, (0, 15))
    edge_threshold = st.slider("Flag when market differs from fair by more than (%)", 1, 50, 5) / 100
    min_bid = st.number_input("Ignore options with bid below", 0.0, 5.0, 0.05, 0.05)

settings = Settings(
    min_dte=dte_range[0],
    max_dte=dte_range[1],
    max_expiries=max_expiries,
    call_otm_min=call_otm[0] / 100,
    call_otm_max=call_otm[1] / 100,
    put_otm_min=put_otm[0] / 100,
    put_otm_max=put_otm[1] / 100,
    vol_method=vol_method,
    vol_window=vol_window,
    vol_override=vol_override,
    risk_free=rate,
    american=american,
    edge_threshold=edge_threshold,
    min_bid=min_bid,
)


@st.cache_resource
def get_provider(demo: bool):
    return DemoProvider() if demo else YahooProvider()


provider = get_provider(use_demo)

FMT = {
    "spot": "{:,.2f}", "strike": "{:,.2f}", "bid": "{:,.2f}", "ask": "{:,.2f}", "market_mid": "{:,.2f}",
    "fair_value": "{:,.2f}", "bs_european": "{:,.2f}", "edge": "{:+,.2f}", "edge_pct": "{:+.1%}",
    "market_iv": "{:.1%}", "model_vol": "{:.1%}", "delta": "{:.2f}", "theta_day": "{:.3f}", "prob_itm": "{:.1%}",
    "div_yield": "{:.2%}", "premium_total": "${:,.0f}", "premium_yield": "{:.2%}", "annualized_yield": "{:.1%}",
    "upside_cap_pct": "{:.1%}", "cost_basis": "{:,.2f}", "return_if_called": "{:.1%}", "cash_required": "${:,.0f}",
    "effective_buy_price": "{:,.2f}", "discount_to_spot": "{:.1%}", "target_price": "{:,.2f}",
}


def color_signal(val: str) -> str:
    if str(val).startswith("Rich"):
        return "background-color: rgba(46,160,67,0.25)"
    if str(val).startswith("Cheap"):
        return "background-color: rgba(218,54,51,0.25)"
    return ""


def show_table(df: pd.DataFrame, cols: list[str], key: str) -> None:
    view = df[[c for c in cols if c in df.columns]]
    styled = view.style.format({k: v for k, v in FMT.items() if k in view.columns}, na_rep="–")
    if "signal" in view.columns:
        styled = styled.map(color_signal, subset=["signal"])
    st.dataframe(styled, width="stretch", hide_index=True)
    st.download_button("Download CSV", df.to_csv(index=False).encode(), f"{key}.csv", "text/csv", key=f"dl_{key}")


def show_results(df: pd.DataFrame, errors: list[str], summary_cols: list[str], detail_cols: list[str], key: str):
    for e in errors:
        st.warning(e)
    if df.empty:
        st.info("No options matched the filters.")
        return
    c1, c2, c3 = st.columns(3)
    c1.metric("Options analysed", len(df))
    c2.metric("Rich (market > fair)", int(df["signal"].str.startswith("Rich").sum()))
    c3.metric("Cheap (market < fair)", int(df["signal"].str.startswith("Cheap").sum()))

    st.subheader("Best per stock")
    min_otm = st.slider(
        "Minimum probability of expiring worthless", 0, 95, 60, 5, key=f"pot_{key}",
        help="Restricts the summary to strikes the model expects to stay out of the money.",
    )
    show_table(best_per_symbol(df, min_otm / 100), summary_cols, f"{key}_best")

    st.subheader("All options")
    symbols = sorted(df["symbol"].unique())
    pick = st.multiselect("Stocks", symbols, default=symbols, key=f"sym_{key}")
    only_rich = st.checkbox("Only options priced above fair value", key=f"rich_{key}")
    d = df[df["symbol"].isin(pick)]
    if only_rich:
        d = d[d["edge"] > 0]
    show_table(d, detail_cols, f"{key}_all")

    if len(pick) == 1:
        sym = pick[0]
        sub = df[df["symbol"] == sym]
        exp = st.selectbox("Expiry for chart", sorted(sub["expiry"].unique()), key=f"exp_{key}")
        chart = sub[sub["expiry"] == exp].set_index("strike")[["market_mid", "fair_value"]]
        st.line_chart(chart, x_label="strike", y_label="price per share")


tab_calls, tab_puts, tab_calc = st.tabs(["Covered calls (my portfolio)", "Cash-secured puts (stocks to buy)", "Single option"])

# ---------- covered calls ----------
with tab_calls:
    st.markdown(
        "Upload brokerage position exports (CSV/XLSX; PDF best-effort). Holdings are merged across files; "
        "existing short calls reduce the contracts available to sell. You can edit the table before analysing."
    )
    files = st.file_uploader("Statements", type=["csv", "xlsx", "xls", "pdf"], accept_multiple_files=True)
    use_samples = st.checkbox("Use the sample statements in sample_data/", value=False)
    inputs = [(f.name, f.getvalue()) for f in files or []]
    if use_samples:
        inputs += [(p.name, p.read_bytes()) for p in sorted(SAMPLE_DIR.glob("*_positions.csv"))]
    positions, options = [], []
    for name, data in inputs:
        try:
            res = parse_statement(data, name)
            positions += res.positions
            options += res.options
        except Exception as exc:
            st.error(f"{name}: {exc}")
    if not inputs:
        st.caption("No statement yet: upload one, tick the sample box, or type holdings into the table.")

    base = positions_to_frame(merge_positions(positions)) if positions else positions_to_frame([])
    edited = st.data_editor(
        base[["symbol", "quantity", "cost_basis"]],
        num_rows="dynamic",
        width="stretch",
        key=f"holdings_{'_'.join(n for n, _ in inputs)}_{sum(len(d) for _, d in inputs)}",
        column_config={
            "symbol": st.column_config.TextColumn("Symbol"),
            "quantity": st.column_config.NumberColumn("Shares", format="%.0f"),
            "cost_basis": st.column_config.NumberColumn("Cost / share", format="%.2f"),
        },
    )
    short_calls = short_calls_by_symbol(options)
    if options:
        with st.expander(f"Existing option positions found ({len(options)})"):
            st.dataframe(
                pd.DataFrame([{"underlying": o.underlying, "type": o.kind, "qty": o.quantity, "description": o.description} for o in options]),
                hide_index=True, width="stretch",
            )

    if st.button("Analyse covered calls", type="primary", disabled=edited.dropna(subset=["symbol"]).empty):
        with st.spinner("Pulling option chains and pricing…"):
            st.session_state["calls"] = analyze_covered_calls(
                frame_to_positions(edited), provider, settings, short_calls=short_calls
            )
    if "calls" in st.session_state:
        df, errors = st.session_state["calls"]
        show_results(
            df, errors,
            ["symbol", "expiry", "dte", "strike", "spot", "market_mid", "fair_value", "edge", "edge_pct", "signal",
             "market_iv", "model_vol", "prob_itm", "annualized_yield", "contracts_available", "premium_total",
             "return_if_called"],
            ["symbol", "expiry", "dte", "strike", "spot", "bid", "ask", "market_mid", "fair_value", "bs_european",
             "edge", "edge_pct", "signal", "market_iv", "model_vol", "delta", "theta_day", "prob_itm",
             "annualized_yield", "upside_cap_pct", "return_if_called", "contracts_available", "premium_total",
             "volume", "open_interest"],
            "calls",
        )

# ---------- cash-secured puts ----------
with tab_puts:
    st.markdown(
        "Stocks you plan to buy. Selling a put pays premium while you wait; if assigned, your effective purchase "
        "price is strike − premium. *Target price* is the most you'd pay per share."
    )
    wl_file = st.file_uploader("Watchlist CSV (symbol, target_price, shares)", type=["csv"], key="wl")
    wl_df = pd.DataFrame({"symbol": ["GOOGL", "JPM"], "target_price": [None, None], "shares": [100, 100]})
    if wl_file is not None:
        try:
            items = watchlist_from_frame(pd.read_csv(wl_file))
            wl_df = pd.DataFrame([{"symbol": i.symbol, "target_price": i.target_price, "shares": i.shares} for i in items])
        except Exception as exc:
            st.error(str(exc))
    wl_edit = st.data_editor(
        wl_df, num_rows="dynamic", width="stretch", key=f"wl_{wl_file.name if wl_file else 'manual'}",
        column_config={
            "symbol": st.column_config.TextColumn("Symbol"),
            "target_price": st.column_config.NumberColumn("Target price", format="%.2f"),
            "shares": st.column_config.NumberColumn("Shares to buy", format="%.0f"),
        },
    )
    if st.button("Analyse puts", type="primary"):
        items = [
            WatchItem(
                str(r.symbol).strip().upper(),
                float(r.target_price) if pd.notna(r.target_price) else None,
                float(r.shares) if pd.notna(r.shares) and r.shares else 100,
            )
            for r in wl_edit.itertuples()
            if pd.notna(r.symbol) and str(r.symbol).strip()
        ]
        with st.spinner("Pulling option chains and pricing…"):
            st.session_state["puts"] = analyze_cash_secured_puts(items, provider, settings)
    if "puts" in st.session_state:
        df, errors = st.session_state["puts"]
        show_results(
            df, errors,
            ["symbol", "expiry", "dte", "strike", "spot", "market_mid", "fair_value", "edge", "edge_pct", "signal",
             "market_iv", "model_vol", "prob_itm", "effective_buy_price", "discount_to_spot", "target_price",
             "meets_target", "annualized_yield", "cash_required"],
            ["symbol", "expiry", "dte", "strike", "spot", "bid", "ask", "market_mid", "fair_value", "bs_european",
             "edge", "edge_pct", "signal", "market_iv", "model_vol", "delta", "theta_day", "prob_itm",
             "effective_buy_price", "discount_to_spot", "target_price", "meets_target", "annualized_yield",
             "contracts", "cash_required", "premium_total", "volume", "open_interest"],
            "puts",
        )

# ---------- single option calculator ----------
with tab_calc:
    st.markdown("Price any option by hand, optionally pulling spot / vol / dividend from market data.")
    c1, c2, c3 = st.columns(3)
    sym = c1.text_input("Symbol (optional, to auto-fill)").strip().upper()
    defaults = {"spot": 100.0, "vol": 0.30, "q": 0.0}
    if sym:
        try:
            from options_app.volatility import estimate_vol

            defaults["spot"] = round(provider.spot(sym), 2)
            defaults["vol"] = round(estimate_vol(provider.history(sym, 400), vol_method, vol_window) or 0.3, 4)
            defaults["q"] = round(provider.dividend_yield(sym), 4)
        except Exception as exc:
            st.warning(f"Could not load {sym}: {exc}")
    kind = c1.radio("Type", ["call", "put"], horizontal=True)
    S = c2.number_input("Spot", 0.01, value=float(defaults["spot"]), key=f"S_{sym}")
    K = c2.number_input("Strike", 0.01, value=float(defaults["spot"]))
    expiry = c2.date_input("Expiry", dt.date.today() + dt.timedelta(days=30))
    sigma = c3.number_input("Volatility", 0.01, 5.0, float(defaults["vol"]), 0.01, key=f"v_{sym}")
    r = c3.number_input("Risk-free rate", 0.0, 0.2, rate if rate is not None else 0.04, 0.0025, format="%.4f")
    q = c3.number_input("Dividend yield", 0.0, 0.2, float(defaults["q"]), 0.001, format="%.4f", key=f"q_{sym}")
    market = c1.number_input("Market price (optional, 0 = none)", 0.0, value=0.0, step=0.05)

    T = max((expiry - dt.date.today()).days + 1, 1) / 365.0
    fv = fair_value(S, K, T, r, sigma, kind, q, american)
    euro = bs_price(S, K, T, r, sigma, kind, q)
    g = bs_greeks(S, K, T, r, sigma, kind, q)
    m1, m2, m3, m4 = st.columns(4)
    m1.metric(f"Fair value ({'American' if american else 'European'})", f"{fv:,.3f}")
    m2.metric("Black-Scholes European", f"{euro:,.3f}")
    m3.metric("Delta", f"{g.delta:.3f}")
    m4.metric("Prob. in the money", f"{g.prob_itm:.1%}")
    st.caption(f"Gamma {g.gamma:.4f} · Theta/day {g.theta:.4f} · Vega (per vol pt) {g.vega:.4f} · Rho {g.rho:.4f}")
    if market > 0:
        iv = implied_vol(market, S, K, T, r, kind, q)
        edge = market - fv
        st.metric("Market − fair", f"{edge:+.3f}", f"{edge / fv:+.1%}" if fv > 0 else None)
        st.write(f"Implied volatility of market price: **{iv:.2%}**" if iv else "Implied volatility: n/a")

st.divider()
st.caption(
    "Fair value uses a CRR binomial tree (American exercise) with a continuous dividend yield from trailing-12-month "
    "dividends, the 13-week T-bill as risk-free rate, and volatility estimated from price history. 'Edge' = market mid − "
    "fair value: positive means the market pays more than the model value, which favours selling. Model output, not "
    "investment advice; Yahoo quotes can be delayed or stale outside market hours."
)
