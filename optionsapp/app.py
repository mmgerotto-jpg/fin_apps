"""Flask app: run with `python -m optionsapp.app` then open http://127.0.0.1:5000"""
from __future__ import annotations

import argparse
import os

import numpy as np
from flask import Flask, jsonify, render_template, request
from flask.json.provider import DefaultJSONProvider

from . import analysis
from .market import get_provider
from .statements import parse_statement
from .store import Store

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class _JSON(DefaultJSONProvider):
    @staticmethod
    def default(o):
        if isinstance(o, np.generic):
            return o.item()
        return DefaultJSONProvider.default(o)


def create_app(provider_name="yahoo", data_path=None):
    app = Flask(__name__)
    app.json = _JSON(app)
    app.config["provider"] = get_provider(provider_name)
    store = Store(data_path or os.path.join(ROOT, "data", "portfolio.json"))

    def P():
        return app.config["provider"]

    @app.get("/")
    def index():
        return render_template("index.html")

    @app.get("/api/state")
    def state():
        return jsonify({**store.load(), "provider": P().name})

    @app.post("/api/provider")
    def set_provider():
        app.config["provider"] = get_provider(request.json.get("name", "yahoo"))
        return jsonify(ok=True, provider=P().name)

    @app.post("/api/statement")
    def upload_statement():
        files = request.files.getlist("file")
        if not files:
            return jsonify(error="no file"), 400
        mode = request.form.get("mode", "replace")
        parsed = []
        for f in files:
            try:
                parsed += [h.to_dict() for h in parse_statement(f.filename, f.read())]
            except Exception as e:
                return jsonify(error=f"{f.filename}: could not parse ({e})"), 400
        if not parsed:
            return jsonify(error="No stock positions found. Ensure the statement has Symbol and Quantity columns."), 400
        d = store.load()
        merged = {} if mode == "replace" else {h["symbol"]: h for h in d["holdings"]}
        for h in parsed:
            if h["symbol"] in merged and mode != "replace":
                merged[h["symbol"]]["quantity"] = h["quantity"]
            elif h["symbol"] in merged:
                merged[h["symbol"]]["quantity"] += h["quantity"]
            else:
                merged[h["symbol"]] = h
        d["holdings"] = sorted(merged.values(), key=lambda h: h["symbol"])
        store.save(d)
        return jsonify(holdings=d["holdings"])

    @app.put("/api/holdings")
    def put_holdings():
        d = store.load()
        d["holdings"] = [_clean(h, "quantity") for h in request.json if h.get("symbol")]
        store.save(d)
        return jsonify(holdings=d["holdings"])

    @app.put("/api/watchlist")
    def put_watchlist():
        d = store.load()
        d["watchlist"] = [_clean(w, "shares", "target_price") for w in request.json if w.get("symbol")]
        store.save(d)
        return jsonify(watchlist=d["watchlist"])

    @app.put("/api/settings")
    def put_settings():
        d = store.load()
        d["settings"] = request.json or {}
        store.save(d)
        return jsonify(settings=d["settings"])

    def _run(fn, items):
        st = _settings(store.load()["settings"])
        res = []
        for it in items:
            try:
                res.append(fn(P(), it, st))
            except Exception as e:
                res.append(dict(symbol=it.get("symbol"), error=str(e), rows=[]))
        return res

    @app.post("/api/analyze/calls")
    def calls():
        items = request.json.get("holdings") if request.is_json and request.json else None
        return jsonify(results=_run(analysis.analyze_calls, items or store.load()["holdings"]))

    @app.post("/api/analyze/puts")
    def puts():
        items = request.json.get("watchlist") if request.is_json and request.json else None
        return jsonify(results=_run(analysis.analyze_puts, items or store.load()["watchlist"]))

    @app.post("/api/calc")
    def calc():
        j = request.json
        try:
            res = analysis.quick_price(j.get("kind", "call"), float(j["spot"]), float(j["strike"]),
                                       float(j["dte"]), float(j.get("rate", 4.3)) / 100,
                                       float(j.get("div", 0)) / 100, float(j["vol"]) / 100,
                                       float(j["market"]) if j.get("market") else None)
        except (KeyError, ValueError) as e:
            return jsonify(error=f"invalid input: {e}"), 400
        return jsonify(res)

    return app


def _clean(d, *num_keys):
    out = {"symbol": str(d["symbol"]).strip().upper()}
    for k in num_keys:
        v = d.get(k)
        out[k] = float(v) if v not in (None, "") else None
    for k in ("cost_basis", "last_price"):
        if d.get(k) not in (None, ""):
            out[k] = float(d[k])
    return out


def _settings(s):
    """UI settings (percent units) -> analysis settings (fractions)."""
    def f(k, default):
        v = s.get(k)
        return float(v) if v not in (None, "") else default
    return dict(min_dte=int(f("min_dte", 20)), max_dte=int(f("max_dte", 60)),
                n_expiries=int(f("n_expiries", 2)), vol_source=s.get("vol_source") or "blend",
                vol_override=(f("vol_override", 0) / 100) or None,
                call_otm=(f("call_otm_min", 0) / 100, f("call_otm_max", 10) / 100),
                put_otm=(-f("put_otm_max", 15) / 100, -f("put_otm_min", 0) / 100),
                threshold=f("threshold", 10) / 100)


def main():
    ap = argparse.ArgumentParser(description="Options fair-value app")
    ap.add_argument("--demo", action="store_true", help="use synthetic offline market data")
    ap.add_argument("--port", type=int, default=5000)
    ap.add_argument("--data", help="path to portfolio JSON")
    a = ap.parse_args()
    app = create_app("demo" if a.demo else "yahoo", a.data)
    print(f"Open http://127.0.0.1:{a.port}")
    app.run(host="127.0.0.1", port=a.port, debug=False)


if __name__ == "__main__":
    main()
