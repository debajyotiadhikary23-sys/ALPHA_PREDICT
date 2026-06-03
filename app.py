# app.py
"""
Flask server for Advanced Stock Analysis Dashboard.

- Serves static/index.html
- /analyze      -> runs chart_model.run_analysis
- /predictions.csv -> downloads prediction log
- /search_tickers  -> LIVE ticker/company suggestions (Yahoo Finance)
"""

import os
import logging
from flask import Flask, request, jsonify, send_from_directory
import pandas as pd
import numpy as np
import requests  # <--- IMPORTANT for live suggestions

import chart_model

APP_ROOT = os.path.dirname(os.path.abspath(__file__))
STATIC_FOLDER = os.path.join(APP_ROOT, "static")
PRED_LOG = getattr(chart_model, "PRED_LOG", "predictions_log.csv")

LOG = logging.getLogger("stock_api")
LOG.setLevel(logging.INFO)
if not LOG.handlers:
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s %(message)s"))
    LOG.addHandler(ch)

app = Flask(__name__, static_folder=STATIC_FOLDER, static_url_path="/static")


def _normalize_request_json(req_json):
    if not isinstance(req_json, dict):
        return {}

    ticker = req_json.get("ticker") or req_json.get("symbol") or ""
    own_raw = req_json.get("ownStock", req_json.get("own", "No"))
    if isinstance(own_raw, str):
        own = own_raw.strip().lower() in ("yes", "true", "1")
    else:
        own = bool(own_raw)

    try:
        avg_price = float(req_json.get("avgPrice", req_json.get("avg_price", 0) or 0) or 0)
    except Exception:
        avg_price = 0.0
    try:
        quantity = int(req_json.get("quantity", 0) or 0)
    except Exception:
        quantity = 0

    chart_range = req_json.get("chart_range", req_json.get("period", "1y"))
    horizon = req_json.get("horizon", None)
    hold_plan = req_json.get("holdPlan", req_json.get("hold_plan", None))

    return {
        "ticker": ticker,
        "own": own,
        "avg_price": avg_price,
        "quantity": quantity,
        "chart_range": chart_range,
        "horizon": horizon,
        "hold_plan": hold_plan,
    }


def build_price_history(ticker: str, chart_range: str = "1y"):
    try:
        t = chart_model._normalize_ticker(ticker)
        yf_t = __import__("yfinance").Ticker(t)
        hist = yf_t.history(period=chart_range)
        if hist is None or hist.empty:
            return {"dates": [], "closes": [], "ma20": [], "ma50": []}
        use_col = "Adj Close" if "Adj Close" in hist.columns else "Close"
        closes = hist[use_col].astype(float).tolist()
        dates = [pd.to_datetime(idx).strftime("%Y-%m-%d") for idx in hist.index]
        ma20 = hist[use_col].rolling(window=20, min_periods=1).mean().astype(float).tolist()
        ma50 = hist[use_col].rolling(window=50, min_periods=1).mean().astype(float).tolist()
        return {"dates": dates, "closes": closes, "ma20": ma20, "ma50": ma50}
    except Exception:
        LOG.exception("build_price_history failed")
        return {"dates": [], "closes": [], "ma20": [], "ma50": []}


def build_debug_metrics_from_hist(ticker: str, chart_range: str = "1y"):
    try:
        t = chart_model._normalize_ticker(ticker)
        yf_t = __import__("yfinance").Ticker(t)
        hist = yf_t.history(period=chart_range)
        if hist is None or hist.empty:
            return {}
        last = hist.iloc[-1]
        sample = []
        tail = hist.tail(30)
        for idx, row in tail.iterrows():
            date_str = pd.to_datetime(idx).strftime("%Y-%m-%d")
            close_val = float(
                row["Adj Close"] if "Adj Close" in row and not np.isnan(row["Adj Close"]) else row.get("Close", np.nan)
            )
            sample.append([date_str, close_val])
        metrics = {
            "open": float(last.get("Open", np.nan)) if "Open" in last else None,
            "high": float(last.get("High", np.nan)) if "High" in last else None,
            "low": float(last.get("Low", np.nan)) if "Low" in last else None,
            "volume": int(last.get("Volume", 0)) if "Volume" in last else None,
            "sample_prices": sample,
        }
        return metrics
    except Exception:
        LOG.exception("build_debug_metrics_from_hist error")
        return {}


@app.route("/", methods=["GET"])
def index():
    index_path = os.path.join(STATIC_FOLDER, "index.html")
    if os.path.exists(index_path):
        return send_from_directory(STATIC_FOLDER, "index.html")
    return "<h3>Index not found (place your HTML at static/index.html)</h3>", 404


@app.route("/analyze", methods=["POST"])
def analyze():
    try:
        req_json = request.get_json(force=True)
    except Exception:
        return jsonify({"reasons": ["Invalid JSON in request"]}), 400

    params = _normalize_request_json(req_json)
    ticker = params.get("ticker", "")
    if not ticker:
        return jsonify({"reasons": ["Ticker empty"]}), 400

    try:
        res = chart_model.run_analysis(
            ticker=ticker,
            chart_range=params.get("chart_range", "1y"),
            own=params.get("own", False),
            avg_price=params.get("avg_price", 0.0),
            quantity=params.get("quantity", 0),
            hold_plan=params.get("hold_plan", None),
            horizon=params.get("horizon", None),
        )
    except Exception:
        LOG.exception("chart_model.run_analysis raised an exception")
        return jsonify({"reasons": ["Internal model error"]}), 500

    resp = {}
    resp["forecast_price"] = res.get("forecast_price") or res.get("current_price") or None
    resp["forecast_date"] = res.get("forecast_date") or "-"
    resp["score"] = int(res.get("score", 0) or 0)
    resp["decision"] = (res.get("decision") or "-").upper()

    gs = res.get("group_scores") or {}
    for k in ["Momentum", "Profitability", "Quality", "Sentiment", "Valuation"]:
        if k not in gs:
            gs[k] = 50
    resp["group_scores"] = {k: int(gs[k]) for k in gs}

    resp["reasons"] = res.get("reasons") or []
    resp["model_score"] = res.get("model_score")
    resp["model_confidence"] = res.get("model_confidence")
    resp["strategy_summary"] = res.get("strategy_summary")
    resp["recommendation_primary"] = res.get("recommendation_primary")
    resp["recommendation_options"] = res.get("recommendation_options") or []
    resp["n_training_rows"] = res.get("n_training_rows", None)

    resp["live_stats"] = res.get("live_stats") or {}
    resp["company"] = res.get("company") or {}
    resp["news"] = res.get("news") or []

    # price history
    if "price_history" in res and isinstance(res["price_history"], dict):
        ph = res["price_history"]
        resp["price_history"] = {
            "dates": ph.get("dates", []),
            "closes": ph.get("closes", []),
            "ma20": ph.get("ma20", []),
            "ma50": ph.get("ma50", []),
        }
    else:
        resp["price_history"] = build_price_history(ticker, chart_range=params.get("chart_range", "1y"))

    # current price
    try:
        if resp["price_history"]["closes"]:
            resp["current_price"] = float(resp["price_history"]["closes"][-1])
        else:
            resp["current_price"] = float(resp.get("forecast_price") or 0) or None
    except Exception:
        resp["current_price"] = None

    resp["debug_metrics"] = res.get("debug_metrics") or build_debug_metrics_from_hist(
        ticker, chart_range=params.get("chart_range", "1y")
    )

    LOG.info(
        f"/analyze {ticker} -> decision={resp['decision']} "
        f"score={resp['score']} rows={resp.get('n_training_rows')}"
    )
    return jsonify(resp)


@app.route("/predictions.csv", methods=["GET"])
def predictions_csv():
    if os.path.exists(PRED_LOG):
        return send_from_directory(APP_ROOT, PRED_LOG, as_attachment=True)
    return jsonify({"error": "No predictions log found"}), 404


@app.route("/static/<path:filename>")
def static_files(filename):
    return send_from_directory(STATIC_FOLDER, filename)


@app.route("/search_tickers", methods=["GET"])
def search_tickers():
    """
    LIVE search: user can type part of company name or symbol.
    We call Yahoo Finance search API and filter to Indian stocks (.NS / .BO).
    Returns [{symbol, displaySymbol, name, exchange}, ...]
    """
    q = request.args.get("q", "").strip()
    if not q:
        return jsonify({"results": []})

    try:
        url = "https://query1.finance.yahoo.com/v1/finance/search"
        params = {
            "q": q,              # user text: can be symbol or company short name
            "quotesCount": 15,
            "newsCount": 0,
            "region": "IN",
            "lang": "en-IN",
        }
        r = requests.get(url, params=params, timeout=5)
        data = r.json()
        out = []
        for item in data.get("quotes", []):
            sym = item.get("symbol")
            if not sym:
                continue
            # only Indian listed stocks
            if not (sym.endswith(".NS") or sym.endswith(".BO")):
                continue
            if item.get("quoteType") not in (None, "EQUITY", "COMMON_STOCK"):
                continue

            name = item.get("shortname") or item.get("longname") or sym
            exch = "NSE" if sym.endswith(".NS") else "BSE" if sym.endswith(".BO") else (item.get("exchange") or "")
            display_symbol = sym.replace(".NS", "").replace(".BO", "")

            out.append(
                {
                    "symbol": sym,                  # full yfinance symbol
                    "displaySymbol": display_symbol, # short symbol like RELIANCE
                    "name": name,                   # company name
                    "exchange": exch,
                }
            )
        return jsonify({"results": out})
    except Exception:
        LOG.exception("search_tickers failed")
        return jsonify({"results": []})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
