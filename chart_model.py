"""
chart_model.py  — extended with additional factors and 'owners: sell-only' policy

Behavior:
- If user indicates own=True:
    - If model/logic results in SELL -> return SELL and recommendation_primary "Consider selling..."
    - Otherwise -> return NEUTRAL and recommendation_primary "Don’t sell now – the signal is largely neutral."
    - recommendation_options will be an empty list for owners.

Other behavior (models, advanced factors, logging) unchanged.
"""

from datetime import timedelta, datetime
import numpy as np
import pandas as pd
import yfinance as yf
import logging
import os
import csv

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import r2_score, mean_squared_error

# optional shap (explainability)
try:
    import shap
    HAS_SHAP = True
except Exception:
    HAS_SHAP = False

# optional TextBlob for simple sentiment
try:
    from textblob import TextBlob
    HAS_TEXTBLOB = True
except Exception:
    HAS_TEXTBLOB = False

LOG = logging.getLogger("chart_model")
LOG.setLevel(logging.INFO)
if not LOG.handlers:
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s %(message)s"))
    LOG.addHandler(ch)

PRED_LOG = "predictions_log.csv"

# Tunable thresholds
BUY_SCORE_THRESHOLD = 65
SELL_SCORE_THRESHOLD = 40
MIN_TRAIN_ROWS = 120
DEFAULT_CIRCUIT_BAND = 0.05  # 5%

# Basic holiday list (update as needed)
MARKET_HOLIDAYS = {
    "2025-01-26",
    "2025-03-14",
    "2025-04-18",
    "2025-08-15",
    "2025-10-02",
    "2025-11-04",
}


def _next_trading_day(last_dt):
    d = pd.to_datetime(last_dt)
    while True:
        d = d + pd.Timedelta(days=1)
        if d.weekday() >= 5:
            continue
        if d.strftime("%Y-%m-%d") in MARKET_HOLIDAYS:
            continue
        return d


def _format_forecast_date_from_index(last_index):
    next_d = _next_trading_day(last_index)
    return next_d.strftime("%d-%b-%Y (%A)")


def _normalize_ticker(ticker: str) -> str:
    t = str(ticker).strip().upper()
    if not t:
        return t
    if "." in t:
        return t
    return t + ".NS"


def fetch_price_history(stock: str, chart_range: str = "2y"):
    try:
        if not stock or str(stock).strip() == "":
            return []
        t = _normalize_ticker(stock)
        ticker = yf.Ticker(t)
        hist = ticker.history(period=chart_range)
        if hist is None or hist.empty:
            return []
        use_col = "Adj Close" if "Adj Close" in hist.columns else "Close"
        out = []
        for idx, row in hist.iterrows():
            try:
                date_str = pd.to_datetime(idx).strftime("%Y-%m-%d")
                close = float(row.get(use_col, np.nan))
                if np.isnan(close):
                    close = float(row.get("Close", np.nan))
                if np.isnan(close):
                    continue
                out.append({"x": date_str, "c": close})
            except Exception:
                continue
        return out
    except Exception:
        LOG.exception("fetch_price_history error")
        return []


def _build_features(df: pd.DataFrame, use_adj=True):
    df = df.copy()
    price_col = "Adj Close" if use_adj and "Adj Close" in df.columns else "Close"
    prices = df[price_col].astype(float)

    df["prev_close"] = prices.shift(1).fillna(method="bfill")
    df["pct_1"] = prices.pct_change(1).fillna(0.0)
    df["ma5"] = prices.rolling(5, min_periods=1).mean()
    df["ma10"] = prices.rolling(10, min_periods=1).mean()
    df["ma20"] = prices.rolling(20, min_periods=1).mean()
    df["std5"] = prices.rolling(5, min_periods=1).std().fillna(0.0)
    df["std10"] = prices.rolling(10, min_periods=1).std().fillna(0.0)
    df["mom3"] = prices.pct_change(3).fillna(0.0)
    df["volume"] = df["Volume"] if "Volume" in df.columns else 0.0

    next_close = prices.shift(-1)
    with np.errstate(divide="ignore", invalid="ignore"):
        df["target_logret"] = np.log(next_close / prices)
    df = df.dropna(subset=["target_logret"])

    feature_cols = [
        "prev_close",
        "pct_1",
        "ma5",
        "ma10",
        "ma20",
        "std5",
        "std10",
        "mom3",
        "volume",
    ]
    X = df[feature_cols].fillna(0.0)
    y = df["target_logret"].astype(float)
    return X, y, df.index


def _train_ensemble(X: pd.DataFrame, y: pd.Series):
    if len(X) < 3:
        model_gb = GradientBoostingRegressor(n_estimators=80, random_state=42)
        model_rf = RandomForestRegressor(n_estimators=80, random_state=42, n_jobs=-1)
        model_gb.fit(X, y)
        model_rf.fit(X, y)
        return (model_gb, model_rf), {"r2": 0.0, "rmse": None}

    n_splits = min(5, max(1, len(X) // 50))
    tscv = TimeSeriesSplit(n_splits=n_splits)
    r2s = []
    rmses = []
    for train_idx, val_idx in tscv.split(X):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        model_gb = GradientBoostingRegressor(n_estimators=200, random_state=42)
        model_rf = RandomForestRegressor(n_estimators=250, random_state=42, n_jobs=-1)
        model_gb.fit(X_train, y_train)
        model_rf.fit(X_train, y_train)
        preds = 0.5 * (model_gb.predict(X_val) + model_rf.predict(X_val))
        try:
            r2s.append(r2_score(y_val, preds))
            rmses.append(np.sqrt(mean_squared_error(y_val, preds)))
        except Exception:
            r2s.append(0.0)
            rmses.append(np.nan)

    model_gb_final = GradientBoostingRegressor(n_estimators=250, random_state=42)
    model_rf_final = RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1)
    model_gb_final.fit(X, y)
    model_rf_final.fit(X, y)

    diag = {
        "r2": float(np.mean(r2s)) if r2s else 0.0,
        "rmse": float(np.mean(rmses)) if rmses and not np.isnan(np.mean(rmses)) else None,
    }
    return (model_gb_final, model_rf_final), diag


def _ensemble_predict(models, X_row):
    model_gb, model_rf = models
    p1 = model_gb.predict(X_row)[0]
    p2 = model_rf.predict(X_row)[0]
    return float(0.5 * (p1 + p2))


def _append_prediction_log(row: dict):
    header = [
        "timestamp",
        "ticker",
        "last_close",
        "forecast_price",
        "forecast_date",
        "expected_pct",
        "decision",
        "score",
        "model_r2",
    ]
    exists = os.path.exists(PRED_LOG)
    try:
        with open(PRED_LOG, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=header)
            if not exists:
                writer.writeheader()
            writer.writerow(row)
    except Exception:
        LOG.exception("Failed to append prediction log")


# -------------------------
# Additional indicator helpers
# -------------------------
def compute_RSI(series: pd.Series, period: int = 14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()
    rs = avg_gain / (avg_loss.replace(0, np.nan))
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50.0)
    return rsi


def compute_MACD(series: pd.Series, short: int = 12, long: int = 26, signal: int = 9):
    ema_short = series.ewm(span=short, adjust=False).mean()
    ema_long = series.ewm(span=long, adjust=False).mean()
    macd = ema_short - ema_long
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    hist = macd - signal_line
    return macd, signal_line, hist


def compute_ATR(df: pd.DataFrame, period: int = 14):
    high = df["High"]
    low = df["Low"]
    close = df["Close"]
    tr1 = (high - low).abs()
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=period, min_periods=1).mean()
    return atr


def compute_volume_trend(df: pd.DataFrame, lookback: int = 20):
    vol = df["Volume"].astype(float)
    avg_vol = vol.rolling(window=lookback, min_periods=1).mean()
    ratio = vol / (avg_vol.replace(0, np.nan))
    ratio = ratio.fillna(1.0)
    return ratio


def compute_news_sentiment(news_items):
    if not HAS_TEXTBLOB or not news_items:
        return 50.0
    vals = []
    for it in news_items:
        t = (it.get("title") or "").strip()
        if not t:
            continue
        try:
            p = TextBlob(t).sentiment.polarity
            vals.append(p)
        except Exception:
            continue
    if not vals:
        return 50.0
    avg = float(np.mean(vals))
    mapped = np.clip((avg + 1) / 2.0 * 100.0, 0, 100)
    return mapped


# -------------------------
# Advanced factor blending
# -------------------------
def compute_advanced_factor_scores(df: pd.DataFrame, news_items=None):
    out = {
        "rsi_score": 50,
        "macd_score": 50,
        "atr_score": 50,
        "volume_score": 50,
        "news_sentiment": 50,
    }
    try:
        if df is None or df.empty:
            return out

        price_col = "Adj Close" if "Adj Close" in df.columns else "Close"
        series = df[price_col].astype(float).dropna()
        if series.empty:
            return out

        # RSI
        try:
            rsi = compute_RSI(series, period=14)
            last_rsi = float(rsi.iloc[-1])
            if last_rsi >= 70:
                rsi_score = 80
            elif last_rsi >= 60:
                rsi_score = 70
            elif last_rsi >= 40:
                rsi_score = 50
            elif last_rsi >= 30:
                rsi_score = 35
            else:
                rsi_score = 20
            out["rsi_score"] = int(rsi_score)
        except Exception:
            out["rsi_score"] = 50

        # MACD
        try:
            macd, signal, hist = compute_MACD(series, short=12, long=26, signal=9)
            hist_val = float(hist.iloc[-1])
            if hist_val > 0.005 * series.iloc[-1]:
                macd_score = 80
            elif hist_val > 0:
                macd_score = 70
            elif hist_val < -0.005 * series.iloc[-1]:
                macd_score = 25
            else:
                macd_score = 40
            out["macd_score"] = int(macd_score)
        except Exception:
            out["macd_score"] = 50

        # ATR
        try:
            if {"High", "Low", "Close"}.issubset(df.columns):
                atr = compute_ATR(df, period=14)
                last_atr = float(atr.iloc[-1])
                rel_atr = last_atr / (series.iloc[-1] if series.iloc[-1] != 0 else 1.0)
                if rel_atr <= 0.008:
                    atr_score = 85
                elif rel_atr <= 0.015:
                    atr_score = 70
                elif rel_atr <= 0.03:
                    atr_score = 55
                else:
                    atr_score = 35
                out["atr_score"] = int(atr_score)
            else:
                out["atr_score"] = 50
        except Exception:
            out["atr_score"] = 50

        # Volume trend
        try:
            if "Volume" in df.columns:
                vol_ratio = compute_volume_trend(df, lookback=20)
                last_ratio = float(vol_ratio.iloc[-1])
                if last_ratio >= 1.5:
                    vol_score = 80
                elif last_ratio >= 1.1:
                    vol_score = 70
                elif last_ratio >= 0.7:
                    vol_score = 50
                else:
                    vol_score = 30
                out["volume_score"] = int(vol_score)
            else:
                out["volume_score"] = 50
        except Exception:
            out["volume_score"] = 50

        # News sentiment
        try:
            out["news_sentiment"] = int(compute_news_sentiment(news_items or []))
        except Exception:
            out["news_sentiment"] = 50

    except Exception:
        LOG.exception("compute_advanced_factor_scores failed")

    return out


# -------------------------
# Original factor scoring (kept)
# -------------------------
def compute_factor_scores_from_df(df: pd.DataFrame):
    momentum = 50
    profitability = 50
    quality = 50
    sentiment = 50
    valuation = 50

    if df is None or df.empty:
        return {
            "Momentum": momentum,
            "Profitability": profitability,
            "Quality": quality,
            "Sentiment": sentiment,
            "Valuation": valuation,
        }

    price_col = "Adj Close" if "Adj Close" in df.columns else "Close"
    prices = df[price_col].astype(float).dropna()
    n = len(prices)

    if n >= 11:
        last = prices.iloc[-1]
        prev_10 = prices.iloc[-11]
        if prev_10 != 0:
            momentum_change = (last - prev_10) / prev_10 * 100.0
            if momentum_change >= 10:
                momentum = 90
            elif momentum_change >= 5:
                momentum = 75
            elif momentum_change >= 2:
                momentum = 65
            elif momentum_change >= -1:
                momentum = 50
            elif momentum_change >= -5:
                momentum = 35
            else:
                momentum = 20

    if n >= 21:
        base = prices.iloc[-21]
        if base != 0:
            slope = (prices.iloc[-1] - base) / base * 100.0
            if slope >= 10:
                profitability = 85
            elif slope >= 5:
                profitability = 70
            elif slope >= 1:
                profitability = 55
            else:
                profitability = 40

    if n >= 15:
        vol10 = prices.pct_change().rolling(10).std().iloc[-1]
        if not np.isnan(vol10):
            if vol10 <= 0.008:
                quality = 85
            elif vol10 <= 0.015:
                quality = 70
            elif vol10 <= 0.03:
                quality = 55
            else:
                quality = 35

    if n >= 50:
        ma20 = prices.rolling(20).mean().iloc[-1]
        ma50 = prices.rolling(50).mean().iloc[-1]
        if ma50 != 0:
            if ma20 > ma50 * 1.01:
                sentiment = 80
            elif ma20 > ma50:
                sentiment = 65
            elif ma20 < ma50 * 0.99:
                sentiment = 30
            else:
                sentiment = 50
    elif n >= 20:
        ma10 = prices.rolling(10).mean().iloc[-1]
        ma20 = prices.rolling(20).mean().iloc[-1]
        sentiment = 65 if ma10 > ma20 else 45

    if n >= 20:
        ma20v = prices.rolling(20).mean().iloc[-1]
        if ma20v != 0:
            deviation = (prices.iloc[-1] - ma20v) / ma20v * 100.0
            if abs(deviation) <= 2:
                valuation = 80
            elif abs(deviation) <= 6:
                valuation = 65
            elif abs(deviation) <= 12:
                valuation = 50
            else:
                valuation = 35
            if deviation < -8:
                valuation = min(90, valuation + 10)

    scores = {
        "Momentum": int(np.clip(momentum, 0, 100)),
        "Profitability": int(np.clip(profitability, 0, 100)),
        "Quality": int(np.clip(quality, 0, 100)),
        "Sentiment": int(np.clip(sentiment, 0, 100)),
        "Valuation": int(np.clip(valuation, 0, 100)),
    }
    return scores


# -------------------------
# Main run_analysis (integrates advanced factors)
# -------------------------
def run_analysis(
    ticker: str,
    chart_range: str = "2y",
    own: bool = False,
    avg_price: float = 0.0,
    quantity: int = 0,
    hold_plan: str = None,
    horizon: int = None,
):
    result = {
        "forecast_price": None,
        "forecast_date": "-",
        "score": 0,
        "decision": "-",
        "group_scores": {},
        "reasons": [],
        "model_score": None,
        "n_training_rows": 0,
        "shap": None,
        "live_stats": {},
        "company": {},
        "news": [],
        "recommendation_primary": "",
        "recommendation_options": [],
        "model_confidence": None,
        "strategy_summary": "",
        "debug_metrics": {},
    }

    try:
        if not ticker or str(ticker).strip() == "":
            result["reasons"].append("No stock selected.")
            return result

        t = _normalize_ticker(ticker)
        yf_t = yf.Ticker(t)

        # ---- LIVE INFO / COMPANY ----
        info = {}
        company_name = None
        symbol = t
        exchange = None
        currency = None
        mcap = None
        f52h = None
        f52l = None

        try:
            info = yf_t.info or {}
            company_name = info.get("longName") or info.get("shortName")
            symbol = info.get("symbol") or t
            exchange = info.get("exchange") or info.get("market")
            currency = info.get("currency") or "INR"
            f52h = info.get("fiftyTwoWeekHigh")
            f52l = info.get("fiftyTwoWeekLow")
            mcap = info.get("marketCap")
        except Exception:
            info = {}

        # ---- HISTORICAL DATA (for model + charts) ----
        hist = yf_t.history(period="5y")
        if hist is None or hist.empty:
            hist = yf_t.history(period=chart_range)

        if hist is None or hist.empty:
            result["reasons"].append(f"No usable historical data found for {ticker}.")
            return result

        use_adj = "Adj Close" in hist.columns
        price_col = "Adj Close" if use_adj else "Close"
        last_close_hist = float(hist[price_col].iloc[-1])

        # ---- TODAY STATS ----
        today_open = today_high = today_low = today_close = today_volume = None
        try:
            daily = yf_t.history(period="5d", interval="1d")
            if daily is not None and not daily.empty:
                row = daily.iloc[-1]
                today_open = float(row.get("Open"))
                today_high = float(row.get("High"))
                today_low = float(row.get("Low"))
                today_close = float(row.get("Close"))
                today_volume = float(row.get("Volume"))
        except Exception:
            pass

        last_close = float(today_close if today_close is not None else last_close_hist)

        # ---- price_history for UI ----
        try:
            closes = hist[price_col].astype(float).tolist()
            dates = [pd.to_datetime(idx).strftime("%Y-%m-%d") for idx in hist.index]
            ma20 = (
                hist[price_col]
                .rolling(window=20, min_periods=1)
                .mean()
                .astype(float)
                .tolist()
            )
            ma50 = (
                hist[price_col]
                .rolling(window=50, min_periods=1)
                .mean()
                .astype(float)
                .tolist()
            )
            opens = hist["Open"].astype(float).tolist()
            highs = hist["High"].astype(float).tolist()
            lows = hist["Low"].astype(float).tolist()

            result["price_history"] = {
                    "dates": dates,
                    "closes": closes,
                    "opens": opens,
                    "highs": highs,
                    "lows": lows,
                    "ma20": ma20,
                    "ma50": ma50
                }
        except Exception:
            result["price_history"] = {"dates": [], "closes": [], "ma20": [], "ma50": []}

        # ---- ML features ----
        X, y, idx = _build_features(hist, use_adj)
        n_rows = len(X)
        result["n_training_rows"] = n_rows

        forecast_price = None
        forecast_date = None
        models = None
        model_diag = None
        insufficient_data = False

        if n_rows >= MIN_TRAIN_ROWS:
            try:
                models, model_diag = _train_ensemble(X, y)
                result["model_score"] = model_diag.get("r2", None)
                last_feat = X.iloc[-1:].copy()
                pred_logret = _ensemble_predict(models, last_feat)
                forecast_price = float(last_close * np.exp(pred_logret))
                forecast_date = _format_forecast_date_from_index(idx[-1])

                if HAS_SHAP:
                    try:
                        explainer = shap.Explainer(models[1])
                        shap_vals = explainer(last_feat)
                        sv = list(zip(last_feat.columns.tolist(), shap_vals.values[0].tolist()))
                        result["shap"] = sv
                    except Exception:
                        LOG.exception("SHAP explain failed")
                        result["shap"] = None
            except Exception:
                LOG.exception("Model training/prediction failed")
                result["reasons"].append("The model could not train properly, so we used the latest price as a simple fallback.")
                forecast_price = last_close
                forecast_date = _format_forecast_date_from_index(idx[-1])
        else:
            insufficient_data = True
            forecast_price = last_close
            forecast_date = _format_forecast_date_from_index(hist.index[-1])
            result["reasons"].append(f"Limited history ({n_rows} usable days). Using the latest price as a simple forecast.")

        if forecast_price is None:
            forecast_price = last_close
            forecast_date = _format_forecast_date_from_index(hist.index[-1])
            result["reasons"].append("No forecast could be generated, so we used the latest price as a fallback.")

        expected_pct = (forecast_price - last_close) / last_close if last_close != 0 else 0.0

        # ---- overall score ----
        

        # 🔥 NEW BALANCED SCORE (FIXED)
        score_move = 50 + (expected_pct * 300)

        # soft clamp (prevents extreme 0 or 100)
        if score_move > 100:
            score_move = 100
        elif score_move < 10:
            score_move = 10

        # model contribution
        score_model = 50.0
        if isinstance(result.get("model_score"), float) and result["model_score"] is not None:
            score_model = np.clip(25 + result["model_score"] * 25, 0, 50)

        # final combined score
        final_score = int(np.clip(0.65 * score_move + 0.35 * score_model, 0, 100))


        # ---- factor scores (original) ----
        group_scores = compute_factor_scores_from_df(hist)


        # ---- advanced factors (new) ----
        try:
            raw_news = getattr(yf_t, "news", []) or []
            adv = compute_advanced_factor_scores(hist, raw_news)
        except Exception:
            adv = compute_advanced_factor_scores(hist, [])


        # ---- Blend advanced scores with original group_scores ----
        try:
            m_orig = group_scores.get("Momentum", 50)
            rsi_s = adv.get("rsi_score", 50)
            macd_s = adv.get("macd_score", 50)
            momentum_new = int(np.clip(0.6 * m_orig + 0.25 * rsi_s + 0.15 * macd_s, 0, 100))
            group_scores["Momentum"] = momentum_new
        except Exception:
            pass


        try:
            q_orig = group_scores.get("Quality", 50)
            atr_s = adv.get("atr_score", 50)
            vol_s = adv.get("volume_score", 50)
            quality_new = int(np.clip(0.6 * q_orig + 0.25 * atr_s + 0.15 * vol_s, 0, 100))
            group_scores["Quality"] = quality_new
        except Exception:
            pass


        try:
            s_orig = group_scores.get("Sentiment", 50)
            news_s = adv.get("news_sentiment", 50)
            macd_s = adv.get("macd_score", 50)
            sentiment_new = int(np.clip(0.6 * s_orig + 0.3 * news_s + 0.1 * macd_s, 0, 100))
            group_scores["Sentiment"] = sentiment_new
        except Exception:
            pass


        group_scores["Valuation"] = int(np.clip(group_scores.get("Valuation", 50), 0, 100))


        # model influence on factors
        if result.get("model_score") is not None:
            ms = result["model_score"] or 0.0
            add = int(np.clip(ms * 50, -25, 50))
            group_scores["Quality"] = int(np.clip(group_scores.get("Quality", 50) + add // 2, 0, 100))
            group_scores["Profitability"] = int(np.clip(group_scores.get("Profitability", 50) + add // 3, 0, 100))

        # ==========================
        # DECISION LOGIC
        # ==========================
        mom = group_scores.get("Momentum", 50)
        sent = group_scores.get("Sentiment", 50)
        short_term_score = (mom + sent) / 2.0

        decision = "HOLD"

        if not insufficient_data and result.get("model_score") is not None:
            if final_score >= BUY_SCORE_THRESHOLD and expected_pct > 0:
                decision = "BUY"
            elif final_score <= SELL_SCORE_THRESHOLD and expected_pct < 0:
                decision = "SELL"
            else:
                decision = "HOLD"
        else:
            if short_term_score >= 65:
                decision = "BUY"
            elif short_term_score <= 40:
                decision = "SELL"
            else:
                decision = "HOLD"

        # --------------------------
        # OWNER POLICY: SELL-ONLY
        # --------------------------
        # If owner (own=True):
        #   - If computed decision == SELL -> keep SELL
        #   - Otherwise -> force NEUTRAL (do not show BUY/HOLD)
        if own:
            if decision == "SELL":
                # keep SELL for owners (advise selling)
                decision = "SELL"
            else:
                # for owners, show a neutral "don't sell now" message
                decision = "NEUTRAL"

        # ---- human-friendly reasons ----
        reasons = list(result.get("reasons", []))
        if result.get("model_score") is not None and not insufficient_data:
            reasons.append(f"The model expects around {expected_pct*100:.2f}% price change in the near term.")
            reasons.append("The model fits past price movements reasonably well.")
        else:
            reasons.append("The signal is mainly based on recent price trend and volatility, not a strong model pattern.")

        if mom >= 75:
            reasons.append("Recent price trend has been strong and positive.")
        elif mom <= 35:
            reasons.append("Recent price trend has been weak or negative.")

        if sent >= 70:
            reasons.append("Short-term trend is supportive, with prices above key averages.")
        elif sent <= 40:
            reasons.append("Short-term trend is under pressure, with prices losing strength.")

        qual = group_scores.get("Quality", 50)
        if qual < 50:
            reasons.append("The stock has been more volatile recently, so risk is higher.")
        else:
            reasons.append("Price swings have been moderate, which keeps risk relatively controlled.")

        val = group_scores.get("Valuation", 50)
        if val >= 75:
            reasons.append("Current price is close to its recent average, so valuation looks reasonable.")
        elif val <= 40:
            reasons.append("Price is far away from its recent average, which can mean stretched valuation or stress.")

        prof = group_scores.get("Profitability", 50)
        if prof >= 70:
            reasons.append("The medium-term trend has been steadily upward.")

        if own:
            reasons.append("You already own this stock; owner flow provides sell-only or neutral guidance.")

        if insufficient_data:
            reasons.append("Because data history is short, the forecast has lower confidence.")

        reasons.append("Model used: ensemble (Random Forest + Gradient Boosting).")

        seen = set()
        cleaned = []
        for r in reasons:
            if r not in seen:
                cleaned.append(r)
                seen.add(r)

        # ---- live stats (with prev close + circuits) ----
        prev_close = None
        try:
            daily5 = yf_t.history(period="5d", interval="1d")
            if daily5 is not None and len(daily5) >= 2:
                prev_close = float(daily5["Close"].iloc[-2])
        except Exception:
            prev_close = last_close_hist

        if info.get("previousClose"):
            try:
                prev_close = float(info["previousClose"])
            except Exception:
                pass

        upper_circuit = prev_close * (1 + DEFAULT_CIRCUIT_BAND) if prev_close else None
        lower_circuit = prev_close * (1 - DEFAULT_CIRCUIT_BAND) if prev_close else None

        if f52h is None or f52l is None:
            try:
                hist_1y = yf_t.history(period="1y")
                if hist_1y is not None and not hist_1y.empty:
                    use_col_1y = "Adj Close" if "Adj Close" in hist_1y.columns else "Close"
                    f52h = float(hist_1y[use_col_1y].max())
                    f52l = float(hist_1y[use_col_1y].min())
            except Exception:
                pass

        live_stats = {
            "company_name": company_name,
            "symbol": symbol,
            "exchange": exchange,
            "currency": currency,
            "today_open": today_open,
            "today_high": today_high,
            "today_low": today_low,
            "today_close": today_close,
            "prev_close": prev_close,
            "today_volume": today_volume,
            "upper_circuit": upper_circuit,
            "lower_circuit": lower_circuit,
            "fifty_two_week_high": f52h,
            "fifty_two_week_low": f52l,
            "market_cap": mcap,
        }

        # ---- News (already fetched above as raw_news) ----
        news_items = []
        try:
            raw_news = getattr(yf_t, "news", []) or []
            for item in raw_news[:8]:
                title = item.get("title")
                link = item.get("link")
                source = item.get("publisher") or item.get("provider")
                if title and link:
                    news_items.append({"title": title, "link": link, "source": source or ""})
        except Exception:
            news_items = []

        # ---- model confidence ----
        confidence = "Low"
        ms = result.get("model_score")
        if isinstance(ms, (float, int)):
            if ms >= 0.4:
                confidence = "High"
            elif ms >= 0.2:
                confidence = "Medium"

        strategy_summary = (
            "This view is based mainly on recent price behaviour and model patterns. "
            "It is a general, short-term technical view and not personalised investment advice."
        )

        # ---- recommendation text (owner sell-only)
        owns = bool(own)
        decision_upper = (decision or "HOLD").upper()
        recommendation_primary = ""
        recommendation_options = []

        if not owns:
            # Non-owners: keep full recommendations
            if decision_upper == "BUY":
                recommendation_primary = "Buy: the setup looks favourable for a fresh entry."
                recommendation_options = [
                    "✅ Trend and model score support a positive view in the near term.",
                    "⚠ Enter with proper position sizing and a clear stop-loss.",
                ]
            elif decision_upper == "SELL":
                recommendation_primary = "Don’t buy now – risk looks elevated at current levels."
                recommendation_options = [
                    "⚠ Short-term signals are weak or negative at this price.",
                    "💡 You can re-evaluate when momentum and score improve.",
                ]
            else:  # HOLD
                recommendation_primary = "Don’t buy now – the signal is largely neutral."
                recommendation_options = [
                    "ℹ The model does not see a strong edge to buy at this price.",
                    "💡 Waiting for a clearer uptrend can improve risk–reward.",
                ]
        else:
            # Owners: SELL -> consider selling; otherwise neutral (don't sell now)
            if decision_upper == "SELL":
                recommendation_primary = "Consider selling – downside risk currently looks higher."
                recommendation_options = []
            else:
                recommendation_primary = "Don’t sell now – the signal is largely neutral."
                recommendation_options = []

        # Final result update
        result.update(
            {
                "forecast_price": float(forecast_price),
                "forecast_date": forecast_date,
                "score": final_score,
                "decision": decision_upper,
                "group_scores": group_scores,
                "reasons": cleaned,
                "live_stats": live_stats,
                "company": {"name": company_name, "symbol": symbol, "exchange": exchange},
                "news": news_items,
                "model_confidence": confidence,
                "strategy_summary": strategy_summary,
                "recommendation_primary": recommendation_primary,
                "recommendation_options": recommendation_options,
            }
        )

        # log prediction
        log_row = {
            "timestamp": pd.Timestamp.utcnow().isoformat(),
            "ticker": ticker,
            "last_close": last_close,
            "forecast_price": float(forecast_price),
            "forecast_date": forecast_date,
            "expected_pct": expected_pct,
            "decision": decision_upper,
            "score": final_score,
            "model_r2": result.get("model_score"),
        }
        _append_prediction_log(log_row)
        LOG.info(f"Predicted {ticker} => {forecast_price:.2f} ({decision_upper}, score={final_score})")

        return result

    except Exception as e:
        LOG.exception("Unexpected error in run_analysis")
        result["reasons"].append(f"Unexpected error while analysing the stock: {e}")
        try:
            t = _normalize_ticker(ticker)
            hist = yf.Ticker(t).history(period=chart_range)
            if hist is not None and not hist.empty:
                use_col = "Adj Close" if "Adj Close" in hist.columns else "Close"
                last_close = float(hist[use_col].iloc[-1])
                result["forecast_price"] = last_close
                result["forecast_date"] = _format_forecast_date_from_index(hist.index[-1])
        except Exception:
            pass
        return result
