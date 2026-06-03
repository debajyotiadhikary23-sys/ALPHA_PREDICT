"""
backtester.py

Simple event-driven backtester using the decision logic in chart_model.run_analysis.
Simulates naive daily trading:
  - decision == BUY  -> go long with full capital
  - decision == SELL -> go to cash
No transaction costs by default.

Outputs basic metrics:
  - total_return
  - CAGR
  - annualized return
  - annualized volatility
  - Sharpe ratio
  - max drawdown
  - number of trades
  - win-rate
"""

import pandas as pd
import numpy as np
import chart_model
import logging

LOG = logging.getLogger("backtester")
LOG.setLevel(logging.INFO)
if not LOG.handlers:
    import sys
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s %(message)s"))
    LOG.addHandler(ch)


def compute_metrics(equity_series, freq=252):
    """
    Compute basic performance metrics from an equity curve.
    """
    returns = equity_series.pct_change().dropna()
    if returns.empty:
        return {}
    total_return = equity_series.iloc[-1] / equity_series.iloc[0] - 1
    n_years = len(equity_series) / freq
    if n_years <= 0:
        cagr = 0.0
    else:
        cagr = (equity_series.iloc[-1] / equity_series.iloc[0]) ** (1 / n_years) - 1

    ann_ret = returns.mean() * freq
    ann_vol = returns.std() * (freq ** 0.5)
    sharpe = ann_ret / ann_vol if ann_vol != 0 else None

    roll_max = equity_series.cummax()
    drawdown = (equity_series - roll_max) / roll_max
    max_dd = drawdown.min()
    return {
        "total_return": float(total_return),
        "cagr": float(cagr),
        "ann_return": float(ann_ret),
        "ann_vol": float(ann_vol),
        "sharpe": float(sharpe) if sharpe is not None else None,
        "max_drawdown": float(max_dd)
    }


def run_backtest(ticker, chart_range="5y", initial_capital=100000, verbose=False):
    """
    Run a simple daily backtest using the model's BUY / SELL decisions.

    Logic:
      - Use historical close prices from chart_model.fetch_price_history()
      - Each day, call chart_model.run_analysis(ticker, chart_range=chart_range, own=False)
      - If decision == BUY  and we are in cash -> buy with full capital at next day's close
      - If decision == SELL and we are in a position -> sell at next day's close
      - HOLD -> do nothing
    """
    ticker = ticker or ""
    hist = chart_model.fetch_price_history(ticker, chart_range=chart_range)
    if not hist:
        raise ValueError("No history for backtest")

    df = pd.DataFrame(hist)
    df["date"] = pd.to_datetime(df["x"])
    df["close"] = df["c"].astype(float)
    df = df.set_index("date").sort_index()

    capital = initial_capital
    position = 0.0  # number of shares
    equity = []
    dates = []
    wins = 0
    trades = 0
    entry_equity = None

    # warmup: start after 60 days
    for i in range(60, len(df) - 1):
        # For realism we pass a fixed chart_range (e.g. 5y), model uses full history.
        try:
            res = chart_model.run_analysis(ticker, chart_range=chart_range, own=False)
            decision = (res.get("decision") or "HOLD").upper()
        except Exception:
            decision = "HOLD"

        next_close = float(df["close"].iloc[i + 1])

        if decision == "BUY" and position == 0:
            # buy as many shares as possible
            position = capital / next_close
            capital = 0.0
            trades += 1
            entry_equity = position * next_close

        elif decision == "SELL" and position > 0:
            proceeds = position * next_close
            profit = proceeds - (entry_equity if entry_equity is not None else proceeds)
            capital = proceeds
            position = 0.0
            trades += 1
            if profit > 0:
                wins += 1
            entry_equity = None

        cur_equity = capital + position * next_close
        equity.append(cur_equity)
        dates.append(df.index[i + 1])

    equity_series = pd.Series(equity, index=dates)
    metrics = compute_metrics(equity_series)
    metrics.update({
        "trades": trades,
        "wins": wins,
        "win_rate": (wins / trades) if trades > 0 else None
    })

    if verbose:
        LOG.info(f"Backtest {ticker} metrics: {metrics}")
    return metrics


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--ticker", required=True)
    p.add_argument("--range", default="5y")
    p.add_argument("--initial_capital", type=float, default=100000)
    args = p.parse_args()
    out = run_backtest(args.ticker, chart_range=args.range, initial_capital=args.initial_capital, verbose=True)
    print(out)
