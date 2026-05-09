"""Portfolio construction. Frozen.

Takes a list of trades (entry/exit dates, signed returns) and produces a
daily portfolio return series with proper capital allocation:

  1. Cap entries-per-day to `max_entries_per_day`, ranked by p_profit.
  2. Each trade is allocated 1 / max_entries_per_day of capital.
  3. The realised P&L is *spread* across the holding period (uniform across
     business days from entry+1 to exit), not dumped on entry day.
  4. Days with fewer open positions than `max_entries_per_day` leave excess
     capital in cash (zero return) — bounds leverage.

This eliminates the over-compounding artefact of the v1 aggregator and
produces an honest equity curve.
"""
from __future__ import annotations
import pandas as pd
import numpy as np


def construct_portfolio(
    trades: pd.DataFrame,
    max_entries_per_day: int = 5,
) -> pd.DataFrame:
    """trades columns required: entry_date, exit_date, symbol, p_profit, ret_net.

    Returns DataFrame indexed by business date with columns:
      ret        - portfolio daily return
      n_active   - number of open positions on that day
    """
    cols = ["entry_date", "exit_date", "symbol", "p_profit", "ret_net"]
    if trades.empty or any(c not in trades.columns for c in cols):
        return pd.DataFrame(columns=["ret", "n_active"])

    df = trades[cols].copy()
    df["entry_date"] = pd.to_datetime(df["entry_date"])
    df["exit_date"] = pd.to_datetime(df["exit_date"])
    # cap entries per day by p_profit
    df["rank"] = df.groupby("entry_date")["p_profit"].rank(ascending=False, method="first")
    df = df[df["rank"] <= max_entries_per_day]
    if df.empty:
        return pd.DataFrame(columns=["ret", "n_active"])

    # spread each trade's net return across business days (entry, exit]
    rows = []
    cap_share = 1.0 / max_entries_per_day
    for _, r in df.iterrows():
        days = pd.bdate_range(start=r["entry_date"] + pd.Timedelta(days=1), end=r["exit_date"])
        n = len(days)
        if n == 0:
            continue
        per_day = (r["ret_net"] / n) * cap_share
        for d in days:
            rows.append({"date": d, "contrib": per_day, "symbol": r["symbol"]})

    if not rows:
        return pd.DataFrame(columns=["ret", "n_active"])
    cdf = pd.DataFrame(rows)
    daily = cdf.groupby("date").agg(
        ret=("contrib", "sum"),
        n_active=("symbol", "nunique"),
    ).sort_index()
    return daily
