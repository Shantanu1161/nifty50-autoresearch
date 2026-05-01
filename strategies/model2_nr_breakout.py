"""Model 2 — Volatility-contraction breakout.

Filter at end of day t-1:
  prior-day range in bottom 20% of last 20 days     (low-volatility set-up)
  prior-day volume > 20-day-avg OR > 2-day-avg      (accumulation under the lid)

Entry on day t:
  Long  if high[t] > high[t-1]
  Short if low[t]  < low[t-1]
  else 0
"""
from __future__ import annotations
import pandas as pd


def signals(df: pd.DataFrame, range_pct_threshold: float = 0.20) -> pd.Series:
    range_yest = (df["high"] - df["low"]).shift(1)
    range_pct_yest = df["range_pct20"].shift(1)

    vol_yest = df["volume"].shift(1)
    vol20_yest = df["vol20"].shift(1)
    vol_2d_avg_yest = df["volume"].shift(2).rolling(2).mean()

    setup = (range_pct_yest <= range_pct_threshold) & (
        (vol_yest > vol20_yest) | (vol_yest > vol_2d_avg_yest)
    )

    high_yest = df["high"].shift(1)
    low_yest = df["low"].shift(1)
    long_break = setup & (df["high"] > high_yest)
    short_break = setup & (df["low"] < low_yest)

    sig = pd.Series(0, index=df.index, dtype=int)
    sig[long_break] = 1
    sig[short_break & ~long_break] = -1  # if both, prefer long
    return sig
