"""Model 1 — Volume + Camarilla pivots.

Filter: today's volume > vol_mult * 20-day-avg-volume
Entry on day t (signal applies to forward 2 days):
  Long  if close[t] > R4[t]                      (breakout)
  Long  if low[t] <= S4[t] and close[t] > S4[t]  (S4 reversal)
  Short if close[t] < S4[t]                      (breakdown)
  Short if high[t] >= R4[t] and close[t] < R4[t] (R4 rejection)
  else 0

Camarilla pivots are computed from day t's H/L/C (the user's spec uses
intraday levels off the prior day; for a daily-bar model we adapt: the
signal at end-of-day t informs the next 2 days).
"""
from __future__ import annotations
import pandas as pd
import numpy as np


def camarilla(h: pd.Series, l: pd.Series, c: pd.Series) -> pd.DataFrame:
    rng = h - l
    p = (h + l + c) / 3
    df = pd.DataFrame(index=h.index)
    df["P"] = p
    df["R4"] = c + rng * 1.1 / 2
    df["R3"] = c + rng * 1.1 / 4
    df["R2"] = c + rng * 1.1 / 6
    df["R1"] = c + rng * 1.1 / 12
    df["S1"] = c - rng * 1.1 / 12
    df["S2"] = c - rng * 1.1 / 6
    df["S3"] = c - rng * 1.1 / 4
    df["S4"] = c - rng * 1.1 / 2
    return df


def signals(df: pd.DataFrame, vol_mult: float = 2.0) -> pd.Series:
    cam = camarilla(df["high"], df["low"], df["close"])
    vol_ok = df["volume"] > vol_mult * df["vol20"]

    breakout_long = df["close"] > cam["R4"]
    revers_long = (df["low"] <= cam["S4"]) & (df["close"] > cam["S4"])
    breakdown_short = df["close"] < cam["S4"]
    revers_short = (df["high"] >= cam["R4"]) & (df["close"] < cam["R4"])

    sig = pd.Series(0, index=df.index, dtype=int)
    sig[vol_ok & (breakout_long | revers_long)] = 1
    sig[vol_ok & (breakdown_short | revers_short)] = -1
    return sig
