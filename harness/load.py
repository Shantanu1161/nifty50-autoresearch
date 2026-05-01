"""Daily OHLCV loader for Nifty 50. Frozen — agent must not edit.

Caches raw yfinance pulls to data_cache/<symbol>.parquet. Re-run with
force_refresh=True to invalidate.
"""
from __future__ import annotations
import os
from pathlib import Path
import pandas as pd
import numpy as np
import yfinance as yf

CACHE_DIR = Path(__file__).resolve().parent.parent / "data_cache"
CACHE_DIR.mkdir(exist_ok=True)


def _cache_path(symbol: str) -> Path:
    return CACHE_DIR / f"{symbol.replace('.', '_')}.parquet"


def fetch_one(symbol: str, start: str, end: str, force_refresh: bool = False) -> pd.DataFrame:
    p = _cache_path(symbol)
    if p.exists() and not force_refresh:
        df = pd.read_parquet(p)
        if df.index.min() <= pd.Timestamp(start) and df.index.max() >= pd.Timestamp(end) - pd.Timedelta(days=7):
            return df.loc[start:end].copy()
    df = yf.download(symbol, start=start, end=end, progress=False, auto_adjust=False)
    if df is None or df.empty:
        return pd.DataFrame()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.rename(columns=str.lower)[["open", "high", "low", "close", "volume"]]
    df.index = pd.to_datetime(df.index).tz_localize(None)
    df.to_parquet(p)
    return df.loc[start:end].copy()


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["ret"] = df["close"].pct_change()
    df["log_ret"] = np.log(df["close"] / df["close"].shift(1))
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - df["close"].shift(1)).abs(),
        (df["low"] - df["close"].shift(1)).abs(),
    ], axis=1).max(axis=1)
    df["atr14"] = tr.rolling(14).mean()
    df["vol20"] = df["volume"].rolling(20).mean()
    df["range"] = df["high"] - df["low"]
    df["range_pct20"] = df["range"].rolling(20).rank(pct=True)
    return df


def load_universe(symbols: list[str], start: str, end: str, force_refresh: bool = False) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    for sym in symbols:
        df = fetch_one(sym, start, end, force_refresh)
        if df.empty or len(df) < 60:
            continue
        out[sym] = add_features(df)
    return out
