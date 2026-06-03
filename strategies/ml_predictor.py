"""Meta-labeller. AGENT EDITS THIS.

Given a side from a rule-based strategy (+1 long / -1 short) plus features,
predict the probability the trade is profitable. Final signal is
`side * I(P > threshold)`.

Floor model: logistic regression on a small hand-picked feature set, fit
under CPCV. Improvements the agent may try:
  * widen / engineer features
  * swap to gradient boosting / small MLP
  * sweep label barriers
  * tune the probability threshold
  * try sample-weighting by vol or by recency
"""
from __future__ import annotations
import os
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


FEATURES = [
    "ret_lag1", "ret_lag2", "ret_lag5", "ret_lag10",
    "atr_pct", "vol_ratio", "vol_ratio_5d", "atr_ratio_5_20",
    "range_pct20", "close_to_sma20", "close_to_sma50",
    "rsi14", "dist_to_nearest_pivot",
]


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    up = delta.clip(lower=0).rolling(period).mean()
    dn = (-delta.clip(upper=0)).rolling(period).mean()
    rs = up / dn.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _camarilla_levels(h: pd.Series, l: pd.Series, c: pd.Series) -> pd.DataFrame:
    rng = h - l
    out = pd.DataFrame(index=h.index)
    out["P"] = (h + l + c) / 3
    out["R4"] = c + rng * 1.1 / 2
    out["R3"] = c + rng * 1.1 / 4
    out["R2"] = c + rng * 1.1 / 6
    out["R1"] = c + rng * 1.1 / 12
    out["S1"] = c - rng * 1.1 / 12
    out["S2"] = c - rng * 1.1 / 6
    out["S3"] = c - rng * 1.1 / 4
    out["S4"] = c - rng * 1.1 / 2
    return out


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    f = pd.DataFrame(index=df.index)
    f["ret_lag1"] = df["ret"].shift(1)
    f["ret_lag2"] = df["ret"].shift(2)
    f["ret_lag5"] = df["close"].pct_change(5).shift(1)
    f["ret_lag10"] = df["close"].pct_change(10).shift(1)
    f["atr_pct"] = df["atr14"] / df["close"]
    f["vol_ratio"] = df["volume"] / df["vol20"]
    f["vol_ratio_5d"] = df["volume"] / df["volume"].rolling(5).mean()
    f["atr_ratio_5_20"] = df["atr14"] / df["atr14"].rolling(20).mean()
    f["range_pct20"] = df["range_pct20"]
    sma20 = df["close"].rolling(20).mean()
    sma50 = df["close"].rolling(50).mean()
    f["close_to_sma20"] = df["close"] / sma20 - 1
    f["close_to_sma50"] = df["close"] / sma50 - 1
    # RSI(14)
    f["rsi14"] = _rsi(df["close"], 14)
    # Distance to nearest Camarilla pivot (signed % of price; computed from
    # yesterday's H/L/C so it's strictly causal)
    cam = _camarilla_levels(df["high"].shift(1), df["low"].shift(1), df["close"].shift(1))
    cam_arr = cam.to_numpy()
    close_arr = df["close"].to_numpy().reshape(-1, 1)
    dist = (cam_arr - close_arr) / close_arr  # positive = pivot above price
    abs_dist = np.abs(dist)
    nearest_signed = np.full(len(dist), np.nan)
    valid_rows = ~np.all(np.isnan(abs_dist), axis=1)
    if valid_rows.any():
        idx_valid = np.where(valid_rows)[0]
        nearest_idx = np.nanargmin(abs_dist[idx_valid], axis=1)
        nearest_signed[idx_valid] = dist[idx_valid, nearest_idx]
    f["dist_to_nearest_pivot"] = nearest_signed
    return f


# Switch via MODEL_TYPE env var (preferred) or by editing the default below.
MODEL_TYPE = os.environ.get("MODEL_TYPE", "gbm")  # "logistic" or "gbm"


def make_model() -> Pipeline:
    if MODEL_TYPE == "logistic":
        return Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, C=1.0)),
        ])
    elif MODEL_TYPE == "gbm":
        # HistGradientBoosting handles NaN natively — no scaler needed.
        return Pipeline([
            ("clf", HistGradientBoostingClassifier(
                max_iter=200,
                learning_rate=0.05,
                max_depth=4,
                min_samples_leaf=20,
                l2_regularization=1.0,
                random_state=0,
            )),
        ])
    else:
        raise ValueError(f"Unknown MODEL_TYPE: {MODEL_TYPE}")


def fit_predict_cpcv(
    X: pd.DataFrame,
    y: pd.Series,
    side: pd.Series,
    cpcv_iter,
    threshold: float = 0.55,
) -> pd.DataFrame:
    """Returns DataFrame with columns p_profit, signal (side-gated by threshold),
    and a `fold_count` indicating how often each row appeared in test."""
    p_accum = pd.Series(0.0, index=X.index)
    n_accum = pd.Series(0, index=X.index)
    for tr, te in cpcv_iter:
        Xtr, ytr = X.iloc[tr], y.iloc[tr]
        Xte = X.iloc[te]
        # binary target: trade was profitable
        y_bin = (ytr > 0).astype(int)
        if y_bin.nunique() < 2:
            continue
        m = make_model()
        m.fit(Xtr.fillna(0).values, y_bin.values)
        p = m.predict_proba(Xte.fillna(0).values)[:, 1]
        p_accum.iloc[te] += p
        n_accum.iloc[te] += 1
    p_avg = p_accum / n_accum.replace(0, np.nan)
    sig = side.where(p_avg > threshold, 0).fillna(0).astype(int)
    return pd.DataFrame({"p_profit": p_avg, "signal": sig, "fold_count": n_accum})
