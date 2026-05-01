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
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


FEATURES = [
    "ret_lag1", "ret_lag2", "ret_lag5",
    "atr_pct", "vol_ratio", "range_pct20",
    "close_to_sma20", "close_to_sma50",
]


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    f = pd.DataFrame(index=df.index)
    f["ret_lag1"] = df["ret"].shift(1)
    f["ret_lag2"] = df["ret"].shift(2)
    f["ret_lag5"] = df["close"].pct_change(5).shift(1)
    f["atr_pct"] = df["atr14"] / df["close"]
    f["vol_ratio"] = df["volume"] / df["vol20"]
    f["range_pct20"] = df["range_pct20"]
    sma20 = df["close"].rolling(20).mean()
    sma50 = df["close"].rolling(50).mean()
    f["close_to_sma20"] = df["close"] / sma20 - 1
    f["close_to_sma50"] = df["close"] / sma50 - 1
    return f


def make_model() -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, C=1.0)),
    ])


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
