"""Triple-barrier labelling (Lopez de Prado). Frozen.

For each event timestamp t with side s in {-1, +1}:
  upper = price_t * (1 + k_up * vol_t)        if s=+1, this is TP
  lower = price_t * (1 - k_dn * vol_t)        if s=+1, this is SL
  vertical = t + vert_days
Whichever barrier is hit first determines the label.
Return label is signed by side: +1 = profitable trade, -1 = stop, 0 = timeout.

For directional (no-side) labels we use side=+1 and treat label sign as the
forward direction.
"""
from __future__ import annotations
import pandas as pd
import numpy as np


def triple_barrier(
    df: pd.DataFrame,
    side: pd.Series | int = 1,
    vert_days: int = 2,
    k_up: float = 1.5,
    k_dn: float = 1.5,
    vol_col: str = "atr14",
    price_col: str = "close",
) -> pd.DataFrame:
    """Returns DataFrame indexed by event date with columns:
    label (+1/-1/0), ret (signed return realised at exit), exit_date, side."""
    if isinstance(side, int):
        side = pd.Series(side, index=df.index)

    price = df[price_col]
    vol = df[vol_col] / price  # ATR as fraction of price
    out = []

    idx = df.index
    n = len(df)
    for i in range(n - 1):
        t = idx[i]
        s = side.iloc[i]
        if s == 0 or pd.isna(vol.iloc[i]):
            continue
        p0 = price.iloc[i]
        v = vol.iloc[i]
        if v <= 0 or np.isnan(v):
            continue
        upper = p0 * (1 + k_up * v)
        lower = p0 * (1 - k_dn * v)

        end_i = min(i + vert_days, n - 1)
        future = df.iloc[i + 1 : end_i + 1]

        label = 0
        exit_price = price.iloc[end_i]
        exit_date = idx[end_i]
        for j, row in future.iterrows():
            hit_up = row["high"] >= upper
            hit_dn = row["low"] <= lower
            if hit_up and hit_dn:
                # both touched same bar — assume worst case (SL first)
                if s > 0:
                    label, exit_price, exit_date = -1, lower, j
                else:
                    label, exit_price, exit_date = -1, upper, j
                break
            if hit_up:
                if s > 0:
                    label, exit_price, exit_date = 1, upper, j
                else:
                    label, exit_price, exit_date = -1, upper, j
                break
            if hit_dn:
                if s > 0:
                    label, exit_price, exit_date = -1, lower, j
                else:
                    label, exit_price, exit_date = 1, lower, j
                break

        ret = s * (exit_price - p0) / p0
        if label == 0:
            # timeout — sign the label by realised direction
            label = int(np.sign(ret)) if ret != 0 else 0
        out.append((t, label, ret, exit_date, int(s)))

    return pd.DataFrame(out, columns=["t", "label", "ret", "exit_date", "side"]).set_index("t")


def fixed_horizon_label(df: pd.DataFrame, horizon: int = 2, price_col: str = "close") -> pd.DataFrame:
    """Simple sign-of-forward-return label. Useful for sanity checks vs triple-barrier."""
    fwd_ret = df[price_col].shift(-horizon) / df[price_col] - 1
    label = np.sign(fwd_ret).fillna(0).astype(int)
    out = pd.DataFrame({"label": label, "ret": fwd_ret, "side": 1})
    out["exit_date"] = df.index.to_series().shift(-horizon)
    return out.dropna(subset=["ret"])
