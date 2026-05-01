"""Excel reporter. Frozen.

Writes one workbook with sheets:
  Summary           — single row of headline metrics
  Signals           — 2-day forward signals (direction + predicted % move)
  Equity            — cumulative equity curve
  CPCV_Distribution — distribution of out-of-sample Sharpes across CPCV splits
  Trades            — per-trade record
"""
from __future__ import annotations
from pathlib import Path
import pandas as pd


def write_report(
    path: str | Path,
    summary: dict,
    signals: pd.DataFrame,
    equity: pd.Series,
    cpcv_dist: pd.DataFrame,
    trades: pd.DataFrame | None = None,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(path, engine="openpyxl") as xw:
        pd.DataFrame([summary]).to_excel(xw, sheet_name="Summary", index=False)
        signals.to_excel(xw, sheet_name="Signals", index=True)
        equity.rename("equity").to_frame().to_excel(xw, sheet_name="Equity", index=True)
        cpcv_dist.to_excel(xw, sheet_name="CPCV_Distribution", index=False)
        if trades is not None and len(trades):
            trades.to_excel(xw, sheet_name="Trades", index=False)
    return path
