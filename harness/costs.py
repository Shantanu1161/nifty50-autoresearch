"""Indian-market transaction-cost model. Frozen.

Per-trade frictions (round-trip, intraday equity):
  Brokerage: ~ flat ₹20/order * 2 orders        = ₹40
  STT:       0.025% on sell-side notional
  Exchange + SEBI + GST + stamp ≈ 0.012% notional
  Slippage:  configurable, default 5 bps per side = 10 bps round-trip
"""
from __future__ import annotations
import pandas as pd

DEFAULT_CAPITAL_PER_TRADE = 100_000.0  # INR
BROKERAGE_PER_ORDER = 20.0  # INR
STT_BPS_SELL = 2.5  # 0.025%
EXCHANGE_BPS = 1.2  # 0.012% round-trip approx
SLIPPAGE_BPS_PER_SIDE = 5.0


def cost_bps(slippage_bps_per_side: float = SLIPPAGE_BPS_PER_SIDE,
             capital_per_trade: float = DEFAULT_CAPITAL_PER_TRADE) -> float:
    """Total cost in basis points of notional, round-trip."""
    flat_bps = (BROKERAGE_PER_ORDER * 2) / capital_per_trade * 1e4
    return flat_bps + STT_BPS_SELL + EXCHANGE_BPS + 2 * slippage_bps_per_side


def apply_costs(gross_returns: pd.Series, traded: pd.Series,
                slippage_bps_per_side: float = SLIPPAGE_BPS_PER_SIDE,
                capital_per_trade: float = DEFAULT_CAPITAL_PER_TRADE) -> pd.Series:
    """Subtract round-trip cost from each traded return."""
    c = cost_bps(slippage_bps_per_side, capital_per_trade) / 1e4
    net = gross_returns.copy()
    net = net - traded.astype(int).abs() * c
    return net
