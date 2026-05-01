"""Performance metrics. Frozen.

Sharpe, max drawdown, accuracy, coverage, deflated Sharpe (Lopez de Prado),
and probability of backtest overfitting (PBO).
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy import stats


def sharpe(returns: pd.Series, periods_per_year: int = 252) -> float:
    r = returns.dropna()
    if len(r) < 2 or r.std(ddof=1) == 0:
        return 0.0
    return float(r.mean() / r.std(ddof=1) * np.sqrt(periods_per_year))


def max_drawdown(returns: pd.Series) -> float:
    r = returns.dropna()
    if len(r) == 0:
        return 0.0
    eq = (1 + r).cumprod()
    peak = eq.cummax()
    dd = eq / peak - 1
    return float(dd.min())


def accuracy(y_true: pd.Series, y_pred: pd.Series) -> float:
    """Directional accuracy on the *traded* subset (y_pred != 0)."""
    mask = (y_pred != 0) & y_true.notna() & y_pred.notna()
    if mask.sum() == 0:
        return 0.0
    return float((np.sign(y_true[mask]) == np.sign(y_pred[mask])).mean())


def coverage(y_pred: pd.Series) -> float:
    if len(y_pred) == 0:
        return 0.0
    return float((y_pred != 0).mean())


def deflated_sharpe(observed: float, sharpes: list[float], n_obs: int) -> float:
    """Lopez de Prado's deflated Sharpe ratio probability.
    Returns P(true_SR > 0) given the candidate's observed SR was max of `sharpes` trials."""
    if len(sharpes) < 2 or n_obs < 4:
        return float("nan")
    s = np.array(sharpes, dtype=float)
    s = s[~np.isnan(s)]
    if len(s) < 2:
        return float("nan")
    var_s = s.var(ddof=1)
    if var_s <= 0:
        return float("nan")
    # expected max SR under null
    emc = 0.5772156649  # Euler-Mascheroni
    n_trials = len(s)
    e_max = np.sqrt(var_s) * (
        (1 - emc) * stats.norm.ppf(1 - 1.0 / n_trials)
        + emc * stats.norm.ppf(1 - 1.0 / (n_trials * np.e))
    )
    skew = stats.skew(s)
    kurt = stats.kurtosis(s, fisher=True)
    denom = np.sqrt(1 - skew * observed + (kurt / 4.0) * observed ** 2)
    if denom <= 0 or np.isnan(denom):
        return float("nan")
    z = (observed - e_max) * np.sqrt(n_obs - 1) / denom
    return float(stats.norm.cdf(z))


def pbo(rank_matrix: np.ndarray) -> float:
    """Probability of Backtest Overfitting (Bailey, Borwein, Lopez de Prado).
    rank_matrix: rows = strategies, cols = CPCV splits. Each cell = strategy
    rank (lower = better SR) within that split's training half. We approximate
    by computing the share of splits where the in-sample best strategy
    underperforms the median out-of-sample."""
    if rank_matrix.size == 0:
        return float("nan")
    n_strats, n_splits = rank_matrix.shape
    if n_strats < 2 or n_splits < 2:
        return float("nan")
    underperf = 0
    for j in range(n_splits):
        is_best = int(np.argmin(rank_matrix[:, j]))
        oos_ranks = np.delete(rank_matrix, j, axis=1)[is_best]
        median_rank = (n_strats + 1) / 2.0
        if oos_ranks.mean() > median_rank:
            underperf += 1
    return float(underperf / n_splits)


def summarise(returns: pd.Series, y_true: pd.Series, y_pred: pd.Series) -> dict:
    return {
        "sharpe": sharpe(returns),
        "max_drawdown": max_drawdown(returns),
        "accuracy": accuracy(y_true, y_pred),
        "coverage": coverage(y_pred),
        "n_trades": int((y_pred != 0).sum()),
        "total_return": float((1 + returns.fillna(0)).prod() - 1),
        "mean_daily_return": float(returns.mean()),
        "vol_daily": float(returns.std(ddof=1)) if returns.std(ddof=1) == returns.std(ddof=1) else 0.0,
    }
