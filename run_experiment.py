"""End-to-end experiment runner.

Pipeline:
  1. Load Nifty 50 daily OHLCV (yfinance, cached).
  2. For each ticker:
       - generate side signals from each rule strategy
       - triple-barrier label given side
       - build features
       - run CPCV meta-labeller
  3. Aggregate signals into a portfolio (equal-weight, max N concurrent).
  4. Compute per-split Sharpes -> deflated Sharpe.
  5. Write Excel report.

Agent must NOT edit this file. Edit strategies/ml_predictor.py and program.md.
"""
from __future__ import annotations
import argparse
from datetime import date, datetime
from pathlib import Path
import json
import numpy as np
import pandas as pd

from universe import NIFTY_50, yf_ticker
from harness.load import load_universe
from harness.label import triple_barrier
from harness.cpcv import cpcv_splits
from harness.metrics import sharpe, summarise, deflated_sharpe
from harness.costs import apply_costs
from harness.report import write_report
from harness.portfolio import construct_portfolio
from strategies import model1_camarilla, model2_nr_breakout, ml_predictor

ROOT = Path(__file__).resolve().parent
EXPERIMENTS = ROOT / "experiments"
REPORTS = ROOT / "reports"
EXPERIMENTS.mkdir(exist_ok=True)
REPORTS.mkdir(exist_ok=True)


def per_ticker_pipeline(df: pd.DataFrame, strategy_name: str, vert_days: int,
                        k_up: float, k_dn: float, threshold: float,
                        n_splits: int, k_test: int, embargo_days: int):
    """Returns dict with: signals (post-meta), labels, returns_net, cpcv_split_returns."""
    df = df.dropna().copy()
    if len(df) < 120:
        return None

    if strategy_name == "model1_camarilla":
        side = model1_camarilla.signals(df)
    elif strategy_name == "model2_nr_breakout":
        side = model2_nr_breakout.signals(df)
    elif strategy_name == "always_long":
        side = pd.Series(1, index=df.index)
    else:
        raise ValueError(strategy_name)

    side = side.reindex(df.index).fillna(0).astype(int)
    if (side != 0).sum() < 30:
        return None

    # label only on event rows (where side != 0)
    events = df.loc[side[side != 0].index]
    side_evt = side.loc[events.index]
    lab = triple_barrier(df, side=side, vert_days=vert_days, k_up=k_up, k_dn=k_dn)
    lab = lab.loc[lab.index.intersection(events.index)]
    if len(lab) < 30:
        return None

    feats = ml_predictor.build_features(df).reindex(lab.index)
    keep = feats.dropna().index
    lab = lab.loc[keep]
    feats = feats.loc[keep]
    if len(lab) < 30:
        return None

    ev_times = pd.Series(lab["exit_date"].values, index=lab.index)
    splits = list(cpcv_splits(ev_times, n_splits=n_splits, k_test=k_test, embargo_days=embargo_days))
    if not splits:
        return None

    # binary target = profitable
    y_ret = lab["ret"]
    res = ml_predictor.fit_predict_cpcv(feats, y_ret, lab["side"], iter(splits), threshold=threshold)

    sig = res["signal"]
    gross = y_ret * (sig != 0).astype(int)  # already side-multiplied via triple_barrier(side)
    net = apply_costs(gross, sig)

    # per-split out-of-sample sharpes
    split_sharpes = []
    for tr, te in splits:
        ev_idx = lab.index[te]
        r = net.loc[ev_idx].dropna()
        if len(r) >= 5:
            split_sharpes.append(sharpe(r))

    return {
        "signal": sig,
        "label": lab["label"],
        "ret": y_ret,
        "net": net,
        "split_sharpes": split_sharpes,
        "p_profit": res["p_profit"],
        "exit_date": lab["exit_date"],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2020-01-01")
    ap.add_argument("--end", default=date.today().isoformat())
    ap.add_argument("--strategy", default="model2_nr_breakout",
                    choices=["model1_camarilla", "model2_nr_breakout", "always_long"])
    ap.add_argument("--vert_days", type=int, default=2)
    ap.add_argument("--k_up", type=float, default=1.5)
    ap.add_argument("--k_dn", type=float, default=1.5)
    ap.add_argument("--threshold", type=float, default=0.55)
    ap.add_argument("--n_splits", type=int, default=8)
    ap.add_argument("--k_test", type=int, default=2)
    ap.add_argument("--embargo_days", type=int, default=5)
    ap.add_argument("--top_n_per_day", type=int, default=5)
    ap.add_argument("--max_symbols", type=int, default=50,
                    help="Cap universe for fast iteration")
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()

    syms = [yf_ticker(s) for s in NIFTY_50[: args.max_symbols]]
    print(f"[load] {len(syms)} tickers, {args.start} -> {args.end}")
    data = load_universe(syms, args.start, args.end)
    print(f"[load] {len(data)} loaded")

    per_ticker = {}
    all_split_sharpes = []
    for sym, df in data.items():
        try:
            r = per_ticker_pipeline(
                df, args.strategy, args.vert_days, args.k_up, args.k_dn,
                args.threshold, args.n_splits, args.k_test, args.embargo_days,
            )
        except Exception as e:
            print(f"[skip] {sym}: {e}")
            continue
        if r is None:
            continue
        per_ticker[sym] = r
        all_split_sharpes.extend(r["split_sharpes"])
        print(f"[{sym}] n_trades={(r['signal']!=0).sum()} acc_proxy={(r['net']>0).mean():.3f}")

    # build flat trades record with entry/exit, for the portfolio aggregator
    all_y_true, all_y_pred = [], []
    trades_log = []
    for sym, r in per_ticker.items():
        sig = r["signal"]
        ret = r["ret"]
        mask = sig != 0
        if mask.any():
            all_y_true.extend(np.sign(ret[mask]).fillna(0).astype(int).tolist())
            all_y_pred.extend(sig[mask].astype(int).tolist())
        for t in sig[mask].index:
            trades_log.append({
                "entry_date": t, "exit_date": r["exit_date"].loc[t] if t in r["exit_date"].index else t,
                "symbol": sym, "side": int(sig.loc[t]),
                "p_profit": float(r["p_profit"].loc[t]) if t in r["p_profit"].index else float("nan"),
                "ret_gross": float(ret.loc[t]),
                "ret_net": float(r["net"].loc[t]),
            })
    acc = float(np.mean(np.array(all_y_true) == np.array(all_y_pred))) if all_y_true else 0.0

    trades_df = pd.DataFrame(trades_log)
    print(f"[aggregate] tickers used: {len(per_ticker)}, raw trades: {len(trades_df)}")
    daily = construct_portfolio(trades_df, max_entries_per_day=args.top_n_per_day)
    if daily.empty:
        print("No signals produced. Try lowering threshold or different strategy.")
        return

    daily = daily.sort_index()
    daily["equity"] = (1 + daily["ret"].fillna(0)).cumprod()

    portf_sharpe = sharpe(daily["ret"])
    portf_dd = ((daily["equity"] / daily["equity"].cummax()) - 1).min()

    # deflated Sharpe (treats each per-ticker per-split as a trial)
    obs_sr = portf_sharpe
    p_dsr = deflated_sharpe(obs_sr, all_split_sharpes, n_obs=len(daily))

    summary = {
        "strategy": args.strategy,
        "start": args.start, "end": args.end,
        "n_tickers": len(per_ticker),
        "n_trades": int(sum((r["signal"] != 0).sum() for r in per_ticker.values())),
        "accuracy_on_traded": acc,
        "coverage": float(len(daily) / max(1, (pd.to_datetime(args.end) - pd.to_datetime(args.start)).days * 252 / 365)),
        "sharpe": portf_sharpe,
        "max_drawdown": float(portf_dd),
        "total_return": float(daily["equity"].iloc[-1] - 1),
        "deflated_sharpe_prob": p_dsr,
        "vert_days": args.vert_days, "k_up": args.k_up, "k_dn": args.k_dn,
        "threshold": args.threshold,
        "n_splits": args.n_splits, "k_test": args.k_test, "embargo_days": args.embargo_days,
    }
    print("\n=== SUMMARY ===")
    for k, v in summary.items():
        print(f"  {k}: {v}")

    # build forward-looking signals (last available day per ticker)
    fwd = []
    for sym, r in per_ticker.items():
        sig = r["signal"]
        if (sig != 0).any():
            last_signal_dates = sig[sig != 0].index
            if len(last_signal_dates):
                t = last_signal_dates[-1]
                fwd.append({
                    "date": t, "symbol": sym,
                    "direction": "LONG" if sig.loc[t] > 0 else "SHORT",
                    "p_profit": float(r["p_profit"].loc[t]) if t in r["p_profit"].index else float("nan"),
                    "predicted_pct_move": float(r["net"].loc[t]) * 100,
                    "is_recent": (pd.Timestamp(args.end) - t).days <= 5,
                })
    signals_df = pd.DataFrame(fwd).sort_values(["date", "p_profit"], ascending=[False, False]) if fwd else pd.DataFrame()

    # CPCV distribution
    cpcv_dist = pd.DataFrame({"split_sharpe": all_split_sharpes})

    tag = args.tag or datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = EXPERIMENTS / tag
    exp_dir.mkdir(parents=True, exist_ok=True)
    (exp_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    report_path = REPORTS / f"signals_{tag}.xlsx"
    write_report(report_path, summary, signals_df, daily["equity"], cpcv_dist,
                 trades=pd.DataFrame(trades_log))
    print(f"\n[done] report: {report_path}")
    print(f"[done] experiment dir: {exp_dir}")


if __name__ == "__main__":
    main()
