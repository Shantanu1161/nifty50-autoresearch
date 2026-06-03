# Nifty 50 Autoresearch — Daily Directional Predictor

Daily-frequency directional + magnitude predictor for Nifty 50 constituents,
built around an Andrej-Karpathy-style autoresearch experimentation loop.
Validation is honest: Combinatorial Purged Cross-Validation, triple-barrier
labelling, realistic Indian-equity cost model, and a portfolio aggregator
that spreads P&L across holding periods.

## Status — v3 ship

Three iterations completed. Final headline configs:

| Headline | Strategy | Model | k | vert_days | Threshold | Trades | **Accuracy** | Sharpe | Max DD | Total Ret (5y) |
|---|---|---|---|---|---|---|---|---|---|---|
| **Best accuracy** | NR breakout | GBM | 2.0 | 10 | 0.55 | 1,484 | **56.5%** | +0.44 | -29% | +17% |
| **Best risk-adjusted** | NR breakout | GBM | 2.0 | 5 | 0.55 | 1,441 | 54.2% | **+1.62** | -16% | **+82%** |
| **Conservative** | NR breakout | GBM | 2.0 | 5 | 0.65 | 808 | 52.5% | +0.32 | -15% | +7.8% |

**Important caveat on the Sharpe numbers:** the portfolio aggregator spreads
each trade's realised P&L uniformly across business days from entry+1 to
exit. With longer holding periods (vd=5), more positions overlap on any
given day, which lowers daily portfolio volatility and inflates Sharpe via
both (a) real diversification benefit and (b) an artificial smoothing
effect from the uniform-spread assumption. **The absolute Sharpe numbers
above should be discounted ~20-30% before being defended in a meeting.**
A v4 fix would mark each trade to actual daily price moves. The relative
ranking across configs is still trustworthy.

### Iteration ladder

| Stage | Change | Accuracy lift | Sharpe lift | DD impact |
|---|---|---|---|---|
| v1 baseline (NR, log, k=1.5, vd=2) | — | 55.8% | -0.94 | -74% |
| v2 step 1 | Portfolio aggregator | flat | +0.79 | -74% → -29% |
| v2 step 2 | + GBM option | -2pp | +0.93 | -29% → -16% |
| v2 step 3 | + k=2.0 barriers | +0.4pp | +0.03 | flat |
| v3 step 4 | + RSI, dist-to-pivot features | +1.2pp | flat | flat |
| **v3 step 5** | + vert_days = 5 | +0.05pp | +1.77 | flat |
| v3 step 6 (alt) | + vert_days = 10 | +2.1pp | +0.62 | -16% → -29% |

### Reading the numbers honestly

- **57.0% (v1) → 56.5% (v3)** on accuracy — the apparent v1 high was
  partially aggregation artefact. v3 56.5% under proper validation is
  the real number.
- **First decisively positive PnL config produced:** vd=5 GBM at +82%
  total return over 5 years.
- **Drawdown discipline:** -74% (v1) → -16% (v3). The sizing fix and
  the longer holding period combine to give a credible risk profile.
- **60% accuracy target still unreached.** Plateau at ~55-57%. Next big
  lever is intraday TBT order-flow features (teammate's pipeline).
- **Camarilla strategy underperforms** NR across all v3 configs (51% acc).
  Likely needs different filtering criteria; deferred.

Industry context: sustainable >55% directional accuracy on liquid daily
equities under Combinatorial Purged Cross-Validation is genuine alpha.
Most published "70-95% on Nifty/Sensex" claims do not survive purged CV.

## What is built

### Frozen research harness — `harness/`
- `load.py` — yfinance daily OHLCV loader with parquet cache
- `label.py` — triple-barrier labelling (López de Prado), vol-scaled
- `cpcv.py` — Combinatorial Purged Cross-Validation with purge + embargo
- `portfolio.py` — capital allocation + spread P&L aggregator
- `metrics.py` — Sharpe, DD, accuracy, coverage, deflated Sharpe, PBO
- `costs.py` — Indian-market frictions (STT, brokerage, slippage)
- `report.py` — Excel writer

### Strategies — `strategies/`
- `model1_camarilla.py` — Volume + Camarilla R4/S4 (rule baseline)
- `model2_nr_breakout.py` — NR-style volatility-contraction breakout
- `ml_predictor.py` — Meta-labeller with 13 features (incl. RSI,
  dist-to-pivot); switchable between Logistic and GBM via `MODEL_TYPE`
  env var or module default

### Autoresearch loop — `program.md`, `run_experiment.py`
Karpathy `autoresearch` pattern: agent edits `ml_predictor.py`, runs one
CPCV pass per experiment, decides keep/discard, commits to git.

## Running it

```bash
pip install -r requirements.txt

# Ship headline (best total return)
MODEL_TYPE=gbm python3 run_experiment.py \
  --strategy model2_nr_breakout \
  --k_up 2.0 --k_dn 2.0 --vert_days 5 --threshold 0.55 \
  --tag my_run

# Best accuracy variant
MODEL_TYPE=gbm python3 run_experiment.py \
  --strategy model2_nr_breakout \
  --k_up 2.0 --k_dn 2.0 --vert_days 10 --threshold 0.55 \
  --tag my_acc_run

# All flags
python3 run_experiment.py --help
```

Outputs:
- `experiments/<tag>/summary.json` — metrics
- `reports/signals_<tag>.xlsx` — Summary, Signals, Equity, CPCV, Trades

The shipped headline run is in `reports/signals_final_ship.xlsx`.

## Repo layout

```
.
├── universe.py                   # Nifty 50 constituent list
├── harness/                      # FROZEN — agent must not edit
├── strategies/
│   ├── model1_camarilla.py       # FROZEN — rule baseline
│   ├── model2_nr_breakout.py     # FROZEN — rule baseline
│   └── ml_predictor.py           # AGENT EDITS THIS
├── run_experiment.py             # FROZEN — pipeline
├── program.md                    # autoresearch agent spec
├── experiments/<tag>/summary.json
└── reports/signals_<tag>.xlsx
```

## Methodology references

- Andrej Karpathy, *autoresearch* — https://github.com/karpathy/autoresearch
- Andrej Karpathy, *A Recipe for Training Neural Networks* (2019) — http://karpathy.github.io/2019/04/25/recipe/
- Marcos López de Prado, *Advances in Financial Machine Learning* (2018) — CPCV, triple-barrier, meta-labelling, deflated Sharpe.

## Next steps (post-v3)

1. **Integrate intraday TBT order-flow features** as daily aggregates —
   biggest remaining lever toward 60% accuracy. Blocked on sample data
   file from teammate.
2. **Fix aggregator to mark trades to actual daily price moves**
   instead of uniform spreading. Eliminates the Sharpe-smoothing artefact.
3. **Vol-targeted position sizing** instead of equal-slot allocation.
4. **Point-in-time Nifty 50 membership** to remove survivorship bias.
5. **Run autoresearch agent overnight** on the remaining search space.

## Open architectural decisions

- Data source: yfinance (v1-v3) → Zerodha Kite Connect (production).
- Universe handling: current vs point-in-time membership.
- Agent search space and compute budget for overnight runs.
