# Nifty 50 Autoresearch — Daily Directional Predictor

Daily-frequency directional + magnitude predictor for Nifty 50 constituents,
built around an Andrej-Karpathy-style autoresearch experimentation loop.
Validation is honest: Combinatorial Purged Cross-Validation, triple-barrier
labelling, and a realistic Indian-equity cost model.

## Status — Day 1

**End of foundations. Working end-to-end pipeline + 5 baseline runs.**

### Baseline results (Nifty 50, 2020 → Apr 2025, CPCV-validated)

| Run | Strategy | Threshold | Trades | Accuracy | Sharpe | Max DD | Total Ret |
|---|---|---|---|---|---|---|---|
| baseline_v0_m1 | Camarilla | 0.55 | 1,011 | 52.6% | -0.37 | -71% | -42% |
| baseline_v0_m2 | NR breakout | 0.55 | 1,362 | 55.8% | -0.94 | -74% | -60% |
| **m1_thresh060** | Camarilla | 0.60 | 718 | 53.3% | **+0.23** | -56% | **+0.7%** |
| m2_thresh060 | NR breakout | 0.60 | 1,044 | 55.8% | -1.02 | -72% | -59% |
| **m2_thresh065** | NR breakout | 0.65 | 748 | **57.0%** | -0.78 | -63% | -46% |

- Best directional accuracy: **57.0%** (CPCV-validated, 748 trades).
- Best Sharpe: **+0.23** — only run that survived costs end-to-end.
- Drawdowns -50 to -75% are caused by **unfixed portfolio sizing**, not signal quality.

Industry context: sustainable >55% directional accuracy on liquid daily
equities under proper validation is genuine alpha. 60% target is plausible
with two more iterations. Anything ≥65% on daily equities almost always
indicates leakage — most published "70-95% accuracy" claims do not survive
purged cross-validation.

## What is built

### Frozen research harness — `harness/`
- `load.py` — yfinance daily OHLCV loader with parquet cache
- `label.py` — triple-barrier labelling (López de Prado), volatility-scaled barriers
- `cpcv.py` — Combinatorial Purged Cross-Validation with purge + 5-day embargo
- `metrics.py` — Sharpe, max drawdown, accuracy, coverage, deflated Sharpe, PBO
- `costs.py` — Indian-market intraday-equity frictions (STT, brokerage, slippage)
- `report.py` — Excel writer (Summary / Signals / Equity / CPCV / Trades)

### Strategies — `strategies/`
- `model1_camarilla.py` — Volume + Camarilla R4/S4 breakout & reversal
- `model2_nr_breakout.py` — Volatility-contraction (NR-style) breakout
- `ml_predictor.py` — Logistic-regression meta-labeller (ML floor; agent edits this)

### Autoresearch loop — `program.md`, `run_experiment.py`
Karpathy `autoresearch` pattern: agent edits `ml_predictor.py` only, runs
one CPCV pass per experiment, decides keep/discard, commits to git. Frozen
harness prevents the agent from "winning" via leakage.

## Running it

```bash
pip install -r requirements.txt

# baseline
python3 run_experiment.py --strategy model2_nr_breakout --tag baseline_v0_m2

# stricter conviction threshold
python3 run_experiment.py --strategy model2_nr_breakout --threshold 0.65 --tag m2_strict

# all flags
python3 run_experiment.py --help
```

Outputs land in `experiments/<tag>/summary.json` and `reports/signals_<tag>.xlsx`.

## Repo layout

```
.
├── universe.py                   # Nifty 50 constituent list
├── harness/                      # FROZEN — agent must not edit
├── strategies/
│   ├── model1_camarilla.py       # FROZEN
│   ├── model2_nr_breakout.py     # FROZEN
│   └── ml_predictor.py           # AGENT EDITS THIS
├── run_experiment.py             # FROZEN — pipeline orchestration
├── program.md                    # autoresearch agent spec
├── experiments/<tag>/summary.json
└── reports/signals_<tag>.xlsx
```

## Methodology references

- Andrej Karpathy, *autoresearch* — https://github.com/karpathy/autoresearch
- Andrej Karpathy, *A Recipe for Training Neural Networks* (2019) — http://karpathy.github.io/2019/04/25/recipe/
- Marcos López de Prado, *Advances in Financial Machine Learning* (2018) — CPCV, triple-barrier, meta-labelling, deflated Sharpe.

## Next steps

1. Fix portfolio sizing and concurrent-position cap (expected to flip multiple runs Sharpe-positive without touching the model).
2. Swap logistic regression for gradient boosting in the meta-labeller.
3. Integrate intraday TBT order-flow features (buy/sell volume %, aggression score, mid-price relationship) as daily aggregates.
4. Run the autoresearch agent loop overnight once features are wired.

## Open architectural decisions

- Universe handling: current Nifty 50 vs point-in-time membership (survivorship bias).
- Data source: yfinance (v1) → Zerodha Kite Connect (v2, free historical from Feb 2025).
- Allowed search space for the autoresearch agent.
