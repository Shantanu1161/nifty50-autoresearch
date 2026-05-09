# Nifty 50 Autoresearch — Daily Directional Predictor

Daily-frequency directional + magnitude predictor for Nifty 50 constituents,
built around an Andrej-Karpathy-style autoresearch experimentation loop.
Validation is honest: Combinatorial Purged Cross-Validation, triple-barrier
labelling, realistic Indian-equity cost model, and a portfolio aggregator
that spreads P&L across holding periods (no single-day dump artifacts).

## Status

**End of foundations + first methodology iteration.** Two material upgrades
since the initial baseline:

1. **Portfolio aggregator** — capital is allocated across `max_entries_per_day`
   slots; trade P&L is spread uniformly across business days from entry+1 to
   exit; idle capital sits in cash. This eliminated the v1 over-compounding
   artifact and collapsed drawdowns from -74% → -16-29% with no model change.
2. **Gradient-boosting option** added to the meta-labeller (HistGBM). Selectable
   via `MODEL_TYPE` in `strategies/ml_predictor.py`. GBM gives lower accuracy
   but better-calibrated probabilities and substantially better Sharpe / DD.

### Headline results (CPCV-validated, Nifty 50, 2020 → Apr 2025)

| Run | Strategy | Model | Threshold | k | Trades | Accuracy | Sharpe | Max DD | Total Ret |
|---|---|---|---|---|---|---|---|---|---|
| v1 reference | NR | Logistic (old agg) | 0.65 | 1.5 | 748 | 57.0% | -0.78 | -63% | -46% |
| v1 reference | Camarilla | Logistic (old agg) | 0.60 | 1.5 | 718 | 53.3% | +0.23 | -56% | +0.7% |
| v2 step 1 | NR | GBM | 0.55 | 1.5 | 1,370 | 53.9% | -0.17 | -16% | -5% |
| v2 step 2 | NR | Logistic | 0.55 | 1.5 | 1,433 | 55.6% | -1.10 | -29% | -28% |
| **v2 step 3 (best risk-adjusted)** | NR | GBM | 0.55 | 2.0 | 1,389 | 54.3% | **-0.15** | **-16%** | **-4.9%** |
| **v2 step 3 (best accuracy)** | NR | Logistic | 0.55 | 2.0 | 1,450 | **56.1%** | -0.95 | -27% | -25% |

The v1 headlines (57%, +0.23) were partially aggregation artifacts — the old
"dump on entry date" aggregator both overstated compounding *and* understated
diversification, so neither figure is a reliable apples-to-apples comparison
to v2. The v2 numbers are the honest baseline going forward.

### Reading the results

- **57.0% accuracy was real**, but the equity curve under the v1 aggregator
  was misleading. The signal still has 5-7% edge over coin-flip on traded
  days under proper validation.
- **-16% drawdown + -4.9% total return over 5 years** is the first
  credibly defensible risk profile this project has produced.
- **Triple-barrier multiplier sweep** (k=1.0, 1.5, 2.0, 2.5) showed monotonic
  improvement up to k=2.0; above that the labels collapse to fixed-horizon
  and add no information. k=2.0 stacks cleanly with both logistic and GBM.
- **60% directional accuracy target still unreached.** Next big lever is
  intraday TBT order-flow features (teammate's pipeline), not more model
  classes.

Industry context: sustainable >55% directional accuracy on liquid daily
equities under Combinatorial Purged Cross-Validation is genuine alpha.
Most published "70-95% on Nifty/Sensex" claims do not survive purged CV
and are not tradeable.

## What is built

### Frozen research harness — `harness/`
- `load.py` — yfinance daily OHLCV loader with parquet cache
- `label.py` — triple-barrier labelling (López de Prado), volatility-scaled
- `cpcv.py` — Combinatorial Purged Cross-Validation with purge + 5-day embargo
- `portfolio.py` — proper portfolio aggregator with capital allocation and
  spread P&L
- `metrics.py` — Sharpe, max drawdown, accuracy, coverage, deflated Sharpe, PBO
- `costs.py` — Indian-market intraday-equity frictions (STT, brokerage, slippage)
- `report.py` — Excel writer (Summary / Signals / Equity / CPCV / Trades)

### Strategies — `strategies/`
- `model1_camarilla.py` — Volume + Camarilla R4/S4 breakout & reversal
- `model2_nr_breakout.py` — NR-style volatility-contraction breakout
- `ml_predictor.py` — Meta-labeller (logistic OR GBM, switchable by `MODEL_TYPE`)

### Autoresearch loop — `program.md`, `run_experiment.py`
Karpathy `autoresearch` pattern: agent edits `ml_predictor.py` only, runs
one CPCV pass per experiment, decides keep/discard, commits to git.

## Running it

```bash
pip install -r requirements.txt

# v2 best: NR breakout with GBM
# (set MODEL_TYPE = "gbm" in strategies/ml_predictor.py first)
python3 run_experiment.py --strategy model2_nr_breakout --tag my_run

# Sweep threshold
python3 run_experiment.py --strategy model2_nr_breakout --threshold 0.65 --tag strict

# All flags
python3 run_experiment.py --help
```

Outputs land in `experiments/<tag>/summary.json` and `reports/signals_<tag>.xlsx`.

## Repo layout

```
.
├── universe.py                   # Nifty 50 constituent list
├── harness/                      # FROZEN — agent must not edit
│   ├── load.py                   # yfinance loader + cache
│   ├── label.py                  # triple-barrier labels
│   ├── cpcv.py                   # combinatorial purged CV
│   ├── portfolio.py              # capital allocation + spread P&L
│   ├── metrics.py                # Sharpe, DD, accuracy, deflated Sharpe
│   ├── costs.py                  # Indian-market frictions
│   └── report.py                 # Excel writer
├── strategies/
│   ├── model1_camarilla.py       # FROZEN — Volume + Camarilla rule
│   ├── model2_nr_breakout.py     # FROZEN — NR-style breakout rule
│   └── ml_predictor.py           # AGENT EDITS THIS — meta-labeller
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

1. **Integrate intraday TBT order-flow features** as daily aggregates — biggest
   expected accuracy lift (highest-leverage step toward the 60% target).
2. **Per-symbol or per-cluster meta-labellers** — current model trains one global
   classifier across all tickers; per-sector or per-vol-regime models may help.
3. **Vol-targeted position sizing** instead of equal-weight slot allocation.
4. **Run autoresearch agent overnight** to sweep features, thresholds, and
   barrier multipliers under a fixed compute budget.

## Open architectural decisions

- Universe handling: current Nifty 50 vs point-in-time membership.
- Data source: yfinance (v1) → Zerodha Kite Connect (v2, free historical from Feb 2025).
- Allowed search space for the autoresearch agent.
