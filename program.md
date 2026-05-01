# Autoresearch program — Nifty 50 daily predictor

You are a research agent running experiments to improve a daily-frequency directional predictor for Nifty 50 stocks. Pattern is from Karpathy's `autoresearch`: single-variable, hypothesis-driven experiments, frozen harness, decide keep/discard each round, log everything to `experiments/`.

## What is FROZEN — you must not edit
- `harness/` — loader, triple-barrier labeller, CPCV, metrics, costs, reporter
- `universe.py` — Nifty 50 constituents
- `run_experiment.py` — pipeline orchestration
- `strategies/model1_camarilla.py`, `strategies/model2_nr_breakout.py` — rule baselines

## What you MAY edit
- `strategies/ml_predictor.py` — the meta-labeller. Current floor is logistic regression with an 8-feature set.
- This `program.md` — when you decide to expand the search space.

## Allowed knobs (edit `ml_predictor.py`)
1. **Features** — add new ones to `FEATURES` and `build_features()`. Candidates: rolling Z-scores of volume / range, distance to nearest SMA/EMA, pivot-relative position, sector dummies (later), realised-vol regimes, rolling Sharpe of recent returns, relative strength vs Nifty index.
2. **Model class** — logistic → `GradientBoostingClassifier` → `lightgbm.LGBMClassifier` → small `MLPClassifier`. Always wrap in `Pipeline([...]) ` with `StandardScaler` for linear/MLP.
3. **Threshold** — `threshold` argument in `fit_predict_cpcv`, default 0.55. Sweep 0.50–0.70.
4. **Sample weights** — pass `sample_weight=` to `.fit()`; weight by inverse class frequency or by recency (exp decay).
5. **Label barriers** — `--k_up` / `--k_dn` to `run_experiment.py`. Sweep 1.0–2.5.
6. **Vertical horizon** — `--vert_days`. Default 2; try 1, 3, 5.

## Forbidden
- Editing the harness, universe, or rule strategies.
- Changing the cost model.
- Changing the CPCV `n_splits` or `embargo_days` (causes apples-to-oranges comparisons across experiments).
- Looking at test data inside `fit_predict_cpcv` (CPCV already enforces this; do not bypass).

## Metric (in priority order)
1. **Deflated Sharpe probability** (P(true SR > 0)) reported in summary as `deflated_sharpe_prob`. Higher is better. Target > 0.90.
2. **Accuracy on traded subset** (`accuracy_on_traded`). Target ≥ 0.58 sustained across CPCV.
3. **Coverage**. Want at least ~25% of trading days with at least one signal across the universe.
4. **Max drawdown**. Want > -15%.

A change is **kept** only if (1) does not regress, (2) accuracy improves or coverage improves at flat accuracy, (3) drawdown does not blow up.

## Experiment loop
1. Read this file and `experiments/<latest>/summary.json` to see current best.
2. Form ONE hypothesis. Example: "Adding rolling volume Z-score will improve P(profit) calibration on contraction set-ups."
3. Edit `ml_predictor.py` (one variable change).
4. Run `python3 run_experiment.py --strategy model2_nr_breakout --tag <YYYYMMDD_short_desc>`.
5. Read `experiments/<tag>/summary.json`. Compare to previous best on the metric ladder above.
6. If kept: commit changes with message `[autoresearch] <tag>: dsr=X acc=Y cov=Z`. If discarded: revert.
7. Repeat.

## Starting baseline (run this first to establish floor)
```
python3 run_experiment.py --strategy model2_nr_breakout --tag baseline_v0
python3 run_experiment.py --strategy model1_camarilla   --tag baseline_v0_m1
```

## Notes
- v1 universe is the *current* Nifty 50, not point-in-time. ~2% accuracy bias; don't celebrate sub-2% improvements.
- Costs are realistic Indian intraday-equity round-trips; do not relax them to "look good".
- If you find yourself adding fallbacks or try/except to suppress errors, stop — surface the error instead.
