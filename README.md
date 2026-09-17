# EPL Prediction System

Predicts English Premier League match outcomes (Home Win / Draw / Away Win)
using time-aware machine learning. All features are computed using only
information available before kickoff.

## Quick Start

### Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Run the full pipeline

```bash
python main.py
```

This downloads (and caches) 10 seasons of EPL data, validates and cleans it,
builds the leak-free feature table, runs walk-forward backtesting across four
models, runs the leakage/edge-case test suite, prints a model comparison
table, saves the best model, and demos the live prediction API.

### Make a prediction

```python
import pandas as pd
from src.data_loader import load_epl_data
from src.data_validator import canonicalize_team_names
from src.features import compute_elo_ratings
from src.predict import Predictor

df = canonicalize_team_names(load_epl_data(2015, 2024))
_, elo_ratings = compute_elo_ratings(df)

predictor = Predictor("models/xgboost_v001.pkl", history_df=df, elo_ratings=elo_ratings)
result = predictor.predict_match("Arsenal", "Liverpool", "2024-09-15")

print(f"{result['home_team']} vs {result['away_team']}")
print(f"Home Win: {result['home_win_prob']:.1%}")
print(f"Draw:     {result['draw_prob']:.1%}")
print(f"Away Win: {result['away_win_prob']:.1%}")
print(f"Prediction: {result['predicted_result']}")
```

## Architecture

### Data pipeline

1. **Ingestion** ([`src/data_loader.py`](src/data_loader.py)): downloads each
   season's CSV from football-data.co.uk (cached in `data/raw/`), parses dates
   **per season** (the site's date format changes between seasons — parsing
   after concatenation lets pandas infer one format for the whole column and
   silently turn an entire off-format season into `NaT`, which then vanishes
   via `dropna` with no error).
2. **Validation** ([`src/data_validator.py`](src/data_validator.py)): checks
   for duplicate matches, missing/invalid fields, out-of-range dates, and
   un-canonicalized team names; `canonicalize_team_names` normalizes variants
   like "Man Utd" → "Manchester United".
3. **Features** ([`src/features.py`](src/features.py)): see below.

### Feature engineering — leak-free by construction

The spec sketch computes each match's features by re-filtering the full
history to `Date < match_date` on every call, which is both easy to get wrong
and O(n) per match (O(n²) over a backtest). This implementation instead builds
one vectorized, leak-free feature table for the whole dataset:

- Every rolling/cumulative stat (season-to-date points, last-5/last-10 form,
  home/away splits) is computed per team in chronological order with
  **`.shift(1)` before every `.rolling()`/`.cumsum()`** — so a match's own
  result can never appear in its own features, by construction, not by
  discipline.
- **Elo ratings** are computed once, chronologically, over the full dataset;
  each match stores the rating each team carried **into** that match (not
  after it).
- **League position** is computed per season from a running standings table
  built up date-by-date, using only matches on strictly earlier dates.

A second code path, `compute_features_for_match(...)`, computes the same
feature set on demand for a single hypothetical match (used by the live
`Predictor` API) by filtering history to `Date < match_date` directly — simple
and correct is fine here since it's only ever called once per prediction, not
once per backtested match.

### Models ([`src/models.py`](src/models.py))

All four models share one interface — `fit(df_train)` /
`predict_proba(df_query)` — despite needing different inputs internally (Elo
needs team names, the others need engineered features):

- **Majority**: predicts the most common training-set outcome.
- **Elo**: dynamic ratings + a fixed home-advantage bonus, trained
  chronologically on the training split and then frozen for prediction (no
  in-season updates during validation).
- **Logistic Regression**: multinomial, over the full engineered feature set,
  with standardized inputs.
- **XGBoost**: gradient-boosted trees over the same feature set (best
  performer).

### Evaluation ([`src/backtest.py`](src/backtest.py))

- **Walk-forward backtesting**: train on all seasons strictly before the
  validation season, evaluate on that season, repeat for 2021, 2022, 2023,
  2024. Never a random train/test split.
- **Metrics**: accuracy, log loss, Brier score, macro F1, plus per-class
  (home/draw/away) accuracy from the confusion matrix.

### Leakage tests ([`tests/test_leakage.py`](tests/test_leakage.py), [`tests/test_features.py`](tests/test_features.py))

Run against a small synthetic dataset for speed and determinism:

- A team's own match never appears in its own rolling-form window.
- Season-to-date stats are 0 for a team's first match of a season.
- Elo ratings attached to a match are the pre-match rating, not post-match.
- The on-demand feature function only ever sees rows before `match_date`.
- Cold-start (first match ever, first match of a season, brand-new/promoted
  team with no history) doesn't crash and degrades to sane defaults.
- Team-name canonicalization is idempotent and consistent.
- Backtest probabilities sum to 1.

## Results

Walk-forward backtest, 2015–2024 EPL data (2021–2024 as validation seasons,
1,520 validation matches per model):

| Model | Accuracy | Log Loss | Brier | Macro F1 |
| ----- | -------- | -------- | ----- | -------- |
| Majority | 0.445 | 19.99 | 1.109 | 0.205 |
| Elo | 0.526 | 0.989 | 0.589 | 0.385 |
| Logistic Regression | 0.543 | 0.974 | 0.579 | 0.406 |
| XGBoost | 0.541 | 1.001 | 0.592 | 0.420 |

All three real models comfortably beat the majority-class baseline on every
metric. Logistic Regression and XGBoost are close on accuracy/log-loss;
XGBoost's higher macro F1 comes from better draw/away-class balance. Exact
numbers will shift slightly run-to-run only if the underlying data changes
(models are seeded for reproducibility). Full per-match predictions are in
`validation_results.csv` after running `main.py`.

## Features

### Team strength
- Elo rating (dynamic, pre-match)
- Season-to-date points, goals for/against, matches played
- League position (as of the match date)

### Recent form
- Last 5 / last 10 matches: points, goals for, goals against

### Home/away splits
- Home team's recent home win rate and home goal difference
- Away team's recent away win rate and away goal difference

### Relative strength
- Home attack (L5 goals for) vs. away defense (L5 goals against), and vice versa

## Limitations

- **No injury data**: player availability not included.
- **No squad changes**: transfer windows not accounted for.
- **No betting odds**: market information not used (raw odds columns are
  downloaded but unused — out of scope for this build).
- **No probability calibration, ensembling, or Poisson goal models** — out of
  scope for this build (see Next Steps).
- **Cold start**: a season's opening matches fall back to the previous
  season's Elo rating and show 0 season-to-date stats; a team with zero
  history (e.g. newly promoted, or not yet in the loaded date range) falls
  back to a neutral default rating (1500) and zeroed stats rather than
  crashing.

## Next Steps (future work)

- [ ] Probability calibration (Platt scaling / isotonic)
- [ ] Ensemble: combine Elo + Logistic Regression + XGBoost
- [ ] Poisson goal model
- [ ] Incorporate xG if/when available from the data source
- [ ] Betting-odds experiment (columns are already downloaded, just unused)

## Testing

```bash
pytest tests/ -v
```

- `tests/test_leakage.py`: verifies no future data ever reaches a feature.
- `tests/test_features.py`: cold-start and unseen-team edge cases.

## References

- Football data: [football-data.co.uk](https://www.football-data.co.uk)
- Elo ratings: [Wikipedia](https://en.wikipedia.org/wiki/Elo_rating_system)
