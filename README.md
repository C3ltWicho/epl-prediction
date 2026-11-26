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

Run the pipeline once first; it trains and saves the model.

```bash
python main.py predict --home Arsenal --away Liverpool --date 2026-09-19
python main.py teams   # list valid team names
```

Or start the web UI at http://localhost:5050:

```bash
python app.py
```

Or from Python:

```python
from src.data_loader import load_epl_data
from src.data_validator import canonicalize_team_names
from src.predict import DEFAULT_MODEL_PATH, Predictor

df = canonicalize_team_names(load_epl_data(2015, 2026))
predictor = Predictor(DEFAULT_MODEL_PATH, history_df=df)
result = predictor.predict_match("Arsenal", "Liverpool", "2026-09-19")
print(result)
```

`top_factors` lists the features that moved that particular prediction most,
compared with an average match. `effect` is in log-odds units (not percentage
points) toward the predicted result. Many features are correlated, so they
share credit and can partly cancel each other; treat them as a rough guide.

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
- **Elo ratings** are computed in date order; each match stores the rating each
  team carried **into** that match. For live predictions, `Predictor` rebuilds
  them from matches before the requested date.
- **League position** is computed per season from a running standings table
  built up date-by-date, using only matches on strictly earlier dates.

A second code path, `compute_features_for_match(...)`, computes the same
feature set on demand for a single hypothetical match (used by the live
`Predictor` API) by filtering history to `Date < match_date` directly. It's only
called once per prediction, so clarity matters more than speed. A test checks
that both paths give the same values.

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
- **XGBoost**: gradient-boosted trees over the same feature set (untuned).

Logistic Regression is the saved default because it has the lowest log loss.

### Evaluation ([`src/backtest.py`](src/backtest.py))

- **Walk-forward backtesting**: train on all seasons strictly before the
  validation season, evaluate on that season, repeat for each season from 2021
  to 2026 (2026-27 is partial). Never a random train/test split.
- **Metrics**: accuracy, log loss, Brier score, macro F1, plus per-class
  (home/draw/away) accuracy from the confusion matrix.

### Leakage tests ([`tests/test_leakage.py`](tests/test_leakage.py), [`tests/test_features.py`](tests/test_features.py))

Run against a small synthetic dataset for speed and determinism:

- Changing results on or after a date changes nothing predicted for that date,
  in both the training features and the live `Predictor`.
- Training and live feature code agree on every match with prior history.
- A season's opening day starts from zero season stats.
- Elo ratings start at 1500 and are pre-match.
- All four models' probabilities sum to 1.
- Per-match explanations add up to the model's actual output.
- Cold starts (no history, or a team never seen) are handled.
- Team-name variants map to one canonical name.

## Results

Walk-forward backtest, validation seasons 2021-22 through 2026-27 (data through
2026-09-14), 1,940 matches per model:

| Model | Accuracy | Log Loss | Brier | Macro F1 |
| ----- | -------- | -------- | ----- | -------- |
| Majority | 0.439 | 20.21 | 1.122 | 0.203 |
| Elo | 0.510 | 1.002 | 0.598 | 0.376 |
| **Logistic Regression** | 0.525 | **0.990** | 0.591 | 0.393 |
| XGBoost | 0.522 | 1.013 | 0.601 | 0.402 |

All three real models beat the majority-class baseline. The gaps between them
are small. Logistic Regression has the best probability scores (log loss and
Brier); XGBoost has a higher macro F1. Because the default was chosen by looking
at these seasons, its score here is slightly optimistic. The Majority log loss is
large because it assigns 100% to one outcome. The Elo baseline is also
handicapped: its ratings are frozen during each validation season.

Full per-match predictions are in `validation_results.csv`.

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
- **Cold start**: a season's opening matches carry over Elo ratings and recent
  form but show 0 season-to-date stats; a team with zero
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

- `tests/test_leakage.py`: leakage, train/live consistency, and model output checks.
- `tests/test_features.py`: cold-start and unseen-team edge cases.

## References

- Football data: [football-data.co.uk](https://www.football-data.co.uk)
- Elo ratings: [Wikipedia](https://en.wikipedia.org/wiki/Elo_rating_system)
