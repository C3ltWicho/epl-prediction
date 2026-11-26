"""Tests that verify no prediction ever uses information from its match date or later."""

import joblib
import numpy as np
import pandas as pd
import pytest

from src.data_validator import canonicalize_team_names
from src.features import (
    FEATURE_COLUMNS, build_feature_table, compute_elo_ratings, compute_features_for_match,
)
from src.models import EloModel, LogisticRegressionModel, MajorityBaseline, XGBoostModel
from src.predict import Predictor

TEAMS = ["Arsenal", "Brentford", "Chelsea", "Everton", "Fulham", "Liverpool"]


def _toy_dataset():
    """
    Two synthetic seasons with several matches per date (so standings ties and
    same-day matches occur), and one team swapped out between seasons.
    """
    rng = np.random.default_rng(42)
    rows = []
    for season in (2021, 2022):
        teams = TEAMS if season == 2021 else TEAMS[:-1] + ["Leeds United"]
        for date in pd.date_range(f"{season}-08-07", periods=14, freq="7D"):
            order = rng.permutation(teams)
            for home, away in zip(order[::2], order[1::2]):
                hg, ag = int(rng.integers(0, 4)), int(rng.integers(0, 4))
                rows.append({
                    "Date": date, "Season": season, "HomeTeam": str(home), "AwayTeam": str(away),
                    "HomeGoals": hg, "AwayGoals": ag,
                    "Result": 0 if hg > ag else (2 if ag > hg else 1),
                })
    return pd.DataFrame(rows)


def _rewrite_from(df, cutoff):
    """Replace every result on or after `cutoff` with a lopsided away win."""
    changed = df.copy()
    mask = changed["Date"] >= cutoff
    changed.loc[mask, ["HomeGoals", "AwayGoals", "Result"]] = [0, 9, 2]
    return changed


def _live_features(df, row):
    before = df[df["Date"] < row.Date]
    _, elo = compute_elo_ratings(before)
    return compute_features_for_match(df, elo, row.HomeTeam, row.AwayTeam, row.Date)


@pytest.fixture(scope="module")
def toy_df():
    return _toy_dataset()


@pytest.fixture(scope="module")
def feature_table(toy_df):
    return build_feature_table(toy_df)


@pytest.fixture(scope="module")
def cutoff(toy_df):
    return toy_df["Date"].unique()[18]  # a date partway through the second season


@pytest.fixture(scope="module")
def model_path(feature_table, tmp_path_factory):
    path = tmp_path_factory.mktemp("model") / "toy_logreg.pkl"
    joblib.dump(LogisticRegressionModel().fit(feature_table), path)
    return str(path)


def test_training_features_ignore_same_day_and_later_results(toy_df, feature_table, cutoff):
    changed_table = build_feature_table(_rewrite_from(toy_df, cutoff))
    up_to_cutoff = feature_table["Date"] <= cutoff
    pd.testing.assert_frame_equal(
        feature_table.loc[up_to_cutoff, FEATURE_COLUMNS],
        changed_table.loc[up_to_cutoff, FEATURE_COLUMNS],
    )


@pytest.mark.parametrize("past_position", [0.3, 0.6, 0.95])
def test_predictor_ignores_same_day_and_later_results(toy_df, model_path, past_position):
    """Regression test: Elo built from the full history once leaked into past-date predictions."""
    row = toy_df.iloc[int(len(toy_df) * past_position)]
    original = Predictor(model_path, toy_df).predict_match(row.HomeTeam, row.AwayTeam, row.Date)
    changed = Predictor(model_path, _rewrite_from(toy_df, row.Date)).predict_match(
        row.HomeTeam, row.AwayTeam, row.Date
    )
    assert original == changed


def test_predictor_uses_prematch_elo(toy_df, feature_table, model_path):
    """A live prediction for a past match must match what the model sees for it in training."""
    row = feature_table.iloc[30]
    model = joblib.load(model_path)
    expected = model.predict_proba(feature_table.iloc[[30]])[0]
    result = Predictor(model_path, toy_df).predict_match(row.HomeTeam, row.AwayTeam, row.Date)
    actual = [result["home_win_prob"], result["draw_prob"], result["away_win_prob"]]
    np.testing.assert_allclose(actual, expected)


def test_training_and_live_features_match(toy_df, feature_table):
    """The two feature code paths must agree on every match that has prior history."""
    later_matches = feature_table[feature_table["Date"] > feature_table["Date"].min()]
    mismatches = []
    for row in later_matches.itertuples(index=False):
        live = _live_features(toy_df, row)
        mismatches += [
            (str(row.Date.date()), row.HomeTeam, column, getattr(row, column), live[column])
            for column in FEATURE_COLUMNS
            if not np.isclose(getattr(row, column), live[column])
        ]
    assert not mismatches, f"{len(mismatches)} mismatches, first few: {mismatches[:5]}"


def test_season_opener_starts_from_zero(toy_df):
    """On opening day, live features must not carry over last season's totals."""
    opener = toy_df[toy_df["Season"] == 2022].iloc[0]
    features = _live_features(toy_df, opener)
    assert features["home_season_points"] == 0
    assert features["away_season_matches_played"] == 0


def test_elo_ratings_are_prematch(toy_df):
    with_elo, _ = compute_elo_ratings(toy_df)
    first_day = with_elo[with_elo["Date"] == with_elo["Date"].min()]
    assert (first_day[["HomeEloPre", "AwayEloPre"]] == 1500).all().all()


@pytest.mark.parametrize("model_class", [MajorityBaseline, EloModel, LogisticRegressionModel, XGBoostModel])
def test_model_probabilities_sum_to_one(feature_table, model_class):
    probs = model_class().fit(feature_table).predict_proba(feature_table)
    assert probs.shape == (len(feature_table), 3)
    np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-6)


def test_explanations_reconstruct_the_prediction(feature_table):
    """Per-match factors (plus the intercept) must add up to the model's actual output."""
    model = LogisticRegressionModel().fit(feature_table)
    match = feature_table.iloc[[40]]
    scores = np.array([sum(model.explain(match, c).values()) for c in range(3)]) + model.model.intercept_
    rebuilt = np.exp(scores) / np.exp(scores).sum()
    np.testing.assert_allclose(rebuilt, model.predict_proba(match)[0])


def test_team_name_canonicalization():
    df_test = pd.DataFrame({
        "HomeTeam": ["Man United", "Manchester United", "Man Utd"],
        "AwayTeam": ["Man City", "Manchester City", "Man City"],
    })
    df_canon = canonicalize_team_names(df_test)
    assert (df_canon["HomeTeam"] == "Manchester United").all()
    assert (df_canon["AwayTeam"] == "Manchester City").all()
