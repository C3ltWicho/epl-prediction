"""Edge-case tests for feature engineering: cold starts, unseen teams, missing history."""

import numpy as np
import pandas as pd
import pytest

from src.features import build_feature_table, compute_elo_ratings, compute_features_for_match


def _toy_dataset():
    rows = []
    teams = ["Arsenal", "Brentford", "Chelsea", "Fulham"]
    dates = pd.date_range("2021-08-14", periods=10, freq="7D")
    rng = np.random.default_rng(7)
    for d in dates:
        home, away = rng.choice(teams, size=2, replace=False)
        hg, ag = int(rng.integers(0, 4)), int(rng.integers(0, 4))
        ftr = "H" if hg > ag else ("A" if ag > hg else "D")
        rows.append({
            "Date": d, "Season": 2021, "HomeTeam": home, "AwayTeam": away,
            "HomeGoals": hg, "AwayGoals": ag, "FTR": ftr,
            "Result": {"H": 0, "D": 1, "A": 2}[ftr],
        })
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def toy_df():
    return _toy_dataset()


def test_cold_start_first_match_of_season(toy_df):
    """The very first match in the dataset should still produce valid, non-crashing features."""
    first_row = toy_df.sort_values("Date").iloc[0]
    match_date = first_row["Date"]
    home_team, away_team = first_row["HomeTeam"], first_row["AwayTeam"]

    # There is no history at all before the very first match -> compute_features_for_match
    # should raise a clear error rather than silently fabricating data.
    with pytest.raises(ValueError):
        compute_features_for_match(toy_df, {}, home_team, away_team, match_date)


def test_cold_start_second_match_uses_available_history(toy_df):
    sorted_df = toy_df.sort_values("Date").reset_index(drop=True)
    second_row = sorted_df.iloc[1]
    match_date = second_row["Date"]

    _, elo_ratings = compute_elo_ratings(sorted_df[sorted_df["Date"] < match_date])
    features = compute_features_for_match(
        toy_df, elo_ratings, second_row["HomeTeam"], second_row["AwayTeam"], match_date
    )
    assert features["home_elo"] > 0
    assert "home_season_points" in features


def test_prediction_does_not_crash_on_new_team(toy_df):
    """A team with zero history (e.g. newly promoted) should degrade gracefully, not crash."""
    match_date = toy_df["Date"].max() + pd.Timedelta(days=7)
    features = compute_features_for_match(toy_df, {}, "Arsenal", "Newly Promoted FC", match_date)

    assert features["away_elo"] == 1500  # falls back to the default initial rating
    assert features["away_season_points"] == 0
    assert features["away_l5_points"] == 0
    assert np.isfinite(features["elo_difference"])


def test_build_feature_table_has_no_nans_after_fill(toy_df):
    feature_table = build_feature_table(toy_df)
    numeric_cols = feature_table.select_dtypes(include=[np.number]).columns
    assert not feature_table[numeric_cols].isna().any().any()
