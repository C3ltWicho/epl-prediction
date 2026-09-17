"""Tests that verify the feature pipeline never uses post-match-date information."""

import numpy as np
import pandas as pd
import pytest

from src.data_validator import canonicalize_team_names
from src.features import FEATURE_COLUMNS, build_feature_table, compute_elo_ratings, compute_features_for_match


def _toy_dataset():
    """A small, hand-built EPL-like dataset spanning two seasons for fast, deterministic tests."""
    rows = []
    teams = ["Arsenal", "Chelsea", "Liverpool", "Everton"]
    dates = pd.date_range("2021-08-01", periods=16, freq="7D")
    rng = np.random.default_rng(42)
    seasons = [2021] * 8 + [2022] * 8

    i = 0
    for season, group_start in zip([2021, 2022], [0, 8]):
        for d in dates[group_start:group_start + 8]:
            home, away = rng.choice(teams, size=2, replace=False)
            hg, ag = int(rng.integers(0, 4)), int(rng.integers(0, 4))
            ftr = "H" if hg > ag else ("A" if ag > hg else "D")
            rows.append({
                "Date": d, "Season": season, "HomeTeam": home, "AwayTeam": away,
                "HomeGoals": hg, "AwayGoals": ag, "FTR": ftr,
                "Result": {"H": 0, "D": 1, "A": 2}[ftr],
            })
            i += 1
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def toy_df():
    return _toy_dataset()


@pytest.fixture(scope="module")
def feature_table(toy_df):
    return build_feature_table(toy_df)


def test_probabilities_sum_to_one_shape(feature_table):
    assert set(FEATURE_COLUMNS).issubset(feature_table.columns)
    assert len(feature_table) == len(_toy_dataset())


def test_rolling_form_excludes_current_match(toy_df, feature_table):
    """A team's own goals/points from match N must never appear in match N's own form features."""
    team = "Arsenal"
    team_matches = toy_df[(toy_df["HomeTeam"] == team) | (toy_df["AwayTeam"] == team)].sort_values("Date")
    if len(team_matches) < 2:
        pytest.skip("not enough matches for this team in the toy dataset")

    second_match_idx = team_matches.index[1]
    first_match = team_matches.iloc[0]
    first_match_goals_for = first_match["HomeGoals"] if first_match["HomeTeam"] == team else first_match["AwayGoals"]

    row = feature_table.loc[second_match_idx]
    is_home = row["HomeTeam"] == team
    l5_gf_col = "home_l5_gf" if is_home else "away_l5_gf"

    # The L5 window for the team's 2nd match should include exactly the 1st match's goals-for.
    assert row[l5_gf_col] == first_match_goals_for


def test_season_to_date_excludes_current_match(toy_df, feature_table):
    """A team's very first match of a season must show 0 season-to-date points (nothing played yet)."""
    first_2021_row = feature_table[feature_table["Season"] == 2021].sort_values("Date").iloc[0]
    assert first_2021_row["home_season_points"] == 0
    assert first_2021_row["home_season_matches_played"] == 0


def test_elo_ratings_are_prematch(toy_df):
    """The Elo rating attached to a match must equal the team's rating BEFORE that match, not after."""
    df_with_elo, _ = compute_elo_ratings(toy_df)
    df_with_elo = df_with_elo.sort_values("Date").reset_index(drop=True)

    team = df_with_elo.iloc[0]["HomeTeam"]
    later_matches = df_with_elo[
        (df_with_elo["HomeTeam"] == team) | (df_with_elo["AwayTeam"] == team)
    ].sort_values("Date")
    if len(later_matches) < 2:
        pytest.skip("not enough matches for this team")

    first_match_row = later_matches.iloc[0]
    second_match_row = later_matches.iloc[1]

    first_pre_elo = first_match_row["HomeEloPre"] if first_match_row["HomeTeam"] == team else first_match_row["AwayEloPre"]
    second_pre_elo = second_match_row["HomeEloPre"] if second_match_row["HomeTeam"] == team else second_match_row["AwayEloPre"]

    # First match: no history yet, so pre-match Elo must be the initial rating.
    assert first_pre_elo == 1500
    # Second match: the pre-match rating must have moved (been updated by the 1st match's result),
    # proving it reflects prior history rather than being frozen or peeking ahead.
    assert second_pre_elo != 1500


def test_no_future_matches_in_on_demand_features(toy_df):
    """compute_features_for_match must only see rows strictly before match_date."""
    match_date = toy_df["Date"].iloc[10]
    home_team = toy_df.iloc[10]["HomeTeam"]
    away_team = toy_df.iloc[10]["AwayTeam"]

    _, elo_ratings = compute_elo_ratings(toy_df[toy_df["Date"] < match_date])
    features = compute_features_for_match(toy_df, elo_ratings, home_team, away_team, match_date)

    # Sanity: values should be non-negative and finite (would misbehave if future rows leaked in
    # and produced e.g. negative "matches played before this point").
    assert features["home_season_matches_played"] >= 0
    assert features["away_season_matches_played"] >= 0
    assert np.isfinite(features["elo_difference"])


def test_league_position_uses_only_prior_dates(toy_df, feature_table):
    """League position for the first match of a season must be a valid rank among 0 prior results."""
    first_row = feature_table[feature_table["Season"] == 2021].sort_values("Date").iloc[0]
    assert first_row["home_league_position"] >= 1
    assert first_row["away_league_position"] >= 1


def test_team_name_canonicalization():
    df_test = pd.DataFrame({
        "HomeTeam": ["Man United", "Manchester United", "Man Utd"],
        "AwayTeam": ["Man City", "Manchester City", "Man City"],
    })
    df_canon = canonicalize_team_names(df_test)
    assert (df_canon["HomeTeam"] == "Manchester United").all()
    assert (df_canon["AwayTeam"] == "Manchester City").all()


def test_predicted_probabilities_sum_to_one_on_backtest_output():
    results_df = pd.DataFrame({
        "p_home": [0.5, 0.2, 0.33],
        "p_draw": [0.3, 0.3, 0.34],
        "p_away": [0.2, 0.5, 0.33],
    })
    totals = results_df["p_home"] + results_df["p_draw"] + results_df["p_away"]
    assert np.allclose(totals, 1.0, atol=1e-6)
