"""
Feature engineering for EPL match prediction.

CRITICAL RULE: for a match on date `match_date`, every feature must be derived
only from matches strictly BEFORE that match. Two code paths enforce this:

1. `build_feature_table(df)` — vectorized, used for training/backtesting on the
   full historical dataset. Leak-freedom comes from `.shift(1)` before every
   rolling/cumulative calculation (so a team's own row is never included in its
   own rolling window) and from date-based cutoffs for league position.

2. `compute_features_for_match(...)` — a direct, on-demand version used by the
   live Predictor API (Phase 7) for a single hypothetical match. It filters the
   history to `Date < match_date` explicitly before computing anything.

Both paths produce the same feature columns (see FEATURE_COLUMNS).
"""

from collections import defaultdict

import numpy as np
import pandas as pd

FORM_WINDOWS = (5, 10)
HOME_AWAY_WINDOW = 5

FEATURE_COLUMNS = [
    "home_elo", "away_elo", "elo_difference",
    "home_season_points", "away_season_points",
    "home_season_gf", "away_season_gf", "home_season_ga", "away_season_ga",
    "home_season_matches_played", "away_season_matches_played",
    "home_league_position", "away_league_position",
    "home_l5_points", "away_l5_points", "home_l5_gf", "away_l5_gf", "home_l5_ga", "away_l5_ga",
    "home_l10_points", "away_l10_points", "home_l10_gf", "away_l10_gf", "home_l10_ga", "away_l10_ga",
    "home_home_win_rate", "away_away_win_rate",
    "home_home_gd", "away_away_gd",
    "attack_vs_defense_home", "attack_vs_defense_away",
]


# ---------------------------------------------------------------------------
# Elo ratings (chronological, leak-free by construction: only prior matches
# ever influence a team's rating going into its next match).
# ---------------------------------------------------------------------------

def compute_elo_ratings(df: pd.DataFrame, k_factor: float = 20, home_advantage: float = 60,
                         initial_elo: float = 1500) -> pd.DataFrame:
    """
    Return `df` (sorted by Date) with two extra columns: HomeEloPre, AwayEloPre —
    each team's Elo rating going INTO that match (i.e. before it is played).
    """
    df = df.sort_values("Date").reset_index(drop=True)
    ratings = defaultdict(lambda: initial_elo)
    home_pre = np.empty(len(df))
    away_pre = np.empty(len(df))

    for i, row in enumerate(df.itertuples(index=False)):
        rh, ra = ratings[row.HomeTeam], ratings[row.AwayTeam]
        home_pre[i] = rh
        away_pre[i] = ra

        expected_home = 1.0 / (1.0 + 10 ** (-((rh + home_advantage) - ra) / 400))
        if row.Result == 0:
            score_home = 1.0
        elif row.Result == 1:
            score_home = 0.5
        else:
            score_home = 0.0

        ratings[row.HomeTeam] = rh + k_factor * (score_home - expected_home)
        ratings[row.AwayTeam] = ra + k_factor * ((1 - score_home) - (1 - expected_home))

    df = df.copy()
    df["HomeEloPre"] = home_pre
    df["AwayEloPre"] = away_pre
    return df, dict(ratings)


# ---------------------------------------------------------------------------
# Long format: one row per (team, match) so we can compute rolling / cumulative
# per-team stats with groupby + shift(1) (which excludes the current match).
# ---------------------------------------------------------------------------

def _build_long_format(df: pd.DataFrame) -> pd.DataFrame:
    home = pd.DataFrame({
        "MatchIndex": df.index,
        "Date": df["Date"], "Season": df["Season"],
        "Team": df["HomeTeam"], "Opponent": df["AwayTeam"],
        "GF": df["HomeGoals"], "GA": df["AwayGoals"],
        "IsHome": True,
        "Points": df["Result"].map({0: 3, 1: 1, 2: 0}),
    })
    away = pd.DataFrame({
        "MatchIndex": df.index,
        "Date": df["Date"], "Season": df["Season"],
        "Team": df["AwayTeam"], "Opponent": df["HomeTeam"],
        "GF": df["AwayGoals"], "GA": df["HomeGoals"],
        "IsHome": False,
        "Points": df["Result"].map({0: 0, 1: 1, 2: 3}),
    })
    long_df = pd.concat([home, away], ignore_index=True)
    long_df = long_df.sort_values(["Team", "Date"]).reset_index(drop=True)
    return long_df


def _shifted_rolling(series: pd.Series, window: int, agg: str) -> pd.Series:
    shifted = series.shift(1)
    rolling = shifted.rolling(window, min_periods=1)
    return getattr(rolling, agg)()


def _add_rolling_form(long_df: pd.DataFrame) -> pd.DataFrame:
    for window in FORM_WINDOWS:
        for col, agg in (("Points", "sum"), ("GF", "sum"), ("GA", "sum")):
            out_col = f"L{window}_{col}"
            long_df[out_col] = long_df.groupby("Team")[col].transform(
                lambda s, w=window, a=agg: _shifted_rolling(s, w, a)
            ).fillna(0)
    return long_df


def _add_season_to_date(long_df: pd.DataFrame) -> pd.DataFrame:
    grouped = long_df.groupby(["Team", "Season"])
    for col in ("Points", "GF", "GA"):
        long_df[f"Season_{col}"] = grouped[col].transform(lambda s: s.shift(1).cumsum()).fillna(0)
    long_df["SeasonMatchesPlayed"] = grouped.cumcount()
    return long_df


def _add_home_away_splits(long_df: pd.DataFrame) -> pd.DataFrame:
    long_df["HomeWinFlag"] = np.where(long_df["IsHome"] & (long_df["Points"] == 3), 1.0, np.nan)
    long_df["AwayWinFlag"] = np.where(~long_df["IsHome"] & (long_df["Points"] == 3), 1.0, np.nan)

    home_rows = long_df[long_df["IsHome"]].sort_values(["Team", "Date"]).copy()
    home_rows["Home_L5_WinRate"] = home_rows.groupby("Team")["Points"].transform(
        lambda s: _shifted_rolling((s == 3).astype(float), HOME_AWAY_WINDOW, "mean")
    )
    home_rows["Home_L5_GD"] = home_rows.groupby("Team").apply(
        lambda g: _shifted_rolling(g["GF"], HOME_AWAY_WINDOW, "sum") - _shifted_rolling(g["GA"], HOME_AWAY_WINDOW, "sum")
    ).reset_index(level=0, drop=True)

    away_rows = long_df[~long_df["IsHome"]].sort_values(["Team", "Date"]).copy()
    away_rows["Away_L5_WinRate"] = away_rows.groupby("Team")["Points"].transform(
        lambda s: _shifted_rolling((s == 3).astype(float), HOME_AWAY_WINDOW, "mean")
    )
    away_rows["Away_L5_GD"] = away_rows.groupby("Team").apply(
        lambda g: _shifted_rolling(g["GF"], HOME_AWAY_WINDOW, "sum") - _shifted_rolling(g["GA"], HOME_AWAY_WINDOW, "sum")
    ).reset_index(level=0, drop=True)

    long_df = long_df.merge(
        home_rows[["MatchIndex", "Team", "Home_L5_WinRate", "Home_L5_GD"]],
        on=["MatchIndex", "Team"], how="left",
    )
    long_df = long_df.merge(
        away_rows[["MatchIndex", "Team", "Away_L5_WinRate", "Away_L5_GD"]],
        on=["MatchIndex", "Team"], how="left",
    )
    return long_df


def _league_positions(df: pd.DataFrame) -> pd.DataFrame:
    """
    For every match, rank each team by season-to-date points AS OF that
    calendar date (ties broken by goal difference, then goals for, then name),
    using only matches with Date < match_date. Same-day matches see identical
    standings (date granularity, matching the walk-forward date cutoff used
    elsewhere in this pipeline).
    """
    positions_home = np.empty(len(df))
    positions_away = np.empty(len(df))

    for season, season_df in df.groupby("Season"):
        season_df = season_df.sort_values("Date")
        unique_dates = season_df["Date"].unique()

        points = defaultdict(int)
        goal_diff = defaultdict(int)
        goals_for = defaultdict(int)

        for d in unique_dates:
            standings = sorted(
                points.keys(),
                key=lambda t: (points[t], goal_diff[t], goals_for[t], t),
                reverse=True,
            )
            rank = {team: i + 1 for i, team in enumerate(standings)}
            n_teams_seen = len(standings)

            todays_matches = season_df[season_df["Date"] == d]
            for idx, row in todays_matches.iterrows():
                positions_home[df.index.get_loc(idx)] = rank.get(row["HomeTeam"], n_teams_seen + 1)
                positions_away[df.index.get_loc(idx)] = rank.get(row["AwayTeam"], n_teams_seen + 1)

            for _, row in todays_matches.iterrows():
                h, a, res = row["HomeTeam"], row["AwayTeam"], row["Result"]
                hg, ag = row["HomeGoals"], row["AwayGoals"]
                if res == 0:
                    points[h] += 3
                elif res == 1:
                    points[h] += 1
                    points[a] += 1
                else:
                    points[a] += 3
                goal_diff[h] += hg - ag
                goal_diff[a] += ag - hg
                goals_for[h] += hg
                goals_for[a] += ag

    df = df.copy()
    df["HomeLeaguePosition"] = positions_home
    df["AwayLeaguePosition"] = positions_away
    return df


def build_feature_table(df: pd.DataFrame, k_factor: float = 20, home_advantage: float = 60) -> pd.DataFrame:
    """
    Build a leak-free feature table for every match in `df`.

    Returns a DataFrame with one row per match: Date, Season, HomeTeam, AwayTeam,
    Result, plus every column in FEATURE_COLUMNS.
    """
    df = df.sort_values("Date").reset_index(drop=True)

    df, final_elo = compute_elo_ratings(df, k_factor=k_factor, home_advantage=home_advantage)
    df = _league_positions(df)

    long_df = _build_long_format(df)
    long_df = _add_rolling_form(long_df)
    long_df = _add_season_to_date(long_df)
    long_df = _add_home_away_splits(long_df)

    # Reindex to df's row order (0..n-1 == MatchIndex) — long_df is sorted by
    # (Team, Date), NOT by match index, so a plain set_index here would silently
    # misalign features to the wrong matches when we later pull out `.values`.
    home_feats = long_df[long_df["IsHome"]].set_index("MatchIndex").reindex(df.index)
    away_feats = long_df[~long_df["IsHome"]].set_index("MatchIndex").reindex(df.index)

    out = pd.DataFrame(index=df.index)
    out["Date"] = df["Date"]
    out["Season"] = df["Season"]
    out["HomeTeam"] = df["HomeTeam"]
    out["AwayTeam"] = df["AwayTeam"]
    out["Result"] = df["Result"]

    out["home_elo"] = df["HomeEloPre"]
    out["away_elo"] = df["AwayEloPre"]
    out["elo_difference"] = out["home_elo"] - out["away_elo"]

    out["home_season_points"] = home_feats["Season_Points"].values
    out["away_season_points"] = away_feats["Season_Points"].values
    out["home_season_gf"] = home_feats["Season_GF"].values
    out["away_season_gf"] = away_feats["Season_GF"].values
    out["home_season_ga"] = home_feats["Season_GA"].values
    out["away_season_ga"] = away_feats["Season_GA"].values
    out["home_season_matches_played"] = home_feats["SeasonMatchesPlayed"].values
    out["away_season_matches_played"] = away_feats["SeasonMatchesPlayed"].values

    out["home_league_position"] = df["HomeLeaguePosition"].values
    out["away_league_position"] = df["AwayLeaguePosition"].values

    for window in FORM_WINDOWS:
        out[f"home_l{window}_points"] = home_feats[f"L{window}_Points"].values
        out[f"away_l{window}_points"] = away_feats[f"L{window}_Points"].values
        out[f"home_l{window}_gf"] = home_feats[f"L{window}_GF"].values
        out[f"away_l{window}_gf"] = away_feats[f"L{window}_GF"].values
        out[f"home_l{window}_ga"] = home_feats[f"L{window}_GA"].values
        out[f"away_l{window}_ga"] = away_feats[f"L{window}_GA"].values

    out["home_home_win_rate"] = home_feats["Home_L5_WinRate"].fillna(0.5).values
    out["away_away_win_rate"] = away_feats["Away_L5_WinRate"].fillna(0.5).values
    out["home_home_gd"] = home_feats["Home_L5_GD"].fillna(0).values
    out["away_away_gd"] = away_feats["Away_L5_GD"].fillna(0).values

    out["attack_vs_defense_home"] = out["home_l5_gf"] - out["away_l5_ga"]
    out["attack_vs_defense_away"] = out["away_l5_gf"] - out["home_l5_ga"]

    return out.reset_index(drop=True)


# ---------------------------------------------------------------------------
# On-demand features for a single hypothetical match (live Predictor API).
# ---------------------------------------------------------------------------

def _team_last_n(df_before: pd.DataFrame, team: str, n: int) -> pd.DataFrame:
    involved = df_before[(df_before["HomeTeam"] == team) | (df_before["AwayTeam"] == team)]
    return involved.sort_values("Date").tail(n)


def _team_form_stats(df_before: pd.DataFrame, team: str, n: int) -> dict:
    matches = _team_last_n(df_before, team, n)
    if matches.empty:
        return {"points": 0, "gf": 0, "ga": 0}
    is_home = matches["HomeTeam"] == team
    gf = np.where(is_home, matches["HomeGoals"], matches["AwayGoals"])
    ga = np.where(is_home, matches["AwayGoals"], matches["HomeGoals"])
    result = matches["Result"].values
    points = np.where(
        is_home,
        np.where(result == 0, 3, np.where(result == 1, 1, 0)),
        np.where(result == 2, 3, np.where(result == 1, 1, 0)),
    )
    return {"points": int(points.sum()), "gf": int(gf.sum()), "ga": int(ga.sum())}


def _team_season_stats(df_before: pd.DataFrame, team: str, season) -> dict:
    df_season = df_before[df_before["Season"] == season]
    return _team_form_stats(df_season, team, n=len(df_season) + 1)


def _team_home_away_split(df_before: pd.DataFrame, team: str, home: bool, n: int) -> dict:
    if home:
        matches = df_before[df_before["HomeTeam"] == team].sort_values("Date").tail(n)
        win_rate = (matches["Result"] == 0).mean() if len(matches) else 0.5
        gd = (matches["HomeGoals"] - matches["AwayGoals"]).sum() if len(matches) else 0
    else:
        matches = df_before[df_before["AwayTeam"] == team].sort_values("Date").tail(n)
        win_rate = (matches["Result"] == 2).mean() if len(matches) else 0.5
        gd = (matches["AwayGoals"] - matches["HomeGoals"]).sum() if len(matches) else 0
    return {"win_rate": float(win_rate), "gd": int(gd)}


def _league_position_asof(df_before: pd.DataFrame, team: str, season) -> int:
    df_season = df_before[df_before["Season"] == season]
    if df_season.empty:
        return len(set(df_before["HomeTeam"]) | set(df_before["AwayTeam"])) or 1

    points, gd = defaultdict(int), defaultdict(int)
    for row in df_season.itertuples(index=False):
        if row.Result == 0:
            points[row.HomeTeam] += 3
        elif row.Result == 1:
            points[row.HomeTeam] += 1
            points[row.AwayTeam] += 1
        else:
            points[row.AwayTeam] += 3
        gd[row.HomeTeam] += row.HomeGoals - row.AwayGoals
        gd[row.AwayTeam] += row.AwayGoals - row.HomeGoals

    standings = sorted(points.keys(), key=lambda t: (points[t], gd[t]), reverse=True)
    if team not in standings:
        return len(standings) + 1
    return standings.index(team) + 1


def compute_features_for_match(df: pd.DataFrame, elo_ratings: dict, home_team: str, away_team: str,
                                match_date, season=None) -> dict:
    """
    Compute the FEATURE_COLUMNS feature dict for one hypothetical match, using
    only rows in `df` with Date < match_date. `elo_ratings` is a {team: rating}
    dict (e.g. the final ratings returned by compute_elo_ratings on history).
    """
    match_date = pd.Timestamp(match_date)
    df_before = df[df["Date"] < match_date]
    if df_before.empty:
        raise ValueError(f"No historical data before {match_date.date()}")

    if season is None:
        season = df_before["Season"].max()

    home_elo = elo_ratings.get(home_team, 1500)
    away_elo = elo_ratings.get(away_team, 1500)

    home_season = _team_season_stats(df_before, home_team, season)
    away_season = _team_season_stats(df_before, away_team, season)
    home_season_matches = len(df_before[(df_before["Season"] == season) &
                                         ((df_before["HomeTeam"] == home_team) | (df_before["AwayTeam"] == home_team))])
    away_season_matches = len(df_before[(df_before["Season"] == season) &
                                         ((df_before["HomeTeam"] == away_team) | (df_before["AwayTeam"] == away_team))])

    home_l5 = _team_form_stats(df_before, home_team, 5)
    away_l5 = _team_form_stats(df_before, away_team, 5)
    home_l10 = _team_form_stats(df_before, home_team, 10)
    away_l10 = _team_form_stats(df_before, away_team, 10)

    home_split = _team_home_away_split(df_before, home_team, home=True, n=5)
    away_split = _team_home_away_split(df_before, away_team, home=False, n=5)

    features = {
        "home_elo": home_elo, "away_elo": away_elo, "elo_difference": home_elo - away_elo,
        "home_season_points": home_season["points"], "away_season_points": away_season["points"],
        "home_season_gf": home_season["gf"], "away_season_gf": away_season["gf"],
        "home_season_ga": home_season["ga"], "away_season_ga": away_season["ga"],
        "home_season_matches_played": home_season_matches, "away_season_matches_played": away_season_matches,
        "home_league_position": _league_position_asof(df_before, home_team, season),
        "away_league_position": _league_position_asof(df_before, away_team, season),
        "home_l5_points": home_l5["points"], "away_l5_points": away_l5["points"],
        "home_l5_gf": home_l5["gf"], "away_l5_gf": away_l5["gf"],
        "home_l5_ga": home_l5["ga"], "away_l5_ga": away_l5["ga"],
        "home_l10_points": home_l10["points"], "away_l10_points": away_l10["points"],
        "home_l10_gf": home_l10["gf"], "away_l10_gf": away_l10["gf"],
        "home_l10_ga": home_l10["ga"], "away_l10_ga": away_l10["ga"],
        "home_home_win_rate": home_split["win_rate"], "away_away_win_rate": away_split["win_rate"],
        "home_home_gd": home_split["gd"], "away_away_gd": away_split["gd"],
    }
    features["attack_vs_defense_home"] = features["home_l5_gf"] - features["away_l5_ga"]
    features["attack_vs_defense_away"] = features["away_l5_gf"] - features["home_l5_ga"]
    return features
