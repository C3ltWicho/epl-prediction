"""
EPL prediction CLI.

    python main.py run                                            # full pipeline (default if no subcommand given)
    python main.py predict --home Brentford --away Chelsea --date 2026-09-18
    python main.py teams                                          # list valid team names
"""

import argparse
import difflib
import json
import pathlib
import sys
from datetime import date, timedelta

import joblib
import pytest

from src.backtest import evaluate_model_performance, print_model_comparison_table, walk_forward_backtest
from src.data_loader import load_epl_data
from src.data_validator import TEAM_NAME_MAP, canonicalize_team_names, report_data_quality, validate_dataset
from src.features import FEATURE_COLUMNS, build_feature_table
from src.models import EloModel, LogisticRegressionModel, MajorityBaseline, XGBoostModel
from src.predict import DEFAULT_MODEL_PATH, Predictor

START_SEASON = 2015
END_SEASON = 2026  # includes the in-progress 2026-27 season, up through the most recent played matchday
VALIDATION_SEASONS = [2021, 2022, 2023, 2024, 2025, 2026]
# Chosen for the lowest walk-forward log loss. This was picked by looking at the
# validation seasons, so its score there is slightly optimistic.
BEST_MODEL_NAME = "LogisticRegression"


def run_pipeline(args):
    print("Loading EPL data...")
    df = load_epl_data(START_SEASON, END_SEASON)
    df = canonicalize_team_names(df)

    validate_dataset(df)
    report_data_quality(df)

    pathlib.Path("data/processed").mkdir(parents=True, exist_ok=True)
    df.to_csv("data/processed/epl_matches.csv", index=False)
    print(f"\nSaved cleaned dataset to data/processed/epl_matches.csv ({len(df)} matches)")

    print("\nBuilding leak-free feature table...")
    feature_table = build_feature_table(df)
    feature_table.to_csv("data/processed/epl_features.csv", index=False)
    print(f"Feature table: {feature_table.shape[0]} matches x {len(FEATURE_COLUMNS)} features")

    print("\nRunning walk-forward backtesting across 4 baseline models...")
    models = {
        "Majority": MajorityBaseline(),
        "Elo": EloModel(),
        "LogisticRegression": LogisticRegressionModel(),
        "XGBoost": XGBoostModel(),
    }
    results_df = walk_forward_backtest(feature_table, models, VALIDATION_SEASONS)
    results_df.to_csv("validation_results.csv", index=False)
    print(f"\nSaved {len(results_df)} predictions to validation_results.csv")

    print("\nRunning leakage + feature edge-case tests...")
    test_exit_code = pytest.main(["tests/", "-v"])
    if test_exit_code != 0:
        print("WARNING: tests did not all pass. Review before trusting these results.")

    print("\nEvaluating models...")
    metrics_list = [evaluate_model_performance(results_df, name) for name in models]
    print_model_comparison_table(metrics_list)

    print(f"\nSaving best model ({BEST_MODEL_NAME}) trained on all available data...")
    pathlib.Path("models").mkdir(exist_ok=True)
    best_model = models[BEST_MODEL_NAME]
    best_model.fit(feature_table)
    joblib.dump(best_model, DEFAULT_MODEL_PATH)

    print("\nTesting the live prediction API...")
    predictor = Predictor(DEFAULT_MODEL_PATH, history_df=df)

    sample_home, sample_away = df.iloc[-1]["HomeTeam"], df.iloc[-1]["AwayTeam"]
    sample_date = df.iloc[-1]["Date"]
    result = predictor.predict_match(sample_home, sample_away, sample_date)
    print("\nExample prediction (most recent match in the dataset):")
    print(json.dumps(result, indent=2))


def _load_history(start_season, end_season):
    df = load_epl_data(start_season, end_season)
    return canonicalize_team_names(df)


def _known_teams(df):
    return sorted(set(df["HomeTeam"].unique()) | set(df["AwayTeam"].unique()))


def _closest_matches(name, known_teams, n=5):
    """Substring hits first (handles 'Man' -> 'Manchester United'/'Manchester City'),
    then fuzzy string matches (handles 'Man Utd' -> 'Manchester United')."""
    name_lower = name.lower()
    substring_hits = [t for t in known_teams if name_lower in t.lower()]
    fuzzy_hits = difflib.get_close_matches(name, known_teams, n=n, cutoff=0.4)
    ordered = list(dict.fromkeys(substring_hits + fuzzy_hits))
    return ordered[:n] or sorted(known_teams)[:n]


def cmd_predict(args):
    model_path = pathlib.Path(args.model)
    if not model_path.exists():
        print(f"Error: model file not found at {model_path}. Run `python main.py run` first to train and save it.",
              file=sys.stderr)
        sys.exit(1)

    print(f"Loading history ({args.start_season}-{args.end_season})...", file=sys.stderr)
    df = _load_history(args.start_season, args.end_season)
    known_teams = _known_teams(df)

    home = TEAM_NAME_MAP.get(args.home, args.home)
    away = TEAM_NAME_MAP.get(args.away, args.away)

    for label, team in (("--home", home), ("--away", away)):
        if team not in known_teams:
            suggestions = _closest_matches(team, known_teams)
            print(f"Error: unknown team for {label}: '{team}'", file=sys.stderr)
            print(f"Closest known names: {suggestions}", file=sys.stderr)
            print("Run `python main.py teams` to see the full list.", file=sys.stderr)
            sys.exit(1)

    match_date = args.date or (date.today() + timedelta(days=1)).isoformat()

    predictor = Predictor(str(model_path), history_df=df)

    try:
        result = predictor.predict_match(home, away, match_date)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    print(json.dumps(result, indent=2))


def cmd_teams(args):
    df = _load_history(args.start_season, args.end_season)
    for team in _known_teams(df):
        print(team)


def build_parser():
    parser = argparse.ArgumentParser(description="EPL match outcome prediction system")
    subparsers = parser.add_subparsers(dest="command")

    run_parser = subparsers.add_parser("run", help="Run the full pipeline: load data, train, backtest, save model")
    run_parser.set_defaults(func=run_pipeline)

    predict_parser = subparsers.add_parser("predict", help="Predict a single match with the saved model")
    predict_parser.add_argument("--home", required=True, help="Home team name (see `python main.py teams`)")
    predict_parser.add_argument("--away", required=True, help="Away team name (see `python main.py teams`)")
    predict_parser.add_argument("--date", default=None, help="Match date, YYYY-MM-DD (default: tomorrow)")
    predict_parser.add_argument("--model", default=DEFAULT_MODEL_PATH,
                                help=f"Path to a saved model (default: {DEFAULT_MODEL_PATH})")
    predict_parser.add_argument("--start-season", type=int, default=START_SEASON, dest="start_season")
    predict_parser.add_argument("--end-season", type=int, default=END_SEASON, dest="end_season")
    predict_parser.set_defaults(func=cmd_predict)

    teams_parser = subparsers.add_parser("teams", help="List team names recognized by the loaded dataset")
    teams_parser.add_argument("--start-season", type=int, default=START_SEASON, dest="start_season")
    teams_parser.add_argument("--end-season", type=int, default=END_SEASON, dest="end_season")
    teams_parser.set_defaults(func=cmd_teams)

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    # No subcommand given -> preserve old `python main.py` behavior (run the full pipeline).
    if args.command is None:
        run_pipeline(args)
        return

    args.func(args)


if __name__ == "__main__":
    main()
