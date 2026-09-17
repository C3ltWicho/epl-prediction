"""End-to-end EPL prediction pipeline: load data -> validate -> engineer features
-> walk-forward backtest 4 models -> compare -> save best model -> demo the live
prediction API.
"""

import json
import pathlib

import joblib
import pytest

from src.backtest import evaluate_model_performance, print_model_comparison_table, walk_forward_backtest
from src.data_loader import load_epl_data
from src.data_validator import canonicalize_team_names, report_data_quality, validate_dataset
from src.features import FEATURE_COLUMNS, build_feature_table, compute_elo_ratings
from src.models import EloModel, LogisticRegressionModel, MajorityBaseline, XGBoostModel
from src.predict import Predictor

START_SEASON = 2015
END_SEASON = 2026  # includes the in-progress 2026-27 season, up through the most recent played matchday
VALIDATION_SEASONS = [2021, 2022, 2023, 2024, 2025, 2026]
BEST_MODEL_NAME = "XGBoost"


def main():
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
    joblib.dump(best_model, "models/xgboost_v001.pkl")

    print("\nTesting the live prediction API...")
    _, elo_ratings = compute_elo_ratings(df)
    predictor = Predictor("models/xgboost_v001.pkl", history_df=df, elo_ratings=elo_ratings)

    sample_home, sample_away = df.iloc[-1]["HomeTeam"], df.iloc[-1]["AwayTeam"]
    sample_date = df.iloc[-1]["Date"]
    result = predictor.predict_match(sample_home, sample_away, sample_date)
    print("\nExample prediction (most recent match in the dataset):")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
