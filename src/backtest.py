"""Time-aware (walk-forward) backtesting for EPL prediction models."""

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, log_loss

from src.features import FEATURE_COLUMNS

RESULT_NAMES = {0: "HOME_WIN", 1: "DRAW", 2: "AWAY_WIN"}


def walk_forward_backtest(feature_table: pd.DataFrame, models_dict: dict, validation_seasons: list) -> pd.DataFrame:
    """
    For each season in `validation_seasons`, train every model on all seasons
    strictly before it and evaluate on that season. Each model only ever sees
    training data with Date/Season earlier than what it's scored against.

    `feature_table` is the output of features.build_feature_table (one row per
    match, already leak-free). Returns a long DataFrame of per-match predictions.
    """
    results = []

    for val_season in validation_seasons:
        train_end_season = val_season - 1
        df_train = feature_table[feature_table["Season"] <= train_end_season].copy()
        df_val = feature_table[feature_table["Season"] == val_season].copy()

        if df_train.empty or df_val.empty:
            print(f"Skipping validation season {val_season}: insufficient data.")
            continue

        print(f"\n{'=' * 60}")
        print(f"Train: seasons <= {train_end_season}  ({len(df_train)} matches)")
        print(f"Validate: season {val_season}  ({len(df_val)} matches)")
        print(f"{'=' * 60}")

        for model_name, model in models_dict.items():
            print(f"  Training {model_name}...")
            model.fit(df_train)

            probs = model.predict_proba(df_val)
            pred_class = np.argmax(probs, axis=1)

            for i, (_, row) in enumerate(df_val.iterrows()):
                results.append({
                    "date": row["Date"],
                    "season": val_season,
                    "home_team": row["HomeTeam"],
                    "away_team": row["AwayTeam"],
                    "model": model_name,
                    "p_home": probs[i, 0],
                    "p_draw": probs[i, 1],
                    "p_away": probs[i, 2],
                    "predicted_result": int(pred_class[i]),
                    "actual_result": int(row["Result"]),
                })

    return pd.DataFrame(results)


def evaluate_model_performance(results_df: pd.DataFrame, model_name: str) -> dict:
    """Compute accuracy, log loss, Brier score, and macro F1 for one model's predictions."""
    model_results = results_df[results_df["model"] == model_name]

    y_true = model_results["actual_result"].values
    y_pred = model_results["predicted_result"].values
    y_proba = model_results[["p_home", "p_draw", "p_away"]].values.astype(np.float64)
    # XGBoost's predict_proba is float32; renormalize away sub-1e-6 rounding drift
    # so sklearn's log_loss doesn't warn that rows don't sum to exactly 1.
    y_proba = y_proba / y_proba.sum(axis=1, keepdims=True)

    y_true_onehot = np.eye(3)[y_true]
    brier = np.mean(np.sum((y_proba - y_true_onehot) ** 2, axis=1))

    metrics = {
        "model": model_name,
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
        "log_loss": log_loss(y_true, y_proba, labels=[0, 1, 2]),
        "brier_score": brier,
        "matches": len(model_results),
    }

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    metrics["home_accuracy"] = cm[0, 0] / cm[0].sum() if cm[0].sum() > 0 else 0.0
    metrics["draw_accuracy"] = cm[1, 1] / cm[1].sum() if cm[1].sum() > 0 else 0.0
    metrics["away_accuracy"] = cm[2, 2] / cm[2].sum() if cm[2].sum() > 0 else 0.0

    return metrics


def print_model_comparison_table(metrics_list: list) -> pd.DataFrame:
    df_metrics = pd.DataFrame(metrics_list)
    print("\n" + "=" * 80)
    print("MODEL COMPARISON TABLE")
    print("=" * 80)
    print(df_metrics[["model", "accuracy", "log_loss", "brier_score", "macro_f1", "matches"]]
          .to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("=" * 80)
    return df_metrics
