"""Live prediction API: predict a single EPL match using only pre-kickoff data."""

import joblib
import numpy as np
import pandas as pd

from src.features import FEATURE_COLUMNS, compute_features_for_match

RESULT_NAMES = {0: "HOME_WIN", 1: "DRAW", 2: "AWAY_WIN"}


class Predictor:
    """Wraps a trained model (LogisticRegressionModel or XGBoostModel) for live predictions."""

    def __init__(self, model_path: str, history_df: pd.DataFrame, elo_ratings: dict, feature_cols=None):
        self.model = joblib.load(model_path)
        self.history_df = history_df
        self.elo_ratings = elo_ratings
        self.feature_cols = feature_cols or FEATURE_COLUMNS

    def predict_match(self, home_team: str, away_team: str, match_date) -> dict:
        """
        Predict an EPL match outcome using only data available before match_date.

        Returns a dict with home_win_prob / draw_prob / away_win_prob, the
        predicted result, top contributing features (if the model exposes
        feature importances), and a confidence note.
        """
        match_date = pd.Timestamp(match_date)

        features = compute_features_for_match(
            self.history_df, self.elo_ratings, home_team, away_team, match_date
        )
        X = pd.DataFrame([features])[self.feature_cols].fillna(0)

        probs = self.model.predict_proba(X)[0]
        pred_class = int(np.argmax(probs))
        confidence = float(np.max(probs))

        return {
            "home_team": home_team,
            "away_team": away_team,
            "match_date": str(match_date.date()),
            "home_win_prob": float(probs[0]),
            "draw_prob": float(probs[1]),
            "away_win_prob": float(probs[2]),
            "predicted_result": RESULT_NAMES[pred_class],
            "model_version": "xgboost_v001",
            "top_factors": self._top_factors(),
            "confidence": confidence,
            "confidence_note": (
                "high" if confidence > 0.55 else "moderate" if confidence > 0.40 else "low — close match"
            ),
        }

    def _top_factors(self, n: int = 5):
        if hasattr(self.model, "feature_importance"):
            importance = self.model.feature_importance()
            top = sorted(importance.items(), key=lambda kv: kv[1], reverse=True)[:n]
            return [f"{name}: {score:.3f}" for name, score in top]
        return []
