"""Live prediction API: predict a single EPL match using only pre-kickoff data."""

import pathlib

import joblib
import numpy as np
import pandas as pd

from src.features import FEATURE_COLUMNS, compute_elo_ratings, compute_features_for_match

RESULT_NAMES = {0: "HOME_WIN", 1: "DRAW", 2: "AWAY_WIN"}
DEFAULT_MODEL_PATH = "models/logreg_v002.pkl"


class Predictor:
    """Wraps a trained, saved model for live predictions."""

    def __init__(self, model_path: str, history_df: pd.DataFrame, feature_cols=None):
        self.model = joblib.load(model_path)
        self.model_version = pathlib.Path(model_path).stem
        self.history_df = history_df
        self.feature_cols = feature_cols or FEATURE_COLUMNS

    def predict_match(self, home_team: str, away_team: str, match_date) -> dict:
        """
        Predict an EPL match outcome using only data available before match_date.

        Returns a dict with home_win_prob / draw_prob / away_win_prob, the
        predicted result, the factors behind this particular prediction (if the
        model can explain itself), and a confidence note.
        """
        match_date = pd.Timestamp(match_date)

        # Elo must be rebuilt per request: ratings built from the full history
        # would leak results from on or after match_date.
        history_before = self.history_df[self.history_df["Date"] < match_date]
        _, elo_ratings = compute_elo_ratings(history_before)
        features = compute_features_for_match(
            history_before, elo_ratings, home_team, away_team, match_date
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
            "model_version": self.model_version,
            "top_factors": self._top_factors(X, pred_class),
            "confidence": confidence,
            "confidence_note": (
                "high" if confidence > 0.55 else "moderate" if confidence > 0.40 else "low — close match"
            ),
        }

    def _top_factors(self, X: pd.DataFrame, pred_class: int, n: int = 5) -> list:
        """The features that moved this prediction most, compared with an average match."""
        if not hasattr(self.model, "explain"):
            return []
        effects = self.model.explain(X, pred_class)
        top = sorted(effects, key=lambda name: abs(effects[name]), reverse=True)[:n]
        return [
            {"feature": name, "value": round(float(X[name].iloc[0]), 2), "effect": round(effects[name], 3)}
            for name in top
        ]
