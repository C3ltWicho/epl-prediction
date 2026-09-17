"""
Baseline and learned models for EPL match outcome prediction.

Common interface (deliberately unified across all four models, since the spec's
per-model signatures don't agree with each other — Elo needs team names, the
others need engineered features):

    model.fit(df_train)             # df_train has Result + FEATURE_COLUMNS + HomeTeam/AwayTeam
    model.predict_proba(df_query)   # -> np.ndarray shape (n, 3), columns [P(Home), P(Draw), P(Away)]

Class labels: 0 = Home win, 1 = Draw, 2 = Away win.
"""

import numpy as np
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from src.features import FEATURE_COLUMNS, compute_elo_ratings

N_CLASSES = 3
EMPIRICAL_DRAW_PROB = 0.25  # long-run EPL draw rate, used to shape the Elo model's 3-class output


class MajorityBaseline:
    """Always predicts the most common class seen in training."""

    def __init__(self):
        self.most_common = 1

    def fit(self, df_train):
        self.most_common = int(df_train["Result"].mode().iloc[0])
        return self

    def predict_proba(self, df_query):
        probs = np.zeros((len(df_query), N_CLASSES))
        probs[:, self.most_common] = 1.0
        return probs


class EloModel:
    """
    Elo rating system with a fixed home-advantage bonus. Ratings are trained
    once, chronologically, on df_train and then frozen for prediction (no
    in-season updates during validation), per the weekend-scope spec.
    """

    def __init__(self, k_factor: float = 20, home_advantage: float = 60, initial_elo: float = 1500):
        self.k_factor = k_factor
        self.home_advantage = home_advantage
        self.initial_elo = initial_elo
        self.elo_ratings = {}

    def fit(self, df_train):
        _, self.elo_ratings = compute_elo_ratings(
            df_train, k_factor=self.k_factor, home_advantage=self.home_advantage, initial_elo=self.initial_elo
        )
        return self

    def _expected_home_score(self, home_team, away_team):
        rh = self.elo_ratings.get(home_team, self.initial_elo)
        ra = self.elo_ratings.get(away_team, self.initial_elo)
        return 1.0 / (1.0 + 10 ** (-((rh + self.home_advantage) - ra) / 400))

    def predict_proba(self, df_query):
        probs = np.zeros((len(df_query), N_CLASSES))
        for i, row in enumerate(df_query.itertuples(index=False)):
            p_home_win_binary = self._expected_home_score(row.HomeTeam, row.AwayTeam)

            p_draw = EMPIRICAL_DRAW_PROB
            p_home = p_home_win_binary * (1 - p_draw)
            p_away = (1 - p_home_win_binary) * (1 - p_draw)
            probs[i] = [p_home, p_draw, p_away]
        return probs


class LogisticRegressionModel:
    """Multinomial logistic regression over the engineered feature set."""

    def __init__(self, feature_cols=None):
        self.feature_cols = feature_cols or FEATURE_COLUMNS
        self.model = LogisticRegression(max_iter=2000, random_state=42)
        self.scaler = StandardScaler()

    def fit(self, df_train):
        X = df_train[self.feature_cols].fillna(0)
        y = df_train["Result"]
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        return self

    def predict_proba(self, df_query):
        X = df_query[self.feature_cols].fillna(0)
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)


class XGBoostModel:
    """Gradient-boosted trees over the engineered feature set."""

    def __init__(self, feature_cols=None, n_estimators=200, max_depth=4, learning_rate=0.05):
        self.feature_cols = feature_cols or FEATURE_COLUMNS
        self.model = xgb.XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            random_state=42,
            objective="multi:softprob",
            num_class=N_CLASSES,
            eval_metric="mlogloss",
        )

    def fit(self, df_train):
        X = df_train[self.feature_cols].fillna(0)
        y = df_train["Result"]
        self.model.fit(X, y)
        return self

    def predict_proba(self, df_query):
        X = df_query[self.feature_cols].fillna(0)
        return self.model.predict_proba(X)

    def feature_importance(self):
        return dict(zip(self.feature_cols, self.model.feature_importances_))
