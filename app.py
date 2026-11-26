"""Flask web UI for the EPL prediction system.

Loads the trained model and match history once at startup, then serves a
single page: pick two teams and a date, get win/draw/loss probabilities.
"""

from datetime import date, timedelta

from flask import Flask, render_template, request

from src.data_loader import load_epl_data
from src.data_validator import TEAM_NAME_MAP, canonicalize_team_names
from src.predict import DEFAULT_MODEL_PATH, Predictor

START_SEASON = 2015
END_SEASON = 2026

app = Flask(__name__)

print("Loading EPL history and model (this happens once at startup)...")
_history_df = canonicalize_team_names(load_epl_data(START_SEASON, END_SEASON))
_predictor = Predictor(DEFAULT_MODEL_PATH, history_df=_history_df)
_teams = sorted(set(_history_df["HomeTeam"].unique()) | set(_history_df["AwayTeam"].unique()))
print(f"Ready — {len(_teams)} teams, history through {_history_df['Date'].max().date()}.")


@app.route("/", methods=["GET", "POST"])
def index():
    default_date = (date.today() + timedelta(days=1)).isoformat()
    form = {"home": "", "away": "", "date": default_date}
    result = None
    error = None

    if request.method == "POST":
        form["home"] = request.form.get("home", "")
        form["away"] = request.form.get("away", "")
        form["date"] = request.form.get("date") or default_date

        home = TEAM_NAME_MAP.get(form["home"], form["home"])
        away = TEAM_NAME_MAP.get(form["away"], form["away"])

        if not home or not away:
            error = "Pick a home team and an away team."
        elif home == away:
            error = "Home and away teams must be different."
        elif home not in _teams or away not in _teams:
            error = "Unrecognized team — pick one from the list."
        else:
            try:
                result = _predictor.predict_match(home, away, form["date"])
                result["home_win_pct"] = round(result["home_win_prob"] * 100, 1)
                result["draw_pct"] = round(result["draw_prob"] * 100, 1)
                result["away_win_pct"] = round(result["away_win_prob"] * 100, 1)
            except ValueError as exc:
                error = str(exc)

    return render_template("index.html", teams=_teams, form=form, result=result, error=error)


if __name__ == "__main__":
    app.run(debug=True, port=5050)
