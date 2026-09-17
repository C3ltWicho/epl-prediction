"""Download and load historical EPL match data from football-data.co.uk."""

import pathlib

import pandas as pd
import requests

BASE_URL = "https://www.football-data.co.uk/mmz4281/{code}/E0.csv"

# Columns we care about from the raw football-data.co.uk CSVs.
RAW_COLUMNS = [
    "Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR",
    "HS", "AS", "HST", "AST", "HF", "AF", "HC", "AC", "HY", "AY", "HR", "AR", "Referee",
]

RESULT_CODE = {"H": 0, "D": 1, "A": 2}


def _season_code(season_start_year: int) -> str:
    """2015 -> '1516' (football-data.co.uk's season code for 2015-16)."""
    start = season_start_year % 100
    end = (season_start_year + 1) % 100
    return f"{start:02d}{end:02d}"


def _raw_csv_path(season_start_year: int, raw_dir: pathlib.Path) -> pathlib.Path:
    return raw_dir / f"E0_{season_start_year}_{season_start_year + 1}.csv"


def download_season(season_start_year: int, raw_dir: str = "data/raw", force: bool = False) -> pathlib.Path:
    """Download one season's CSV to disk if not already cached. Returns the local path."""
    raw_dir = pathlib.Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    dest = _raw_csv_path(season_start_year, raw_dir)

    if dest.exists() and not force:
        return dest

    url = BASE_URL.format(code=_season_code(season_start_year))
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    if len(response.content) < 100:
        raise ValueError(f"Downloaded file for season {season_start_year} looks empty ({url})")

    dest.write_bytes(response.content)
    return dest


def _load_one_season(season_start_year: int, raw_dir: str) -> pd.DataFrame:
    path = download_season(season_start_year, raw_dir)
    df = pd.read_csv(path)

    missing = [c for c in RAW_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Season {season_start_year} CSV is missing columns: {missing}")

    df = df[RAW_COLUMNS].copy()
    df["Season"] = season_start_year
    # Parse dates per-season, not after concatenation: football-data.co.uk changes
    # date format across seasons (e.g. DD/MM/YYYY vs DD/MM/YY), and pandas infers a
    # single format across a mixed column, silently turning an entire off-format
    # season into NaT (which then gets dropped downstream and vanishes with no error).
    # Try both known formats explicitly rather than falling back to slow per-row
    # dateutil inference.
    parsed = pd.to_datetime(df["Date"], format="%d/%m/%Y", errors="coerce")
    still_missing = parsed.isna()
    if still_missing.any():
        parsed.loc[still_missing] = pd.to_datetime(df.loc[still_missing, "Date"], format="%d/%m/%y", errors="coerce")
    df["Date"] = parsed
    return df


def load_epl_data(start_season: int = 2015, end_season: int = 2024, raw_dir: str = "data/raw") -> pd.DataFrame:
    """
    Load EPL match data for seasons [start_season, end_season] (inclusive, by season start year).

    Returns a DataFrame with at least:
        Date, Season, HomeTeam, AwayTeam, HomeGoals, AwayGoals, Result (0=Home,1=Draw,2=Away)
    plus the raw match-stat columns (shots, corners, cards, etc.) for feature engineering.
    """
    frames = []
    for year in range(start_season, end_season + 1):
        try:
            frames.append(_load_one_season(year, raw_dir))
        except Exception as exc:
            print(f"  WARNING: could not load season {year}-{year + 1}: {exc}")

    if not frames:
        raise RuntimeError("No EPL seasons could be loaded.")

    df = pd.concat(frames, ignore_index=True)

    df = df.rename(columns={
        "FTHG": "HomeGoals",
        "FTAG": "AwayGoals",
        "FTR": "FTR",
    })
    df["Result"] = df["FTR"].map(RESULT_CODE)

    df = df.dropna(subset=["Date", "HomeTeam", "AwayTeam", "Result"]).copy()
    df["HomeGoals"] = df["HomeGoals"].astype(int)
    df["AwayGoals"] = df["AwayGoals"].astype(int)
    df["Result"] = df["Result"].astype(int)

    df = df.sort_values("Date").reset_index(drop=True)
    return df
