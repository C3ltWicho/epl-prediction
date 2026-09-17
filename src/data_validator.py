"""Validate and clean the raw EPL match dataset."""

import pandas as pd

# Maps known name variants (as they appear in football-data.co.uk exports over the years,
# and common alternate spellings) to one canonical name.
TEAM_NAME_MAP = {
    "Man United": "Manchester United",
    "Man Utd": "Manchester United",
    "Manchester Utd": "Manchester United",
    "Man City": "Manchester City",
    "Newcastle": "Newcastle United",
    "Nott'm Forest": "Nottingham Forest",
    "Nottm Forest": "Nottingham Forest",
    "Wolves": "Wolverhampton Wanderers",
    "Spurs": "Tottenham Hotspur",
    "Tottenham": "Tottenham Hotspur",
    "West Brom": "West Bromwich Albion",
    "West Ham": "West Ham United",
    "Leicester": "Leicester City",
    "Norwich": "Norwich City",
    "Cardiff": "Cardiff City",
    "Stoke": "Stoke City",
    "Swansea": "Swansea City",
    "Hull": "Hull City",
    "QPR": "Queens Park Rangers",
    "Leeds": "Leeds United",
    "Luton": "Luton Town",
    "Sheffield Utd": "Sheffield United",
    "Brighton": "Brighton & Hove Albion",
}


def canonicalize_team_names(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize HomeTeam/AwayTeam name variants to a single canonical spelling."""
    df = df.copy()
    for col in ("HomeTeam", "AwayTeam"):
        if col in df.columns:
            df[col] = df[col].replace(TEAM_NAME_MAP)
    return df


def validate_dataset(df: pd.DataFrame) -> list:
    """
    Check the dataset for common data-quality problems.

    Returns a list of issue strings (empty list = clean). Also prints a summary.
    """
    issues = []

    dup_mask = df.duplicated(subset=["Date", "HomeTeam", "AwayTeam"], keep=False)
    if dup_mask.any():
        issues.append(f"{dup_mask.sum()} duplicate (Date, HomeTeam, AwayTeam) rows found")

    required = ["Date", "HomeTeam", "AwayTeam", "HomeGoals", "AwayGoals", "Result"]
    for col in required:
        if col not in df.columns:
            issues.append(f"missing required column: {col}")
        elif df[col].isna().any():
            issues.append(f"{df[col].isna().sum()} missing values in required column '{col}'")

    if "HomeGoals" in df.columns and (df["HomeGoals"] < 0).any():
        issues.append("negative HomeGoals values found")
    if "AwayGoals" in df.columns and (df["AwayGoals"] < 0).any():
        issues.append("negative AwayGoals values found")

    if "Date" in df.columns:
        today = pd.Timestamp.now().normalize()
        future = df["Date"] > today
        if future.any():
            issues.append(f"{future.sum()} rows have a Date in the future ({today.date()})")

        if "Season" in df.columns:
            for season, group in df.groupby("Season"):
                sorted_dates = group["Date"].sort_values()
                if not sorted_dates.is_monotonic_increasing and not sorted_dates.equals(sorted_dates):
                    pass  # order within season isn't required to be sorted in the raw file itself

    if "HomeTeam" in df.columns and "AwayTeam" in df.columns:
        all_names = set(df["HomeTeam"].unique()) | set(df["AwayTeam"].unique())
        suspicious = {n for n in all_names if n in TEAM_NAME_MAP}
        if suspicious:
            issues.append(
                f"{len(suspicious)} team name variant(s) found that should be canonicalized: "
                f"{sorted(suspicious)}"
            )

    if issues:
        print("Data validation found issues:")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print("Data validation passed: no issues found.")

    return issues


def report_data_quality(df: pd.DataFrame) -> None:
    """Print a summary of the dataset's size, coverage, and missingness."""
    print("=" * 60)
    print("DATA QUALITY REPORT")
    print("=" * 60)
    print(f"Matches:        {len(df)}")
    if "Season" in df.columns:
        print(f"Seasons:        {df['Season'].nunique()} ({df['Season'].min()}-{df['Season'].max()})")
    if "HomeTeam" in df.columns:
        teams = set(df["HomeTeam"].unique()) | set(df["AwayTeam"].unique())
        print(f"Teams:          {len(teams)}")
    if "Date" in df.columns:
        print(f"Date range:     {df['Date'].min().date()} to {df['Date'].max().date()}")

    print("\nMissing values per column:")
    missing = df.isna().sum()
    missing = missing[missing > 0]
    if missing.empty:
        print("  none")
    else:
        for col, count in missing.items():
            print(f"  {col}: {count}")

    if "Season" in df.columns:
        print("\nMatches per season:")
        print(df.groupby("Season").size().to_string())
    print("=" * 60)
