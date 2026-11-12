"""Phase 0a: download + clean MoneyPuck shot data into a tidy table.

MoneyPuck ships one 137-column CSV per season. We keep the handful of columns
relevant to the goalie-pull problem and derive a few game-state fields.

Key subtlety baked in here: an empty net is NOT always a goalie pull. Teams also
pull the goalie for a 6th attacker on a *delayed penalty* (can happen any time,
score often tied). We separate genuine late trailing pulls (`is_trailing_pull`)
from all empty-net situations so the rate model isn't polluted by delayed-penalty
events. Reconstructing the continuous empty-net *intervals* (exposure time, the
denominator for Poisson rates) is Phase 0b and lives in exposure.py.

CLI:
    python -m goalie_pull.ingest --season 2024
    python -m goalie_pull.ingest --all            # DEFAULT_SEASONS
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd
import requests

from . import config

# Columns the rate model truly needs: absence is a hard error.
REQUIRED_COLS = [
    "season", "game_id",
    "period", "time",
    "event", "goal", "xGoal",
    "homeTeamCode", "awayTeamCode", "teamCode", "isHomeTeam",
    "homeSkatersOnIce", "awaySkatersOnIce",
    "homeEmptyNet", "awayEmptyNet",
    "homeTeamGoals", "awayTeamGoals",   # score BEFORE this shot
]
# Nice-to-have columns; older MoneyPuck files omit some. Filled if missing.
OPTIONAL_COLS = [
    "isPlayoffGame", "shotOnEmptyNet", "timeLeft",
    "wentToOT", "wentToShootout", "homeTeamWon",
]
USECOLS = REQUIRED_COLS + OPTIONAL_COLS

# How late (seconds remaining in regulation) a pull must be to count as a
# "late trailing pull" rather than an anomaly. 5 minutes is generous; real pulls
# cluster in the last ~3 minutes.
LATE_PULL_WINDOW_S = 300


def download_season(season: int, *, force: bool = False) -> "config.Path":
    """Download and cache one season's shots zip. Returns the local path."""
    dest = config.raw_zip_path(season)
    if dest.exists() and not force and dest.stat().st_size > 0:
        return dest
    url = config.shots_zip_url(season)
    print(f"  downloading {url}", file=sys.stderr)
    with requests.get(url, stream=True, timeout=180) as r:
        r.raise_for_status()
        tmp = dest.with_suffix(".zip.part")
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)
        tmp.replace(dest)
    return dest


def load_raw_season(season: int, *, force_download: bool = False) -> pd.DataFrame:
    """Load one season's shots CSV (from the cached zip) with just USECOLS."""
    zip_path = download_season(season, force=force_download)
    # pandas reads the single CSV inside the zip directly.
    df = pd.read_csv(
        zip_path,
        compression="zip",
        usecols=lambda c: c in USECOLS,
        low_memory=False,
    )
    missing_required = set(REQUIRED_COLS) - set(df.columns)
    if missing_required:
        raise ValueError(
            f"season {season}: required columns missing from MoneyPuck file: "
            f"{sorted(missing_required)}. The source schema may have changed."
        )
    # Backfill any missing optional columns so clean() can rely on them.
    for col in OPTIONAL_COLS:
        if col not in df.columns:
            df[col] = 0
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Standardise names/types and derive game-state + pull-context fields."""
    out = pd.DataFrame()
    out["season"] = df["season"].astype("int16")
    out["game_id"] = df["game_id"].astype("int32")
    # Unique game key across seasons.
    out["game_uid"] = out["season"].astype("int64") * 1_000_000 + out["game_id"]
    out["is_playoff"] = df["isPlayoffGame"].astype("int8").astype(bool)

    out["period"] = df["period"].astype("int8")
    out["time"] = df["time"].astype("int32")  # seconds since game start
    # Seconds remaining in regulation (meaningful for period<=3; 0 implies OT).
    out["sec_remaining_reg"] = (config.REGULATION_SECONDS - out["time"]).clip(lower=0)

    out["event"] = df["event"].astype("category")   # SHOT / MISS / GOAL / (BLOCK)
    out["is_goal"] = df["goal"].fillna(0).astype("int8")
    out["xg"] = df["xGoal"].astype("float32")
    out["shot_on_empty_net"] = df["shotOnEmptyNet"].fillna(0).astype("int8")

    out["home_team"] = df["homeTeamCode"].astype("category")
    out["away_team"] = df["awayTeamCode"].astype("category")
    out["shooting_team"] = df["teamCode"].astype("category")
    out["shooting_is_home"] = (df["isHomeTeam"].astype("float32") > 0.5)

    out["home_skaters"] = df["homeSkatersOnIce"].astype("int8")
    out["away_skaters"] = df["awaySkatersOnIce"].astype("int8")
    home_en = df["homeEmptyNet"].fillna(0).astype("int8")
    away_en = df["awayEmptyNet"].fillna(0).astype("int8")

    # Score BEFORE the shot.
    hg = df["homeTeamGoals"].astype("int16")
    ag = df["awayTeamGoals"].astype("int16")
    out["home_goals_before"] = hg
    out["away_goals_before"] = ag

    # --- Pull context ---------------------------------------------------------
    # Which team (if any) has the empty net == the team that pulled its goalie.
    pulling = np.where(home_en == 1, "home",
              np.where(away_en == 1, "away", "none"))
    out["empty_net_team"] = pd.Categorical(pulling, categories=["home", "away", "none"])
    out["is_empty_net_situation"] = pulling != "none"

    # Score differential FROM THE PULLING TEAM'S perspective (their goals - opp).
    pull_diff = np.where(
        pulling == "home", hg - ag,
        np.where(pulling == "away", ag - hg, np.nan),
    ).astype("float32")
    out["pulling_team_score_diff"] = pull_diff  # negative == trailing

    # Genuine late trailing pull: net empty, puller trailing, regulation, late.
    out["is_trailing_pull"] = (
        out["is_empty_net_situation"]
        & (pull_diff < 0)
        & (out["period"] <= 3)
        & (out["sec_remaining_reg"] <= LATE_PULL_WINDOW_S)
    )

    # For shots during an empty-net situation, is the shooter the pulling (6-man)
    # team? If so it's a goal-FOR chance (trying to tie); otherwise it's a shot
    # AT the empty net (goal-AGAINST chance).
    shooter_is_puller = (
        (pulling == "home") & out["shooting_is_home"].to_numpy()
    ) | (
        (pulling == "away") & (~out["shooting_is_home"].to_numpy())
    )
    out["shot_by_pulling_team"] = np.where(
        out["is_empty_net_situation"], shooter_is_puller, np.nan
    )

    out["went_to_ot"] = df["wentToOT"].fillna(0).astype("int8").astype(bool)
    out["went_to_so"] = df["wentToShootout"].fillna(0).astype("int8").astype(bool)
    return out


def build_season(season: int, *, force_download: bool = False) -> pd.DataFrame:
    raw = load_raw_season(season, force_download=force_download)
    return clean(raw)


def build(seasons: list[int] | None = None, *, force_download: bool = False) -> pd.DataFrame:
    seasons = seasons or config.DEFAULT_SEASONS
    frames = []
    for s in seasons:
        print(f"[ingest] season {s} ...", file=sys.stderr)
        frames.append(build_season(s, force_download=force_download))
    df = pd.concat(frames, ignore_index=True)
    out_path = config.processed_path("shots.parquet")
    df.to_parquet(out_path, index=False)
    print(f"[ingest] wrote {len(df):,} shots -> {out_path}", file=sys.stderr)
    return df


def _summary(df: pd.DataFrame) -> str:
    en = df[df["is_empty_net_situation"]]
    pulls = df[df["is_trailing_pull"]]
    lines = [
        f"total shots:                  {len(df):,}",
        f"seasons:                      {sorted(df['season'].unique().tolist())}",
        f"empty-net-situation shots:    {len(en):,}",
        f"  of which late trailing pull:{len(pulls):,}",
        f"  (remainder = delayed-penalty / non-trailing / OT empty nets)",
        f"goals during trailing pulls:  {int(pulls['is_goal'].sum()):,}",
        f"  by pulling (6-man) team:    {int(pulls.loc[pulls['shot_by_pulling_team']==1, 'is_goal'].sum()):,}",
        f"  into empty net (against):   {int(pulls.loc[pulls['shot_by_pulling_team']==0, 'is_goal'].sum()):,}",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Ingest MoneyPuck shot data.")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--season", type=int, help="single season (starting year)")
    g.add_argument("--all", action="store_true", help="all DEFAULT_SEASONS")
    p.add_argument("--force-download", action="store_true")
    args = p.parse_args(argv)

    seasons = config.DEFAULT_SEASONS if args.all else [args.season]
    df = build(seasons, force_download=args.force_download)
    print("\n=== ingest summary ===")
    print(_summary(df))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
