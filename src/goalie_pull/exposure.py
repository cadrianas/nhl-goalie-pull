"""Phase 0b: reconstruct empty-net *intervals* from NHL play-by-play.

Shot data gives events (the numerator for goal rates) but not continuous time.
To get the exposure denominator -- how many seconds a trailing team actually
spent with its goalie pulled -- we walk the NHL play-by-play, where every play
carries a `situationCode` encoding the on-ice state.

situationCode is a 4-char string: [awayGoalie][awaySkaters][homeSkaters][homeGoalie]
    "1551" -> both goalies in, 5v5
    "1560" -> home goalie pulled (0), 6 home skaters
    "0651" -> away goalie pulled (0), 6 away skaters

Between consecutive plays the state is constant, so each adjacent pair yields a
timed interval. We tag each interval with the pulled team, the running score, and
time remaining, then aggregate seconds of exposure per game-state.

We only fetch play-by-play for games that the shots table already flags as having
a trailing pull (`is_trailing_pull`), which is a few hundred games/season rather
than all ~1,300. (Caveat: a pull that generated zero shots won't be in that set;
pass --all-games to fetch every game and capture those too.)

NHL game id = season * 1_000_000 + moneypuck_game_id   (2024 + 20001 -> 2024020001)

CLI:
    python -m goalie_pull.exposure --season 2024 --limit 25   # smoke test
    python -m goalie_pull.exposure --season 2024              # pull-games only
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd
import requests

from . import config

PBP_URL = "https://api-web.nhle.com/v1/gamecenter/{game_id}/play-by-play"
PBP_CACHE = config.RAW_DIR / "pbp"
PBP_CACHE.mkdir(parents=True, exist_ok=True)

REQUEST_DELAY_S = 0.3  # be polite to the NHL API


# --------------------------------------------------------------------------- #
# Fetching
# --------------------------------------------------------------------------- #
def nhl_game_id(season: int, mp_game_id: int) -> int:
    """MoneyPuck (season, game_id) -> full NHL game id."""
    return season * 1_000_000 + int(mp_game_id)


def fetch_pbp(game_id: int, *, force: bool = False) -> dict:
    """Fetch + cache one game's play-by-play JSON."""
    cache = PBP_CACHE / f"{game_id}.json"
    if cache.exists() and not force and cache.stat().st_size > 0:
        return json.loads(cache.read_text())
    url = PBP_URL.format(game_id=game_id)
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    data = r.json()
    cache.write_text(json.dumps(data))
    time.sleep(REQUEST_DELAY_S)
    return data


# --------------------------------------------------------------------------- #
# Parsing
# --------------------------------------------------------------------------- #
def parse_situation(sc: str | None) -> tuple[int, int, int, int] | None:
    """Return (away_goalie, away_skaters, home_skaters, home_goalie) or None."""
    if not sc or len(str(sc)) != 4 or not str(sc).isdigit():
        return None
    sc = str(sc)
    return int(sc[0]), int(sc[1]), int(sc[2]), int(sc[3])


def _mmss_to_sec(mmss: str) -> int:
    m, s = mmss.split(":")
    return int(m) * 60 + int(s)


def game_time_seconds(period: int, period_type: str, time_in_period: str) -> int:
    """Absolute seconds since the opening faceoff."""
    sec = _mmss_to_sec(time_in_period)
    # Regulation periods are 1200s; OT periods continue the clock.
    return (period - 1) * 1200 + sec


def reconstruct_intervals(pbp: dict) -> pd.DataFrame:
    """Walk one game's plays into timed, state-tagged intervals.

    Each row is the span from one play to the next within the same period, with
    the on-ice/score state that held during that span.
    """
    home_id = pbp["homeTeam"]["id"]
    away_id = pbp["awayTeam"]["id"]
    plays = pbp.get("plays", [])

    rows = []
    home_score = 0
    away_score = 0
    for play in plays:
        pd_ = play.get("periodDescriptor", {})
        period = pd_.get("number")
        ptype = pd_.get("periodType", "REG")
        tip = play.get("timeInPeriod")
        sc = parse_situation(play.get("situationCode"))
        if period is None or tip is None:
            continue
        gt = game_time_seconds(period, ptype, tip)
        # If this play is a goal, which side scored (state is still pre-goal).
        scoring_team = "none"
        if play.get("typeDescKey") == "goal":
            owner = play.get("details", {}).get("eventOwnerTeamId")
            if owner == home_id:
                scoring_team = "home"
            elif owner == away_id:
                scoring_team = "away"
        rows.append(
            {
                "period": period,
                "period_type": ptype,
                "game_time": gt,
                "sit": sc,
                "home_score": home_score,
                "away_score": away_score,
                "type": play.get("typeDescKey"),
                "scoring_team": scoring_team,
            }
        )
        # Update running score AFTER recording the pre-goal state.
        if scoring_team == "home":
            home_score += 1
        elif scoring_team == "away":
            away_score += 1

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    # Interval = gap to the next play within the same period.
    df = df.sort_values(["period", "game_time"]).reset_index(drop=True)
    df["next_game_time"] = df["game_time"].shift(-1)
    df["next_period"] = df["period"].shift(-1)
    same_period = df["next_period"] == df["period"]
    df["duration"] = (df["next_game_time"] - df["game_time"]).where(same_period, 0.0)
    df["duration"] = df["duration"].fillna(0.0).clip(lower=0.0)

    # Decode situation into pull context.
    def _pull(row):
        sc = row["sit"]
        if sc is None:
            return pd.Series({"empty_net_team": "none", "skaters_for": pd.NA,
                              "skaters_against": pd.NA})
        ag, ask, hsk, hg = sc
        if hg == 0 and ag != 0:
            return pd.Series({"empty_net_team": "home", "skaters_for": hsk,
                              "skaters_against": ask})
        if ag == 0 and hg != 0:
            return pd.Series({"empty_net_team": "away", "skaters_for": ask,
                              "skaters_against": hsk})
        return pd.Series({"empty_net_team": "none", "skaters_for": pd.NA,
                          "skaters_against": pd.NA})

    df = pd.concat([df, df.apply(_pull, axis=1)], axis=1)

    # Score diff from the pulling team's perspective (negative == trailing).
    def _diff(row):
        if row["empty_net_team"] == "home":
            return row["home_score"] - row["away_score"]
        if row["empty_net_team"] == "away":
            return row["away_score"] - row["home_score"]
        return pd.NA

    df["pulling_team_score_diff"] = df.apply(_diff, axis=1)
    df["sec_remaining_reg"] = (config.REGULATION_SECONDS - df["game_time"]).clip(lower=0)
    df["is_pulled"] = df["empty_net_team"] != "none"
    df["is_trailing_pull"] = (
        df["is_pulled"]
        & (df["pulling_team_score_diff"].fillna(0) < 0)
        & (df["period"] <= 3)
    )
    return df


def exposure_for_game(game_id: int, *, force: bool = False) -> pd.DataFrame:
    pbp = fetch_pbp(game_id, force=force)
    intervals = reconstruct_intervals(pbp)
    if intervals.empty:
        return intervals
    intervals.insert(0, "game_id", game_id)
    return intervals


# --------------------------------------------------------------------------- #
# Build
# --------------------------------------------------------------------------- #
def pull_game_ids_from_shots(season: int) -> list[int]:
    """NHL game ids for games the shots table flags as having a trailing pull."""
    shots = pd.read_parquet(config.processed_path("shots.parquet"))
    sel = shots[(shots["season"] == season) & (shots["is_trailing_pull"])]
    mp_ids = sorted(sel["game_id"].unique().tolist())
    return [nhl_game_id(season, g) for g in mp_ids]


def build_exposure(season: int, *, limit: int | None = None,
                   force: bool = False) -> pd.DataFrame:
    game_ids = pull_game_ids_from_shots(season)
    if limit:
        game_ids = game_ids[:limit]
    print(f"[exposure] season {season}: {len(game_ids)} pull-games to fetch",
          file=sys.stderr)
    frames = []
    for i, gid in enumerate(game_ids, 1):
        try:
            frames.append(exposure_for_game(gid, force=force))
        except requests.HTTPError as e:
            print(f"  WARN {gid}: {e}", file=sys.stderr)
        if i % 50 == 0:
            print(f"  ... {i}/{len(game_ids)}", file=sys.stderr)
    intervals = pd.concat(frames, ignore_index=True)
    out = config.processed_path(f"intervals_{season}.parquet")
    intervals.to_parquet(out, index=False)

    # Aggregate: seconds of trailing-pull exposure by score diff and minute.
    tp = intervals[intervals["is_trailing_pull"]].copy()
    tp["min_remaining"] = (tp["sec_remaining_reg"] // 60).astype(int)
    expo = (
        tp.groupby(["pulling_team_score_diff", "min_remaining"])["duration"]
        .sum().reset_index().rename(columns={"duration": "exposure_sec"})
    )
    expo_out = config.processed_path(f"exposure_{season}.parquet")
    expo.to_parquet(expo_out, index=False)
    print(f"[exposure] wrote intervals -> {out}", file=sys.stderr)
    print(f"[exposure] wrote exposure table -> {expo_out}", file=sys.stderr)

    total_pull_sec = tp["duration"].sum()
    print("\n=== exposure summary ===")
    print(f"games processed:            {intervals['game_id'].nunique()}")
    print(f"trailing-pull exposure:     {total_pull_sec/60:,.1f} minutes "
          f"({total_pull_sec:,.0f} s)")
    print(f"avg pull time per game:     "
          f"{total_pull_sec/max(intervals['game_id'].nunique(),1):,.1f} s")
    by_diff = tp.groupby("pulling_team_score_diff")["duration"].sum() / 60
    print("minutes of exposure by score diff (pulling team):")
    for d, m in by_diff.items():
        print(f"  down {int(-d)}: {m:,.1f} min")
    return intervals


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Reconstruct empty-net exposure from NHL pbp.")
    p.add_argument("--season", type=int, required=True)
    p.add_argument("--limit", type=int, default=None,
                   help="only fetch the first N pull-games (smoke test)")
    p.add_argument("--force", action="store_true")
    args = p.parse_args(argv)
    build_exposure(args.season, limit=args.limit, force=args.force)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
