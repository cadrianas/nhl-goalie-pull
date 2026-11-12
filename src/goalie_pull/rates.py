"""Phase 1: conjugate Gamma-Poisson posteriors for goal rates.

For the pull decision we need, per deficit d (team trailing by d goals) in the
late game, four per-second goal rates:

    lam_for_out, lam_against_out   goalie PULLED  (6 skaters, empty net)
    lam_for_in,  lam_against_in    goalie IN      (normal, team pressing)

"for"     = the trailing team scores (good; narrows/erases the deficit)
"against" = the opponent scores     (bad; empty-netter effectively ends it)

Each rate gets a Gamma-Poisson model: with a Gamma(a0, b0) prior (b0 in seconds)
and `goals` events over `exposure` seconds, the posterior is
    Gamma(a0 + goals, b0 + exposure).
This is analytic -- no MCMC needed for the baseline. The hierarchical PyMC model
(Phase 3) will expose the same `.draws(d)` interface so decision.py is unchanged.

Numerator (goals) and denominator (exposure seconds) both come from the NHL
play-by-play intervals (exposure.py), so they are mutually consistent.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import config

LATE_WINDOW_S = 300          # last 5 minutes of regulation
MAX_DEFICIT = 3              # model down 1/2/3; deeper is treated as ~lost

# Weakly-informative priors, as (prior_goals, prior_seconds). Equivalent to
# having seen `prior_goals` goals in `prior_seconds` of play -- a few events of
# weight, so hundreds of real events dominate. Rough league intuitions:
#   pulled:  for ~0.6/min, against ~1.6/min   in: for ~0.4/min, against ~0.4/min
PRIORS = {
    ("out", "for"): (2.0, 2.0 / (0.6 / 60)),
    ("out", "against"): (2.0, 2.0 / (1.6 / 60)),
    ("in", "for"): (2.0, 2.0 / (0.4 / 60)),
    ("in", "against"): (2.0, 2.0 / (0.4 / 60)),
}


@dataclass(frozen=True)
class GammaPosterior:
    """Posterior Gamma(alpha, beta) over a per-second rate. beta in seconds."""
    alpha: float
    beta: float
    goals: int
    exposure: float

    @property
    def mean(self) -> float:
        return self.alpha / self.beta

    def draws(self, n: int, rng: np.random.Generator) -> np.ndarray:
        # numpy Gamma uses shape/scale; scale = 1/rate_param = 1/beta.
        return rng.gamma(self.alpha, 1.0 / self.beta, size=n)

    def interval(self, q=(0.025, 0.975)) -> tuple[float, float]:
        from scipy.stats import gamma
        lo, hi = gamma.ppf(q, self.alpha, scale=1.0 / self.beta)
        return float(lo), float(hi)


# --------------------------------------------------------------------------- #
# Build the (deficit -> counts) tables from the play-by-play intervals.
# --------------------------------------------------------------------------- #
def load_intervals(seasons: list[int]) -> pd.DataFrame:
    frames = []
    for s in seasons:
        p = config.processed_path(f"intervals_{s}.parquet")
        if p.exists():
            frames.append(pd.read_parquet(p))
    if not frames:
        raise FileNotFoundError(
            f"no intervals_*.parquet for {seasons}; run exposure.py first."
        )
    return pd.concat(frames, ignore_index=True)


def tabulate(intervals: pd.DataFrame) -> pd.DataFrame:
    """Return a tidy table: one row per (goalie_state, deficit) with
    exposure_sec, goals_for, goals_against."""
    df = intervals
    is_goal = df["type"] == "goal"

    rows = []

    # ---- PULLED (goalie out): the trailing-pull intervals ------------------
    pulled = df[df["is_trailing_pull"]].copy()
    pulled["deficit"] = (-pulled["pulling_team_score_diff"]).clip(upper=MAX_DEFICIT).astype(int)
    for d, g in pulled.groupby("deficit"):
        goal_rows = g[g["type"] == "goal"]
        # "for" = pulling (empty-net) team scored; "against" = opponent scored.
        gf = int((goal_rows["scoring_team"] == goal_rows["empty_net_team"]).sum())
        ga = int(((goal_rows["scoring_team"] != goal_rows["empty_net_team"])
                  & (goal_rows["scoring_team"] != "none")).sum())
        rows.append({"state": "out", "deficit": int(d),
                     "exposure_sec": float(g["duration"].sum()),
                     "goals_for": gf, "goals_against": ga})

    # ---- GOALIE IN: trailing, net NOT empty, 3rd period --------------------
    # NB: estimated over the WHOLE 3rd period, not just the last 5 min. The
    # last-5-min goalie-in scoring rate is biased LOW by selection -- teams that
    # would score late increasingly do so by pulling, removing them from this
    # (non-pulled) sample. Period 3 is a less-censored counterfactual for "what
    # if the goalie stays in." The strong time-dependence here (for-rate falls
    # ~6x from early to late regulation) is exactly what the Phase 3
    # time-inhomogeneous model is meant to capture.
    late_in = df[
        (df["period"] == 3)
        & (df["empty_net_team"] == "none")
        & (df["period_type"] == "REG")
    ].copy()
    # Determine the trailing team and deficit for each interval.
    diff = late_in["home_score"] - late_in["away_score"]
    trailing = np.where(diff < 0, "home", np.where(diff > 0, "away", "none"))
    late_in["trailing_team"] = trailing
    late_in["deficit"] = np.abs(diff).clip(upper=MAX_DEFICIT).astype(int)
    late_in = late_in[late_in["trailing_team"] != "none"]
    for d, g in late_in.groupby("deficit"):
        if d == 0:
            continue
        goal_rows = g[g["type"] == "goal"]
        gf = int((goal_rows["scoring_team"] == goal_rows["trailing_team"]).sum())
        ga = int(((goal_rows["scoring_team"] != goal_rows["trailing_team"])
                  & (goal_rows["scoring_team"] != "none")).sum())
        rows.append({"state": "in", "deficit": int(d),
                     "exposure_sec": float(g["duration"].sum()),
                     "goals_for": gf, "goals_against": ga})

    return pd.DataFrame(rows).sort_values(["state", "deficit"]).reset_index(drop=True)


def fit(table: pd.DataFrame) -> dict[tuple[str, int, str], GammaPosterior]:
    """Fit a Gamma posterior for each (state, deficit, for/against) cell."""
    post: dict[tuple[str, int, str], GammaPosterior] = {}
    for _, r in table.iterrows():
        state, d, expo = r["state"], int(r["deficit"]), float(r["exposure_sec"])
        for direction, goals in (("for", int(r["goals_for"])),
                                  ("against", int(r["goals_against"]))):
            a0, b0 = PRIORS[(state, direction)]
            post[(state, d, direction)] = GammaPosterior(
                alpha=a0 + goals, beta=b0 + expo, goals=goals, exposure=expo)
    return post


def build(seasons: list[int]) -> tuple[pd.DataFrame, dict]:
    intervals = load_intervals(seasons)
    table = tabulate(intervals)
    return table, fit(table)


def _fmt(post: dict) -> str:
    lines = ["state deficit  dir       goals   exp(min)   rate/60s   95% CI/60s"]
    for (state, d, direction), p in sorted(post.items()):
        lo, hi = p.interval()
        lines.append(
            f"{state:>4}  {d:>5}  {direction:>7}  {p.goals:>6}  {p.exposure/60:>8.1f}"
            f"   {p.mean*60:>7.3f}   [{lo*60:.3f}, {hi*60:.3f}]")
    return "\n".join(lines)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--seasons", type=int, nargs="+", default=[2022, 2023, 2024])
    args = ap.parse_args()
    table, post = build(args.seasons)
    print(table.to_string(index=False))
    print()
    print(_fmt(post))
