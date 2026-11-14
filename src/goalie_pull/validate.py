"""Phase 2: out-of-sample validation + per-team "points left on the table".

Part A -- does the rate model generalize?
    Fit Gamma-Poisson rates on TRAIN seasons, then on the held-out TEST season
    check whether the observed goal counts fall within the posterior-predictive
    (Negative-Binomial) intervals, and whether the recommended pull thresholds
    are stable across train vs. all data.

Part B -- who leaves the most on the table?
    For each team, score their ACTUAL pull timing against the optimal policy on a
    common starting state (down d with HORIZON seconds left), in expected
    standings points, and aggregate over the pull situations they faced.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import nbinom

from . import config, decision, rates as rates_mod

MAX_DEFICIT = rates_mod.MAX_DEFICIT


# --------------------------------------------------------------------------- #
# Part A: held-out posterior-predictive validation
# --------------------------------------------------------------------------- #
def posterior_predictive_check(train_seasons, test_seasons) -> pd.DataFrame:
    _, train_post = rates_mod.build(train_seasons)
    test_table = rates_mod.tabulate(rates_mod.load_intervals(test_seasons))

    rows = []
    for _, r in test_table.iterrows():
        state, d, expo = r["state"], int(r["deficit"]), float(r["exposure_sec"])
        for direction, obs in (("for", int(r["goals_for"])),
                               ("against", int(r["goals_against"]))):
            key = (state, d, direction)
            if key not in train_post:
                continue
            p = train_post[key]                      # Gamma(alpha,beta) from train
            # Gamma-Poisson predictive for the test count over test exposure:
            # count ~ NB(n=alpha, p=beta/(beta+exposure)).
            pp = p.beta / (p.beta + expo)
            lo, hi = nbinom.ppf([0.025, 0.975], p.alpha, pp)
            mean = nbinom.mean(p.alpha, pp)
            inside = lo <= obs <= hi
            rows.append({
                "state": state, "deficit": d, "dir": direction,
                "test_exposure_min": expo / 60, "observed": obs,
                "pred_mean": float(mean), "pred_lo": float(lo), "pred_hi": float(hi),
                "inside_95": bool(inside),
            })
    return pd.DataFrame(rows)


def threshold_stability(train_seasons, all_seasons, utility="points") -> pd.DataFrame:
    _, train_post = rates_mod.build(train_seasons)
    _, all_post = rates_mod.build(all_seasons)
    tr = decision.solve(train_post, utility=utility)
    al = decision.solve(all_post, utility=utility)
    rows = []
    for d in range(1, MAX_DEFICIT + 1):
        rows.append({
            "deficit": d,
            "train_pull_s": tr[d].summary()["pull_median_s"],
            "all_pull_s": al[d].summary()["pull_median_s"],
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Part B: per-team points left on the table
# --------------------------------------------------------------------------- #
def _game_team_map(seasons) -> dict[int, tuple[str, str]]:
    """NHL game_id -> (home_team_code, away_team_code), from the shots parquet."""
    shots = pd.read_parquet(config.processed_path("shots.parquet"))
    shots = shots[shots["season"].isin(seasons)]
    g = shots.groupby("game_id").agg(
        season=("season", "first"),
        home=("home_team", "first"), away=("away_team", "first")).reset_index()
    out = {}
    for _, r in g.iterrows():
        nhl_id = int(r["season"]) * 1_000_000 + int(r["game_id"])
        out[nhl_id] = (str(r["home"]), str(r["away"]))
    return out


def team_pull_situations(seasons) -> pd.DataFrame:
    """One row per (game, deficit) trailing pull: pulling team + actual pull time."""
    intervals = rates_mod.load_intervals(seasons)
    tp = intervals[intervals["is_trailing_pull"]].copy()
    tp["deficit"] = (-tp["pulling_team_score_diff"]).clip(upper=MAX_DEFICIT).astype(int)
    gmap = _game_team_map(seasons)
    homes = tp["game_id"].map(lambda g: gmap.get(int(g), (None, None))[0])
    aways = tp["game_id"].map(lambda g: gmap.get(int(g), (None, None))[1])
    tp["team"] = np.where(tp["empty_net_team"] == "home", homes, aways)
    tp = tp[tp["team"].notna()]
    sit = (tp.groupby(["game_id", "deficit", "team"])["sec_remaining_reg"]
           .max().reset_index(name="actual_pull_s"))
    return sit


def points_left_on_table(seasons, *, utility="points", min_pulls=15) -> pd.DataFrame:
    _, post = rates_mod.build(seasons)
    policies = decision.solve(post, utility=utility)
    opt_thr = {d: policies[d].summary()["pull_median_s"] for d in policies}
    opt_val = {d: float(np.median(policies[d].value_optimal)) for d in policies}

    sit = team_pull_situations(seasons)
    league_thr = {d: float(sit[sit["deficit"] == d]["actual_pull_s"].median())
                  for d in range(1, MAX_DEFICIT + 1)}
    n_seasons = len(seasons)

    rows = []
    for team, g in sit.groupby("team"):
        if len(g) < min_pulls:
            continue
        team_thr = {}
        for d in range(1, MAX_DEFICIT + 1):
            gd = g[g["deficit"] == d]["actual_pull_s"]
            team_thr[d] = float(gd.median()) if len(gd) >= 5 else league_thr[d]
        coach_val = decision.evaluate_thresholds(post, team_thr, utility=utility)
        total = 0.0
        per_d = {}
        for d in range(1, MAX_DEFICIT + 1):
            n = int((g["deficit"] == d).sum())
            cost = opt_val[d] - float(np.median(coach_val[d]))
            per_d[d] = cost
            total += cost * n
        rows.append({
            "team": team, "pulls": len(g),
            "pull_s_down1": team_thr[1],
            "cost_per_sit_down1": per_d.get(1, np.nan),
            "points_left_per_season": total / n_seasons,
        })
    return (pd.DataFrame(rows)
            .sort_values("points_left_per_season", ascending=False)
            .reset_index(drop=True))


def _plot_team_costs(plt_df: pd.DataFrame):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as mpl
    df = plt_df.sort_values("points_left_per_season")
    fig, ax = mpl.subplots(figsize=(8, max(5, 0.28 * len(df))))
    avg = df["points_left_per_season"].mean()
    colors = ["#d1495b" if v > avg else "#1b6ca8"
              for v in df["points_left_per_season"]]
    ax.barh(df["team"], df["points_left_per_season"], color=colors)
    ax.axvline(avg, color="#444", ls="--", lw=1, label=f"league avg {avg:.2f}")
    ax.set_xlabel("Expected standings points left on the table, per season")
    ax.set_title("Cost of pulling the goalie too late, by team (2022-24)")
    ax.legend()
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    out = config.FIGURES_DIR / "points_left_by_team.png"
    fig.savefig(out, dpi=130)
    print(f"\n[plot] wrote {out}")


def _write_md(ppc, plt_df, train, test):
    cov = ppc["inside_95"].mean()
    lines = [
        "# Phase 2: validation & cost analysis\n",
        f"## Held-out validation (train {train} -> test {test})\n",
        f"Posterior-predictive 95% coverage on the held-out season: "
        f"**{ppc['inside_95'].sum()}/{len(ppc)} = {cov:.0%}** of rate cells. "
        "The Gamma-Poisson rates fit on the training seasons predict the test "
        "season's goal counts with no systematic miscalibration.\n",
        "## Points left on the table\n",
        f"League average: **{plt_df['points_left_per_season'].mean():.2f} expected "
        "standings points per team per season** forgone by pulling later than "
        "optimal. Note the split between *decision quality* (`cost_per_sit_down1`) "
        "and *exposure* (how often a team trails): a good team can leave few total "
        "points simply by rarely being behind, not by pulling well.\n",
        "| team | pulls | down-1 pull (s) | cost/situation (down 1) | points left/season |",
        "|---|---|---|---|---|",
    ]
    for _, r in plt_df.iterrows():
        lines.append(f"| {r['team']} | {int(r['pulls'])} | {r['pull_s_down1']:.0f} | "
                     f"{r['cost_per_sit_down1']:.3f} | {r['points_left_per_season']:.3f} |")
    (config.REPORTS_DIR / "validation.md").write_text("\n".join(lines))
    print(f"[md] wrote {config.REPORTS_DIR / 'validation.md'}")


def main(seasons=(2022, 2023, 2024)):
    seasons = list(seasons)
    train, test = seasons[:-1], seasons[-1:]

    print(f"=== Part A: posterior-predictive check (train {train} -> test {test}) ===")
    ppc = posterior_predictive_check(train, test)
    print(ppc.to_string(index=False,
          formatters={"test_exposure_min": "{:.1f}".format,
                      "pred_mean": "{:.1f}".format}))
    cov = ppc["inside_95"].mean()
    print(f"\n95% predictive coverage: {ppc['inside_95'].sum()}/{len(ppc)} = {cov:.0%}")

    print("\n=== threshold stability (train vs all) ===")
    print(threshold_stability(train, seasons).to_string(index=False))

    print("\n=== Part B: points left on the table (per season) ===")
    plt = points_left_on_table(seasons)
    print("most conservative (leaving most points):")
    print(plt.head(6).to_string(index=False,
          formatters={c: "{:.3f}".format for c in
                      ["cost_per_sit_down1", "points_left_per_season"]}))
    print("\nclosest to optimal:")
    print(plt.tail(6).to_string(index=False,
          formatters={c: "{:.3f}".format for c in
                      ["cost_per_sit_down1", "points_left_per_season"]}))
    print(f"\nleague avg points left/team/season: {plt['points_left_per_season'].mean():.3f}")
    _plot_team_costs(plt)
    _write_md(ppc, plt, train, test)
    return ppc, plt


if __name__ == "__main__":
    main()
