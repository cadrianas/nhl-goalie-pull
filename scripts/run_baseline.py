"""Run the Phase 1 baseline end to end and produce the headline artifacts.

    python scripts/run_baseline.py --seasons 2022 2023 2024

Outputs:
  - console tables: fitted rates, optimal pull thresholds, actual-vs-optimal
  - reports/figures/optimal_pull_times.png
  - reports/baseline_results.md
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from goalie_pull import config, decision, rates as rates_mod


def actual_pull_times(seasons: list[int]) -> pd.DataFrame:
    """Observed pull-start time (sec remaining in reg) per game & deficit.

    The pull 'starts' at the earliest trailing-pull interval in a game for a
    given deficit -- i.e. the largest sec_remaining_reg among those intervals.
    """
    intervals = rates_mod.load_intervals(seasons)
    tp = intervals[intervals["is_trailing_pull"]].copy()
    tp["deficit"] = (-tp["pulling_team_score_diff"]).clip(upper=rates_mod.MAX_DEFICIT).astype(int)
    # earliest pull moment per (game, deficit)
    starts = (tp.groupby(["game_id", "deficit"])["sec_remaining_reg"]
              .max().reset_index(name="pull_start_s"))
    return starts


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seasons", type=int, nargs="+", default=[2022, 2023, 2024])
    ap.add_argument("--draws", type=int, default=4000)
    args = ap.parse_args()

    table, post, policies = decision.run(args.seasons, n_draws=args.draws)
    actual = actual_pull_times(args.seasons)

    print("=== fitted rates (per 60s) ===")
    print(rates_mod._fmt(post))
    print("\n=== optimal pull policy (Bayesian, with credible intervals) ===")
    print(decision._fmt(policies))

    # Compare optimal vs actual coach behavior.
    print("\n=== optimal vs actual coach pull time (seconds remaining) ===")
    rows = []
    print("deficit   optimal median (90% CI)     actual median [IQR]      games")
    for d, pol in sorted(policies.items()):
        s = pol.summary()
        a = actual[actual["deficit"] == d]["pull_start_s"]
        amed = float(a.median()) if len(a) else float("nan")
        aq1, aq3 = (float(a.quantile(0.25)), float(a.quantile(0.75))) if len(a) else (np.nan, np.nan)
        lo, hi = s["pull_90ci_s"]
        print(f"  down {d}    {s['pull_median_s']:>4.0f}s ({lo:.0f}-{hi:.0f})"
              f"            {amed:>4.0f}s [{aq1:.0f}-{aq3:.0f}]          {len(a)}")
        rows.append({"deficit": d, "optimal_s": s["pull_median_s"],
                     "optimal_lo": lo, "optimal_hi": hi,
                     "actual_med_s": amed, "actual_q1": aq1, "actual_q3": aq3,
                     "tie_gain": s["tie_prob_gain"], "n_games": int(len(a))})
    results = pd.DataFrame(rows)

    _plot(results, policies, actual)
    _write_md(results, table, post)
    return 0


def _plot(results, policies, actual):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    d = results["deficit"].to_numpy()
    opt = results["optimal_s"].to_numpy()
    lo = results["optimal_lo"].to_numpy()
    hi = results["optimal_hi"].to_numpy()
    act = results["actual_med_s"].to_numpy()

    ax1.errorbar(d - 0.06, opt, yerr=[opt - lo, hi - opt], fmt="o", capsize=5,
                 color="#1b6ca8", label="Model optimal (90% CI)", markersize=8)
    ax1.scatter(d + 0.06, act, color="#d1495b", s=70, marker="s",
                label="Actual coach median", zorder=5)
    ax1.set_xticks(d)
    ax1.set_xticklabels([f"down {int(x)}" for x in d])
    ax1.set_ylabel("Pull time (seconds remaining in regulation)")
    ax1.set_title("Optimal vs. actual goalie-pull time")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.invert_yaxis()  # more time remaining = higher = earlier pull, show at top

    # Right panel: posterior of the optimal threshold for down 1.
    for d_, color in zip((1, 2, 3), ("#1b6ca8", "#edae49", "#d1495b")):
        if d_ in policies:
            ax2.hist(policies[d_].threshold_draws, bins=40, alpha=0.55,
                     color=color, label=f"down {d_}", density=True)
    ax2.set_xlabel("Optimal pull time (seconds remaining)")
    ax2.set_ylabel("Posterior density")
    ax2.set_title("Posterior over the optimal pull time")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    out = config.FIGURES_DIR / "optimal_pull_times.png"
    fig.savefig(out, dpi=130)
    print(f"\n[plot] wrote {out}")


def _write_md(results, table, post):
    lines = ["# Baseline results (Phase 1)\n",
             "Conjugate Gamma-Poisson rates + backward-induction optimal stopping.",
             "Utility = P(reach at least a tie in regulation).\n",
             "## Optimal vs. actual pull time (seconds remaining)\n",
             "| deficit | model optimal (90% CI) | actual coach median | tie-prob gain | games |",
             "|---|---|---|---|---|"]
    for _, r in results.iterrows():
        lines.append(
            f"| down {int(r['deficit'])} | {r['optimal_s']:.0f}s "
            f"({r['optimal_lo']:.0f}-{r['optimal_hi']:.0f}) | "
            f"{r['actual_med_s']:.0f}s | +{r['tie_gain']:.3f} | {int(r['n_games'])} |")
    lines.append("\n## Fitted rates (per 60s)\n```")
    lines.append(rates_mod._fmt(post))
    lines.append("```")
    out = config.REPORTS_DIR / "baseline_results.md"
    Path(out).write_text("\n".join(lines))
    print(f"[md] wrote {out}")


if __name__ == "__main__":
    raise SystemExit(main())
