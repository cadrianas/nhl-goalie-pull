"""Phase 3b: fit time-inhomogeneous goalie-in hazards and measure how modeling
the hazard shape changes the optimal pull time vs the constant-rate baseline.

    python scripts/run_timevarying.py
"""

from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import arviz as az

from goalie_pull import config, decision, rates as rates_mod, timevarying as TV

HORIZON = decision.HORIZON_S
N_DRAWS = 4000


def main(seasons=(2022, 2023, 2024)):
    seasons = list(seasons)
    _, post = rates_mod.build(seasons)
    tv = TV.build(seasons, draws=1500, tune=1500, chains=4, seed=3)
    s = az.summary(tv.idata)
    print(f"TV model convergence: max r_hat={s['r_hat'].max():.3f} "
          f"min ess_bulk={int(s['ess_bulk'].min())}")

    # Build time-varying goalie-in rate arrays for the solver.
    rng = np.random.default_rng(0)
    tv_in = {d: (tv.rate_tod("for", d, N_DRAWS, rng, HORIZON),
                 tv.rate_tod("against", d, N_DRAWS, rng, HORIZON))
             for d in (1, 2, 3)}

    base = decision.solve(post, utility="points", n_draws=N_DRAWS)
    tvar = decision.solve(post, utility="points", n_draws=N_DRAWS, tv_in=tv_in)

    print("\n=== optimal pull time: constant vs time-varying goalie-in hazards ===")
    print("deficit   constant-rate      time-varying       shift")
    for d in (1, 2, 3):
        b, t = base[d].summary(), tvar[d].summary()
        bl, bh = b["pull_90ci_s"]; tl, th = t["pull_90ci_s"]
        print(f"  down {d}    {b['pull_median_s']:>4.0f}s ({bl:.0f}-{bh:.0f})   "
              f"{t['pull_median_s']:>4.0f}s ({tl:.0f}-{th:.0f})   "
              f"{t['pull_median_s']-b['pull_median_s']:+.0f}s")

    _plot_hazard(tv, post)
    _write_md(tv, base, tvar, s, seasons)
    return tv, base, tvar


def _plot_hazard(tv, post):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as mpl

    fig, ax = mpl.subplots(figsize=(9, 5))
    colors = {1: "#1b6ca8", 2: "#edae49", 3: "#d1495b"}
    mins = (np.arange(TV.N_BINS) * TV.BIN_S + TV.BIN_S / 2) / 60
    for d in (1, 2, 3):
        lam = tv.lam_bins("for", d) * 60
        med = np.median(lam, axis=0)
        lo, hi = np.percentile(lam, [5, 95], axis=0)
        ax.plot(mins, med, color=colors[d], label=f"down {d}")
        ax.fill_between(mins, lo, hi, color=colors[d], alpha=0.18)
    # constant baseline (period-3 average) for down 1, for reference.
    base_const = post[("in", 1, "for")].mean * 60
    ax.axhline(base_const, color="#444", ls="--", lw=1,
               label=f"constant baseline (down 1) {base_const:.3f}")
    ax.set_xlabel("Minutes remaining in regulation")
    ax.set_ylabel("Goalie-in scoring rate for the trailing team (goals / 60s)")
    ax.set_title("The goalie-in scoring rate collapses as regulation ends")
    ax.invert_xaxis()
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out = config.FIGURES_DIR / "goalie_in_hazard.png"
    fig.savefig(out, dpi=130)
    print(f"\n[plot] wrote {out}")


def _write_md(tv, base, tvar, summ, seasons):
    lines = [
        "# Phase 3b: time-inhomogeneous goalie-in hazards\n",
        f"Seasons {seasons}. PyMC: log lam[d,t] = level[d] + shape(t), shape(t) a "
        "Gaussian random walk over 60s bins (pooled across deficits).\n",
        f"Convergence: max r-hat **{summ['r_hat'].max():.3f}**, "
        f"min ESS **{int(summ['ess_bulk'].min())}**.\n",
        "## Optimal pull time: constant vs. time-varying goalie-in hazards\n",
        "| deficit | constant-rate | time-varying | shift |",
        "|---|---|---|---|",
    ]
    for d in (1, 2, 3):
        b, t = base[d].summary(), tvar[d].summary()
        lines.append(f"| down {d} | {b['pull_median_s']:.0f}s | {t['pull_median_s']:.0f}s | "
                     f"{t['pull_median_s']-b['pull_median_s']:+.0f}s |")
    lines += [
        "",
        "The trailing team's goalie-in scoring rate falls steadily through the last "
        "~15 minutes (figure `goalie_in_hazard.png`), so the constant period-3 "
        "average over-prices the offense available late. Modeling the hazard shape "
        "shifts the optimal pull time as shown -- the methodological point being "
        "that the pull decision is a *non-homogeneous* optimal-stopping problem, "
        "which the constant-rate literature does not capture.",
        "",
        "Selection caveat: the final ~90 s of goalie-in data thins as teams pull, "
        "so the observed rate there is biased low; the decision mostly turns on the "
        "earlier part of the window where most teams still have the goalie in.",
    ]
    (config.REPORTS_DIR / "timevarying.md").write_text("\n".join(lines))
    print(f"[md] wrote {config.REPORTS_DIR / 'timevarying.md'}")


if __name__ == "__main__":
    main()
