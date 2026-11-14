"""Phase 4: interactive goalie-pull recommender (Streamlit).

    streamlit run app/goalie_pull_app.py

Given the current game state (deficit + time remaining), recommends whether to
pull the goalie now or wait, and shows the expected-standings-points curve with a
Bayesian credible band. Uses the fast conjugate rates + expected-points
optimal-stopping solver (Phases 1-2); the hierarchical / time-varying refinements
(Phase 3) were shown to move the answer only marginally, so this baseline is a
faithful live recommender that needs no MCMC at startup.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from goalie_pull import decision, rates as rates_mod  # noqa: E402

SEASONS = [2022, 2023, 2024]
MAXD = rates_mod.MAX_DEFICIT
RATE_TABLE = ROOT / "data" / "processed" / "rate_table.csv"


@st.cache_resource(show_spinner="Fitting goal-rate posteriors…")
def load_model(utility: str):
    # Ship a tiny precomputed rate table so the deployed app needs no raw data.
    if RATE_TABLE.exists():
        post = rates_mod.fit(pd.read_csv(RATE_TABLE))
    else:
        _, post = rates_mod.build(SEASONS)
    policies = decision.solve(post, utility=utility, n_draws=4000)
    opt_thr = {d: policies[d].summary()["pull_median_s"] for d in policies}
    opt_ci = {d: policies[d].summary()["pull_90ci_s"] for d in policies}
    return post, opt_thr, opt_ci


@st.cache_data(show_spinner=False)
def value_curve(utility: str, deficit: int):
    """Standardized curve: E[value] from a down-`deficit` state with a full
    window left, as a function of the pull time. Other deficits play optimally."""
    post, opt_thr, _ = load_model(utility)
    H = decision.HORIZON_S
    grid = list(range(0, H + 1, 15))
    med, lo, hi = [], [], []
    for tp in grid:
        thr = {**opt_thr, deficit: tp}
        v = decision.evaluate_thresholds(post, thr, utility=utility,
                                         n_draws=2000, horizon=H)[deficit]
        med.append(float(np.median(v)))
        lo.append(float(np.quantile(v, 0.05)))
        hi.append(float(np.quantile(v, 0.95)))
    return np.array(grid), np.array(med), np.array(lo), np.array(hi)


def mmss(s: int) -> str:
    return f"{s // 60}:{s % 60:02d}"


st.set_page_config(page_title="Goalie Pull Advisor", page_icon="🥅", layout="wide")
st.title("🥅 Bayesian Goalie-Pull Advisor")
st.caption("When should a trailing NHL team pull its goalie? A Bayesian "
           "optimal-stopping model, fit on 2022–24 play-by-play.")

with st.sidebar:
    st.header("Game state")
    deficit = st.radio("Trailing by", [1, 2, 3], format_func=lambda d: f"{d} goal{'s' if d>1 else ''}")
    t_now = st.slider("Time remaining in regulation", 0, 600, 120, step=5,
                      format="%d s")
    st.write(f"**{mmss(t_now)}** remaining")
    utility = st.radio("Objective", ["points", "tie_prob"],
                       format_func=lambda u: "Expected standings points" if u == "points"
                       else "P(reach at least a tie)")
    st.divider()
    st.caption("Model: conjugate Gamma-Poisson goal rates → backward-induction "
               "optimal stopping, full posterior carried through.")

post, opt_thr, opt_ci = load_model(utility)
t_opt = int(opt_thr[deficit])
ci_lo, ci_hi = opt_ci[deficit]

col1, col2 = st.columns([1, 1.4])

with col1:
    st.subheader("Recommendation")
    if t_now <= t_opt:
        st.error(f"### PULL NOW\nYou're at or past the optimal pull point "
                 f"(~{mmss(t_opt)} remaining).")
    else:
        wait = t_now - t_opt
        st.warning(f"### Wait ~{wait}s, then pull\nOptimal pull point: "
                   f"**{mmss(t_opt)}** remaining (90% CI {mmss(int(ci_hi))}–{mmss(int(ci_lo))}).")
    st.metric("Model-optimal pull time", f"{mmss(t_opt)} remaining")
    coach_median = {1: 119, 2: 163, 3: 252}   # actual 2022-24 medians (seconds)
    st.caption(f"For reference, coaches historically pull around "
               f"**{mmss(coach_median[deficit])}** when down {deficit} — far later "
               f"than the model recommends.")

with col2:
    st.subheader("Expected value vs. when you pull")
    grid, med, lo, hi = value_curve(utility, deficit)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.plot(grid, med, color="#1b6ca8", lw=2)
    ax.fill_between(grid, lo, hi, color="#1b6ca8", alpha=0.18, label="90% CI")
    ax.axvline(t_opt, color="#2a9d4a", ls="--", lw=1.5,
               label=f"optimal pull ({mmss(t_opt)})")
    # the user's current time, marked on the curve.
    val_now = float(np.interp(t_now, grid, med))
    ax.scatter([t_now], [val_now], color="#d1495b", zorder=5, s=70,
               label=f"you are here ({mmss(t_now)})")
    ax.set_xlabel("Pull time (seconds remaining)")
    ax.set_ylabel("Expected standings points" if utility == "points" else "P(tie)")
    ax.invert_xaxis()
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)
    st.caption("Curve = expected value from a down-{} state with ~10 min left, by "
               "when you pull. You give up the shaded gap by waiting past the "
               "green line.".format(deficit))

with st.expander("Goal rates for this situation (per 60s, posterior median & 95% CI)"):
    for state_label, state in [("Goalie pulled (6-on-5)", "out"),
                               ("Goalie in", "in")]:
        cols = st.columns(2)
        for i, direction in enumerate(("for", "against")):
            key = (state, deficit, direction)
            if key in post:
                p = post[key]
                clo, chi = p.interval()
                cols[i].metric(f"{state_label} — {direction}",
                               f"{p.mean*60:.3f}",
                               f"95% CI {clo*60:.3f}–{chi*60:.3f}", delta_color="off")

st.divider()
st.caption("Model notes: utility defaults to expected standings points (reg win 2, "
           "OT/tie 1.5, reg loss 0). Once pulled, the goalie is assumed to stay "
           "pulled. The aggressive pull times are a known, robust feature of this "
           "model class — refinements for team identity and time-varying hazards "
           "(see the research write-up) shift the answer by only seconds.")
