"""Phase 3b: time-inhomogeneous goalie-in hazards.

The baseline treats the goalie-in scoring rate as constant, but it collapses as
regulation ends (trailing-team even-strength for-rate falls from ~0.044/min
across the game to ~0.007/min in the last 5 min). Averaging over period 3
(~0.027/min) overstates the scoring available in the actual decision window
(the last ~10 min), where the rate is both lower and declining -- so the constant
model mis-prices the value of NOT pulling.

Here we model the goalie-in for/against hazards as smooth functions of time
remaining: log lam[d, t] = level[d] + shape(t), where shape(t) is a
Gaussian random walk over 60-second bins (pooled across deficits; the TIME shape
is well estimated from all trailing data while the LEVEL varies by deficit). The
pulled hazards stay constant -- pulls occur in a narrow late window where there
is little time variation to estimate.

Selection caveat: in the final ~90 s the goalie-in sample thins as teams pull,
so the observed rate there is biased low by the pull decision itself. We model
what is observed and flag it; the decision mostly turns on the rate earlier in
the window, where most teams still have the goalie in.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pymc as pm

from . import config, rates as rates_mod

MAX_DEFICIT = rates_mod.MAX_DEFICIT
BIN_S = 60
N_BINS = 1200 // BIN_S           # period-3 bins (0..19), bin b == [b*60, (b+1)*60) s remaining


def build_in_timebins(seasons) -> pd.DataFrame:
    """Per (deficit, time-bin): goalie-in trailing exposure (s) and goals."""
    df = rates_mod.load_intervals(seasons)
    late = df[(df["period"] == 3) & (df["empty_net_team"] == "none")
              & (df["period_type"] == "REG")].copy()
    diff = late["home_score"] - late["away_score"]
    late["trailing"] = np.where(diff < 0, "home", np.where(diff > 0, "away", "none"))
    late["deficit"] = np.abs(diff).clip(upper=MAX_DEFICIT).astype(int)
    late = late[(late["trailing"] != "none") & (late["deficit"] >= 1)]
    late["bin"] = (late["sec_remaining_reg"] // BIN_S).clip(upper=N_BINS - 1).astype(int)
    is_goal = late["type"] == "goal"
    late["gf"] = (is_goal & (late["scoring_team"] == late["trailing"])).astype(int)
    late["ga"] = (is_goal & (late["scoring_team"] != late["trailing"])
                  & (late["scoring_team"] != "none")).astype(int)
    agg = (late.groupby(["deficit", "bin"])
           .agg(exposure=("duration", "sum"), gf=("gf", "sum"), ga=("ga", "sum"))
           .reset_index())
    return agg[agg["exposure"] > 0].reset_index(drop=True)


class TVInRates:
    """Posterior of the time-varying goalie-in hazards; serves per-second draws."""

    def __init__(self, idata):
        p = idata.posterior
        def flat(name):
            v = p[name].values
            return v.reshape(-1, *v.shape[2:])
        self.idata = idata
        self._lvl_for = flat("level_for")       # (S, MAX_DEFICIT)
        self._lvl_ag = flat("level_ag")
        self._shape_for = flat("shape_for")      # (S, N_BINS)
        self._shape_ag = flat("shape_ag")

    def lam_bins(self, direction, deficit):
        """Posterior draws of lam per bin: (S, N_BINS), per second."""
        d = deficit - 1
        if direction == "for":
            return np.exp(self._lvl_for[:, [d]] + self._shape_for)
        return np.exp(self._lvl_ag[:, [d]] + self._shape_ag)

    def rate_tod(self, direction, deficit, n, rng, horizon):
        """n posterior draws of the per-second rate at each t in 0..horizon:
        returns array (n, horizon+1)."""
        lam = self.lam_bins(direction, deficit)          # (S, N_BINS)
        idx = rng.integers(0, lam.shape[0], size=n)
        lam = lam[idx]                                    # (n, N_BINS)
        t = np.arange(horizon + 1)
        bins = np.minimum(t // BIN_S, N_BINS - 1)
        return lam[:, bins]                               # (n, horizon+1)


def fit(agg: pd.DataFrame, *, draws=1000, tune=1000, chains=4, seed=0):
    d_i = (agg["deficit"] - 1).to_numpy()
    b_i = agg["bin"].to_numpy()
    E = agg["exposure"].to_numpy().astype(float)
    gf, ga = agg["gf"].to_numpy(), agg["ga"].to_numpy()

    with pm.Model() as model:
        # Shared smooth time-shape (random walk over bins), anchored near 0;
        # per-deficit level. Separate shape for for/against (different dynamics).
        s_for = pm.HalfNormal("s_for", 0.3)
        shape_for = pm.GaussianRandomWalk(
            "shape_for", sigma=s_for, init_dist=pm.Normal.dist(0, 0.1), shape=N_BINS)
        level_for = pm.Normal("level_for", np.log(0.03 / 60), 1.0, shape=MAX_DEFICIT)
        log_for = level_for[d_i] + shape_for[b_i]
        pm.Poisson("gf_obs", mu=E * pm.math.exp(log_for), observed=gf)

        s_ag = pm.HalfNormal("s_ag", 0.3)
        shape_ag = pm.GaussianRandomWalk(
            "shape_ag", sigma=s_ag, init_dist=pm.Normal.dist(0, 0.1), shape=N_BINS)
        level_ag = pm.Normal("level_ag", np.log(0.04 / 60), 1.0, shape=MAX_DEFICIT)
        log_ag = level_ag[d_i] + shape_ag[b_i]
        pm.Poisson("ga_obs", mu=E * pm.math.exp(log_ag), observed=ga)

        idata = pm.sample(draws=draws, tune=tune, chains=chains,
                          target_accept=0.95, random_seed=seed, progressbar=False)
    return model, idata


def build(seasons, **kw) -> TVInRates:
    agg = build_in_timebins(seasons)
    _, idata = fit(agg, **kw)
    return TVInRates(idata)
