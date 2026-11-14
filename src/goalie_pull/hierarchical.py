"""Phase 3a: hierarchical PyMC model for the pulled goal rates.

The baseline (rates.py) fits one league-average rate per deficit. Here we let the
6-on-5 goal-for and empty-net goal-against rates vary by matchup, with a LOG link
so team and opponent contributions are non-additive on the rate scale -- the
deliberate departure from the league-average models of Asness-Brown / Beaudoin-
Swartz.

For a pull by team i against team j at deficit d (per-second rates):
    log lam_for[i,j,d]     = a_for + b_for[d] + off[i]      + def[j]
    log lam_against[i,j,d]  = a_ag  + b_ag[d]  + eng_off[j]  + eng_prev[i]
where off (6v5 attack), def (6v5 suppression), eng_off (empty-net finishing) and
eng_prev (empty-net prevention) are partially-pooled team effects:
    off ~ Normal(0, sigma_off), etc.   -> shrinkage toward the league rate.

The fitted object exposes posterior rate DRAWS per matchup through the same
interface rates.py uses, so decision.solve() consumes it unchanged -- yielding a
team-specific optimal pull time. The goalie-IN rates stay at the league level
(baseline) for now; only the decision-critical pulled rates are hierarchical.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import pymc as pm

from . import config, rates as rates_mod
from .validate import _game_team_map

MAX_DEFICIT = rates_mod.MAX_DEFICIT


def build_pulled_dataset(seasons) -> pd.DataFrame:
    """Per (puller, defender, deficit): pulled exposure (s), goals for/against."""
    intervals = rates_mod.load_intervals(seasons)
    tp = intervals[intervals["is_trailing_pull"]].copy()
    tp["deficit"] = (-tp["pulling_team_score_diff"]).clip(upper=MAX_DEFICIT).astype(int)
    gmap = _game_team_map(seasons)
    home = tp["game_id"].map(lambda g: gmap.get(int(g), (None, None))[0])
    away = tp["game_id"].map(lambda g: gmap.get(int(g), (None, None))[1])
    tp["puller"] = np.where(tp["empty_net_team"] == "home", home, away)
    tp["defender"] = np.where(tp["empty_net_team"] == "home", away, home)
    tp = tp[tp["puller"].notna() & tp["defender"].notna()]
    is_goal = tp["type"] == "goal"
    tp["gf"] = (is_goal & (tp["scoring_team"] == tp["empty_net_team"])).astype(int)
    tp["ga"] = (is_goal & (tp["scoring_team"] != tp["empty_net_team"])
                & (tp["scoring_team"] != "none")).astype(int)
    agg = (tp.groupby(["puller", "defender", "deficit"])
           .agg(exposure=("duration", "sum"), gf=("gf", "sum"), ga=("ga", "sum"))
           .reset_index())
    return agg[agg["exposure"] > 0].reset_index(drop=True)


@dataclass
class _DrawSource:
    """Wraps a flat array of posterior per-second rate samples; resamples to n."""
    samples: np.ndarray

    def draws(self, n: int, rng: np.random.Generator) -> np.ndarray:
        return rng.choice(self.samples, size=n, replace=True)


class HierRates:
    """Fitted hierarchical model; produces per-matchup rate posteriors and a
    `post`-dict compatible with decision.solve()."""

    def __init__(self, idata, teams, league_post):
        self.idata = idata
        self.teams = teams
        self.tidx = {t: i for i, t in enumerate(teams)}
        self.league = league_post   # league GammaPosteriors (for the 'in' rates)
        p = idata.posterior
        # Flatten chain/draw -> sample axis for the parameters we need.
        def flat(name):
            return p[name].values.reshape(-1, *p[name].values.shape[2:])
        self._a_for = flat("a_for")            # (nsamples, MAX_DEFICIT)
        self._off, self._def = flat("off_for"), flat("def_for")
        self._a_ag = flat("a_ag")
        self._eng_off, self._eng_prev = flat("eng_off"), flat("eng_prev")

    def _rate(self, direction, puller, defender, deficit) -> np.ndarray:
        d = deficit - 1
        pi = self.tidx[puller]
        qi = self.tidx[defender] if defender is not None else None
        if direction == "for":
            lin = self._a_for[:, d] + self._off[:, pi]
            if qi is not None:
                lin = lin + self._def[:, qi]
            return np.exp(lin)
        lin = self._a_ag[:, d] + self._eng_prev[:, pi]
        if qi is not None:
            lin = lin + self._eng_off[:, qi]
        return np.exp(lin)

    def post_for_matchup(self, puller, defender=None) -> dict:
        """decision.solve-ready dict. defender=None -> average opponent.

        Pulled ('out') rates come from the hierarchical posterior; goalie-in
        rates reuse the league-level baseline posteriors."""
        post = {}
        for d in range(1, MAX_DEFICIT + 1):
            post[("out", d, "for")] = _DrawSource(self._rate("for", puller, defender, d))
            post[("out", d, "against")] = _DrawSource(self._rate("against", puller, defender, d))
            for direction in ("for", "against"):
                key = ("in", d, direction)
                if key in self.league:
                    post[key] = self.league[key]
        return post

    def team_rate_table(self) -> pd.DataFrame:
        """League-average-opponent per-team pulled rates (per 60s), deficit 1."""
        rows = []
        for t in self.teams:
            rf = self._rate("for", t, None, 1) * 60
            ra = self._rate("against", t, None, 1) * 60
            rows.append({"team": t,
                         "for_60": float(np.median(rf)),
                         "against_60": float(np.median(ra))})
        return pd.DataFrame(rows)


def fit(agg: pd.DataFrame, *, draws=1000, tune=1000, chains=4, seed=0):
    teams = sorted(set(agg["puller"]) | set(agg["defender"]))
    tidx = {t: i for i, t in enumerate(teams)}
    nT = len(teams)
    pull_i = agg["puller"].map(tidx).to_numpy()
    def_i = agg["defender"].map(tidx).to_numpy()
    d_i = (agg["deficit"] - 1).to_numpy()
    E = agg["exposure"].to_numpy().astype(float)
    gf, ga = agg["gf"].to_numpy(), agg["ga"].to_numpy()

    with pm.Model() as model:
        # FOR: 6-on-5 scoring. Per-deficit fixed intercepts (deficits are 3
        # data-rich categories, not exchangeable units to pool). Non-centered
        # team effects (avoids the near-zero-variance funnel).
        a_for = pm.Normal("a_for", np.log(0.15 / 60), 1.0, shape=MAX_DEFICIT)
        s_off = pm.HalfNormal("s_off", 0.5)
        s_def = pm.HalfNormal("s_def", 0.5)
        off = pm.Deterministic("off_for", pm.Normal("off_z", 0.0, 1.0, shape=nT) * s_off)
        dff = pm.Deterministic("def_for", pm.Normal("def_z", 0.0, 1.0, shape=nT) * s_def)
        log_for = a_for[d_i] + off[pull_i] + dff[def_i]
        pm.Poisson("gf_obs", mu=E * pm.math.exp(log_for), observed=gf)

        # AGAINST: empty-net goals. defender finishes, puller prevents.
        a_ag = pm.Normal("a_ag", np.log(0.40 / 60), 1.0, shape=MAX_DEFICIT)
        s_eo = pm.HalfNormal("s_eng_off", 0.5)
        s_ep = pm.HalfNormal("s_eng_prev", 0.5)
        eng_off = pm.Deterministic("eng_off", pm.Normal("eo_z", 0.0, 1.0, shape=nT) * s_eo)
        eng_prev = pm.Deterministic("eng_prev", pm.Normal("ep_z", 0.0, 1.0, shape=nT) * s_ep)
        log_ag = a_ag[d_i] + eng_off[def_i] + eng_prev[pull_i]
        pm.Poisson("ga_obs", mu=E * pm.math.exp(log_ag), observed=ga)

        idata = pm.sample(draws=draws, tune=tune, chains=chains,
                          target_accept=0.9, random_seed=seed, progressbar=False)
    return model, idata, teams


def build(seasons, **kw) -> HierRates:
    agg = build_pulled_dataset(seasons)
    _, league_post = rates_mod.build(seasons)
    _, idata, teams = fit(agg, **kw)
    return HierRates(idata, teams, league_post)
