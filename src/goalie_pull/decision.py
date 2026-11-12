"""Phase 1: optimal-stopping solver for the goalie-pull decision.

Given per-second goal rates (from rates.py), we solve for the pull-time threshold
that maximizes the probability of avoiding a regulation loss (reaching at least a
tie). Because we carry full posterior draws of the rates through the solver, the
output is a *distribution* over the optimal pull time per deficit -- the headline
Bayesian result.

Model (baseline, time-homogeneous competing Poisson):
    State = (deficit d, seconds remaining t). Each second, with the chosen action
    (goalie IN or OUT) having rates (lam_for, lam_against):
        score    (prob lam_for)      -> deficit d-1
        concede  (prob lam_against)  -> deficit d+1
        nothing                      -> deficit d, one second less
    Value V(d, t) = P(reach tie-or-better by t=0).
      V(d<=0, t) = 1   (tie achieved -> guaranteed at least the OT point)
      V(d, 0)    = 0   for d >= 1
      deficits beyond MAX_DEFICIT are treated as lost (V = 0).
    At each (d, t) we take the action with the higher value; the optimal pull
    threshold is the largest t at which pulling (OUT) is already optimal.

Simplification: V(d<=0)=1 treats reaching a tie as success (an OT point). The
opponent pulling / re-scoring in the last seconds is ignored -- standard for the
baseline and the thing the hierarchical model can later relax.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import rates as rates_mod

HORIZON_S = 600   # solve the last 10 minutes, per second (rates ~constant within P3)
MAX_DEFICIT = rates_mod.MAX_DEFICIT


@dataclass
class PullPolicy:
    deficit: int
    threshold_draws: np.ndarray      # optimal pull time (sec remaining) per draw
    tie_prob_optimal: np.ndarray     # P(tie) under optimal policy, per draw
    tie_prob_never: np.ndarray       # P(tie) if goalie never pulled, per draw

    def summary(self) -> dict:
        t = self.threshold_draws
        return {
            "deficit": self.deficit,
            "pull_median_s": float(np.median(t)),
            "pull_90ci_s": (float(np.quantile(t, 0.05)),
                            float(np.quantile(t, 0.95))),
            "tie_prob_optimal": float(np.median(self.tie_prob_optimal)),
            "tie_prob_never": float(np.median(self.tie_prob_never)),
            "tie_prob_gain": float(np.median(self.tie_prob_optimal
                                             - self.tie_prob_never)),
        }


def _draw_rates(post: dict, n: int, rng: np.random.Generator) -> dict:
    """Posterior draws of all four rates for each deficit 1..MAX_DEFICIT."""
    out = {}
    for d in range(1, MAX_DEFICIT + 1):
        for state in ("in", "out"):
            for direction in ("for", "against"):
                key = (state, d, direction)
                if key in post:
                    out[key] = post[key].draws(n, rng)
                else:  # cell never observed -> fall back to a tiny rate
                    out[key] = np.full(n, 1e-6)
    return out


def solve(post: dict, *, n_draws: int = 4000, horizon: int = HORIZON_S,
          seed: int = 0, force_pull: bool = False) -> dict[int, PullPolicy]:
    """Backward induction, vectorized over posterior draws.

    Returns {deficit: PullPolicy}. If force_pull is True, the goalie is always
    pulled (used to compute the 'never pull' counterfactual by inversion).
    """
    rng = np.random.default_rng(seed)
    r = _draw_rates(post, n_draws, rng)

    # V[d] has shape (n_draws,). d indexes 0..MAX_DEFICIT+1; d<=0 -> 1, d>MAX -> 0.
    def new_V():
        V = {d: np.zeros(n_draws) for d in range(1, MAX_DEFICIT + 1)}
        return V

    def value_table(policy: str):
        """policy in {'optimal','in','out'}. Returns (V_by_t, action_out_by_t).
        V_by_t[t][d], action recorded only for 'optimal'."""
        V = new_V()                      # t = 0
        thresholds_hit = {d: np.full(n_draws, np.nan) for d in range(1, MAX_DEFICIT + 1)}
        for t in range(1, horizon + 1):
            V_next = {}
            for d in range(1, MAX_DEFICIT + 1):
                V_dm1 = np.ones(n_draws) if d - 1 <= 0 else V[d - 1]          # tie or better
                V_dp1 = np.zeros(n_draws) if d + 1 > MAX_DEFICIT else V[d + 1]  # deeper hole
                V_same = V[d]

                def val(state):
                    lf = r[(state, d, "for")]
                    la = r[(state, d, "against")]
                    return lf * V_dm1 + la * V_dp1 + (1.0 - lf - la) * V_same

                if policy == "optimal":
                    v_in, v_out = val("in"), val("out")
                    choose_out = v_out >= v_in
                    V_next[d] = np.where(choose_out, v_out, v_in)
                    # t increases through the loop, so overwriting on every
                    # choose_out leaves the LARGEST t at which pulling is optimal
                    # == the pull threshold (pull when time remaining hits it).
                    thresholds_hit[d] = np.where(choose_out, t, thresholds_hit[d])
                elif policy == "in":
                    V_next[d] = val("in")
                else:
                    V_next[d] = val("out")
            V = V_next
        return V, thresholds_hit

    V_opt, thresholds = value_table("optimal")
    V_never, _ = value_table("in")

    policies = {}
    for d in range(1, MAX_DEFICIT + 1):
        thr = thresholds[d]
        thr = np.where(np.isnan(thr), 0.0, thr)   # never worth pulling -> 0
        policies[d] = PullPolicy(
            deficit=d,
            threshold_draws=thr,
            tie_prob_optimal=V_opt[d],
            tie_prob_never=V_never[d],
        )
    return policies


def run(seasons: list[int], *, n_draws: int = 4000, seed: int = 0):
    table, post = rates_mod.build(seasons)
    policies = solve(post, n_draws=n_draws, seed=seed)
    return table, post, policies


def _fmt(policies: dict[int, PullPolicy]) -> str:
    lines = ["deficit  optimal pull (s remaining)   P(tie) optimal   P(tie) never   gain"]
    for d, pol in sorted(policies.items()):
        s = pol.summary()
        lo, hi = s["pull_90ci_s"]
        lines.append(
            f"  down {d}   {s['pull_median_s']:>5.0f}s  (90% CI {lo:.0f}-{hi:.0f})"
            f"        {s['tie_prob_optimal']:.3f}          {s['tie_prob_never']:.3f}"
            f"      +{s['tie_prob_gain']:.3f}")
    return "\n".join(lines)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--seasons", type=int, nargs="+", default=[2022, 2023, 2024])
    ap.add_argument("--draws", type=int, default=4000)
    args = ap.parse_args()
    table, post, policies = run(args.seasons, n_draws=args.draws)
    print(rates_mod._fmt(post))
    print()
    print(_fmt(policies))
