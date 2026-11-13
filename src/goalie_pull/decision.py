"""Phase 1: optimal-stopping solver for the goalie-pull decision.

Given per-second goal rates (from rates.py), we solve for the pull-time threshold
that maximizes expected value. Posterior draws of the rates are carried through
the solver, so the output is a *distribution* over the optimal pull time per
deficit -- the headline Bayesian result.

Two utilities are supported:

  utility="tie_prob"  -- P(reach at least a tie in regulation). Simple; treats a
     tie as an absorbing success and conceding-while-trailing as (nearly)
     costless. Known to recommend pulling implausibly early.

  utility="points"    -- expected NHL standings points (the Asness-Brown framing,
     DEFAULT). Terminal payoffs: regulation win = 2, reach OT (tie) = 1.5
     (1 guaranteed + a 50/50 OT point), regulation loss = 0. This REQUIRES
     tracking the deficit going negative (the trailing team scoring again to WIN
     in regulation) and tied/leading dynamics, which is why it can differ from
     tie_prob rather than just rescaling it.

Model (baseline, time-homogeneous competing Poisson): state = (signed deficit d,
seconds remaining t). d>0 trailing, d=0 tied, d<0 leading. Each second, with the
chosen action's rates (lam_for, lam_against): score -> d-1, concede -> d+1, else
stay. The goalie is only ever pulled while trailing (d>=1); tied/leading states
play the goalie in. Deeper than MAX_DEFICIT is treated as lost; a lead beyond
MAX_LEAD as won.

For leading states we reuse the trailing even-strength rates with roles swapped
(being up k is, at even strength, the mirror of being down k); the tied rate is
the average of the down-1 for/against rates. These are baseline approximations
the hierarchical model can replace.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import rates as rates_mod

HORIZON_S = 600            # solve the last 10 minutes, per second
MAX_DEFICIT = rates_mod.MAX_DEFICIT
MAX_LEAD = 3

# Terminal standings-points payoff as a function of signed deficit d.
def _points_payoff(d: int) -> float:
    if d <= -1:
        return 2.0       # leading at the buzzer -> regulation win
    if d == 0:
        return 1.5       # tied -> OT: 1 guaranteed + 50/50 for the 2nd point
    return 0.0           # trailing -> regulation loss


@dataclass
class PullPolicy:
    deficit: int
    utility: str
    threshold_draws: np.ndarray      # optimal pull time (sec remaining) per draw
    value_optimal: np.ndarray        # E[value] under optimal policy, per draw
    value_never: np.ndarray          # E[value] if goalie never pulled, per draw

    def summary(self) -> dict:
        t = self.threshold_draws
        return {
            "deficit": self.deficit,
            "utility": self.utility,
            "pull_median_s": float(np.median(t)),
            "pull_90ci_s": (float(np.quantile(t, 0.05)),
                            float(np.quantile(t, 0.95))),
            "value_optimal": float(np.median(self.value_optimal)),
            "value_never": float(np.median(self.value_never)),
            "value_gain": float(np.median(self.value_optimal - self.value_never)),
        }


def _even_strength_rates(post: dict, n: int, rng: np.random.Generator) -> dict:
    """Signed-deficit -> (lam_for, lam_against) draws for goalie-IN play.

    d>=1 trailing: straight from the fitted 'in' cells.
    d<=-1 leading: mirror of being down |d| (for<->against swapped).
    d==0 tied:     average of the down-1 for/against rates.
    """
    def cell(state, d, direction):
        key = (state, d, direction)
        return post[key].draws(n, rng) if key in post else np.full(n, 1e-6)

    rates = {}
    for d in range(1, MAX_DEFICIT + 1):          # trailing
        rates[d] = (cell("in", d, "for"), cell("in", d, "against"))
    for k in range(1, MAX_LEAD + 1):             # leading by k == down k, swapped
        rates[-k] = (cell("in", k, "against"), cell("in", k, "for"))
    tie = 0.5 * (cell("in", 1, "for") + cell("in", 1, "against"))
    rates[0] = (tie.copy(), tie.copy())
    return rates


def _pulled_rates(post: dict, n: int, rng: np.random.Generator) -> dict:
    out = {}
    for d in range(1, MAX_DEFICIT + 1):
        def cell(direction):
            key = ("out", d, direction)
            return post[key].draws(n, rng) if key in post else np.full(n, 1e-6)
        out[d] = (cell("for"), cell("against"))
    return out


def solve(post: dict, *, utility: str = "points", n_draws: int = 4000,
          horizon: int = HORIZON_S, seed: int = 0) -> dict[int, PullPolicy]:
    """Backward induction over the signed-deficit grid, vectorized over draws."""
    rng = np.random.default_rng(seed)
    es = _even_strength_rates(post, n_draws, rng)     # goalie-in rates, all d
    pulled = _pulled_rates(post, n_draws, rng)        # goalie-out rates, d>=1

    grid = list(range(-MAX_LEAD, MAX_DEFICIT + 1))
    won = 2.0 if utility == "points" else 1.0         # value past the lead cap
    lost = 0.0
    z = np.zeros(n_draws)

    def payoff(d: int) -> np.ndarray:
        if utility == "points":
            return z + _points_payoff(d)
        return z + (1.0 if d <= 0 else 0.0)

    def below(V, d):  # value at deficit d-1 (we scored)
        return V[d - 1] if (d - 1) in V else (z + won)
    def above(V, d):  # value at deficit d+1 (we conceded)
        return V[d + 1] if (d + 1) in V else (z + lost)

    def step(V, *, allow_pull: bool, track: dict | None, t: int):
        Vn = {}
        for d in grid:
            Vb, Va, Vs = below(V, d), above(V, d), V[d]
            lf_in, la_in = es[d]
            v_in = lf_in * Vb + la_in * Va + (1 - lf_in - la_in) * Vs
            if allow_pull and d >= 1:
                lf_o, la_o = pulled[d]
                v_out = lf_o * Vb + la_o * Va + (1 - lf_o - la_o) * Vs
                choose_out = v_out >= v_in
                Vn[d] = np.where(choose_out, v_out, v_in)
                if track is not None:
                    track[d] = np.where(choose_out, t, track[d])
            else:
                Vn[d] = v_in
        return Vn

    # Optimal policy (pulling allowed while trailing) + threshold tracking.
    V = {d: payoff(d) for d in grid}
    thr = {d: np.zeros(n_draws) for d in range(1, MAX_DEFICIT + 1)}
    for t in range(1, horizon + 1):
        V = step(V, allow_pull=True, track=thr, t=t)
    V_opt = V

    # Never-pull counterfactual.
    V = {d: payoff(d) for d in grid}
    for t in range(1, horizon + 1):
        V = step(V, allow_pull=False, track=None, t=t)
    V_never = V

    return {
        d: PullPolicy(deficit=d, utility=utility, threshold_draws=thr[d],
                      value_optimal=V_opt[d], value_never=V_never[d])
        for d in range(1, MAX_DEFICIT + 1)
    }


def run(seasons: list[int], *, utility: str = "points", n_draws: int = 4000,
        seed: int = 0):
    table, post = rates_mod.build(seasons)
    policies = solve(post, utility=utility, n_draws=n_draws, seed=seed)
    return table, post, policies


def _fmt(policies: dict[int, PullPolicy]) -> str:
    u = next(iter(policies.values())).utility
    vname = "E[points]" if u == "points" else "P(tie)"
    lines = [f"deficit  optimal pull (s remaining)   {vname} optimal   {vname} never   gain"]
    for d, pol in sorted(policies.items()):
        s = pol.summary()
        lo, hi = s["pull_90ci_s"]
        lines.append(
            f"  down {d}   {s['pull_median_s']:>5.0f}s  (90% CI {lo:.0f}-{hi:.0f})"
            f"        {s['value_optimal']:.3f}          {s['value_never']:.3f}"
            f"      +{s['value_gain']:.3f}")
    return "\n".join(lines)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--seasons", type=int, nargs="+", default=[2022, 2023, 2024])
    ap.add_argument("--utility", choices=["points", "tie_prob"], default="points")
    ap.add_argument("--draws", type=int, default=4000)
    args = ap.parse_args()
    table, post, policies = run(args.seasons, utility=args.utility, n_draws=args.draws)
    print(rates_mod._fmt(post))
    print()
    print(_fmt(policies))
