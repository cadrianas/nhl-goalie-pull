# Bayesian Goalie Pull

When should an NHL team pull its goalie for an extra attacker? This project treats
it as a Bayesian decision problem: estimate the rates at which goals are scored
*for* and *against* once the goalie is pulled, carry the full posterior uncertainty
through, and solve for the pull time that maximizes win/tie probability.

## Why Bayesian

Empty-net situations are rare per game, so naive per-state goal rates are noisy.
A hierarchical Bayesian model pools information across score states, teams, and
eras, and — crucially — produces a posterior *distribution* over the optimal pull
time rather than a single point estimate.

## Approach (baseline first)

1. **Rates.** Competing Poisson processes while pulled: goals-for rate vs.
   empty-net goals-against rate. Baseline uses a conjugate Gamma–Poisson
   (analytic); later a hierarchical PyMC model with the same interface.
2. **Decision.** Backward-induction optimal stopping over the last ~5 minutes on a
   per-second grid: at each second, is expected win/tie probability higher pulling
   now or waiting?
3. **Output.** The optimal pull-time curve by score differential *with credible
   intervals*, a comparison to what coaches actually do, and a live Streamlit tool.

## Data

MoneyPuck shot-level data (one 137-column CSV per season), cached locally from the
peter-tanner.com mirror (the official moneypuck.com paths have moved; override with
`GOALIE_PULL_SHOTS_BASE_URL`). We keep ~20 relevant columns.

**Important data subtlety:** an empty net is *not* always a goalie pull. Teams also
pull for a 6th attacker on a delayed penalty (any time, often tied). `ingest.py`
flags genuine late trailing pulls (`is_trailing_pull`: net empty, puller trailing,
regulation, ≤5 min left) separately so the rate model isn't polluted.

## Project layout

```
src/goalie_pull/
  config.py        paths, seasons, data-source URLs
  ingest.py        download + clean MoneyPuck shots          (Phase 0a) DONE
  exposure.py      reconstruct empty-net intervals / exposure (Phase 0b) TODO
  rates.py         Gamma-Poisson posteriors                   (Phase 1)  TODO
  decision.py      backward-induction optimal stopping        (Phase 1)  TODO
  hierarchical.py  pooled PyMC model (same interface)         (Phase 3)  TODO
app/               Streamlit live tool                        (Phase 4)  TODO
notebooks/         EDA + narrative analysis
```

## Setup

```bash
python3.12 -m venv .venv
./.venv/bin/pip install -e .            # core (ingest) deps
./.venv/bin/pip install -e '.[model,viz]'   # when you reach Phase 1
```

## Usage

```bash
python -m goalie_pull.ingest --season 2024   # one season
python -m goalie_pull.ingest --all           # all DEFAULT_SEASONS
```

Writes `data/processed/shots.parquet` and prints a sanity summary.

## Status

Phase 0a (ingest) complete and verified on 2024. Next: Phase 0b, reconstruct
empty-net *intervals* to get exposure time — the denominator for the Poisson
rates — then the baseline Gamma–Poisson model.

## The key modeling decisions (open)

- **Event = raw goals vs. xG-weighted shots.** Raw empty-net goals are sparse;
  leaning toward modeling the xG-weighted shot process, with raw goals for validation.
- **Utility = win probability vs. expected standings points** (reg win 2, OT loss 1).
  Supported as alternate terminal-value functions in the solver.
