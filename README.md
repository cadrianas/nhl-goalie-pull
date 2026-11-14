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
./.venv/bin/pip install -e '.[model,viz,app]'
```

## Usage

```bash
# 1. data pipeline
python -m goalie_pull.ingest --season 2022 --season 2023 --season 2024
python -m goalie_pull.exposure --season 2022   # + 2023, 2024 (NHL play-by-play)

# 2. baseline model + optimal pull curve
python scripts/run_baseline.py --seasons 2022 2023 2024

# 3. validation + per-team cost / hierarchical / time-varying
python -m goalie_pull.validate
python scripts/run_hierarchical.py
python scripts/run_timevarying.py

# 4. live tool
streamlit run app/goalie_pull_app.py
```

## Key findings (2022–24)

1. **Coaches pull far too late.** The expected-points-optimal pull time is ~6:44
   remaining when down one; coaches' median is ~2:00.
2. **It costs ~0.54 expected standings points per team per season** (NSH worst
   ~0.87, TOR/BOS best ~0.29).
3. **It's a decision problem, not a talent problem.** A hierarchical model shows
   team heterogeneity in the pulled rates is small; team-specific optimal pull
   times barely move (±~28s), dwarfed by the ~350s gap to actual behavior.
4. **The goalie-in scoring rate is strongly time-inhomogeneous** (it collapses
   over the final ~7 min), so the constant-rate assumption used across the
   literature is wrong — but modeling it shifts the optimum only +12–25s. The
   aggressive-pull conclusion is robust to every refinement.

Figures in `reports/figures/`; write-ups in `reports/*.md`.

## Status

Phases 0–4 complete: data pipeline, baseline Gamma–Poisson + optimal-stopping
solver (expected-points utility), held-out validation, per-team cost analysis,
hierarchical PyMC rate model, time-inhomogeneous hazards, and a Streamlit live
tool. Possible extensions: more seasons (2015–21) to resolve team effects,
time-varying pulled hazards, and xG-weighted events.
