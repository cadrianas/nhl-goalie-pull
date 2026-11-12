"""Bayesian inference for when to pull an NHL goalie.

Package layout (built phase by phase):
    config.py        paths, seasons, data-source URLs
    ingest.py        download + clean MoneyPuck shot data   (Phase 0a)  <-- here now
    exposure.py      reconstruct empty-net intervals        (Phase 0b)  TODO
    rates.py         Gamma-Poisson posteriors for goal rates (Phase 1)   TODO
    decision.py      backward-induction optimal-stopping     (Phase 1)   TODO
    hierarchical.py  pooled PyMC model (same interface)      (Phase 3)   TODO
"""

__version__ = "0.0.1"
