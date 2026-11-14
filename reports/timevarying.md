# Phase 3b: time-inhomogeneous goalie-in hazards

Seasons [2022, 2023, 2024]. PyMC: log lam[d,t] = level[d] + shape(t), shape(t) a Gaussian random walk over 60s bins (pooled across deficits).

Convergence: max r-hat **1.007**, min ESS **427**.

## Optimal pull time: constant vs. time-varying goalie-in hazards

| deficit | constant-rate | time-varying | shift |
|---|---|---|---|
| down 1 | 404s | 419s | +15s |
| down 2 | 394s | 419s | +25s |
| down 3 | 377s | 389s | +12s |

The trailing team's goalie-in scoring rate falls steadily through the last ~15 minutes (figure `goalie_in_hazard.png`), so the constant period-3 average over-prices the offense available late. Modeling the hazard shape shifts the optimal pull time as shown -- the methodological point being that the pull decision is a *non-homogeneous* optimal-stopping problem, which the constant-rate literature does not capture.

Selection caveat: the final ~90 s of goalie-in data thins as teams pull, so the observed rate there is biased low; the decision mostly turns on the earlier part of the window where most teams still have the goalie in.