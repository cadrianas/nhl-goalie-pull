# Phase 3a: hierarchical pulled-rate model

Seasons [2022, 2023, 2024]. PyMC hierarchical Poisson, log link, non-centered team effects (6v5 attack/suppression, empty-net finishing/prevention).

Convergence: max r-hat **1.004**, min ESS **1265**.

## Key finding: team heterogeneity is small

After partial pooling, pulled 6v5 for-rates span only 0.142-0.151 per 60s and empty-net against-rates 0.358-0.384 -- ~7% around the mean. Teams pull too rarely for the data to support large team-specific deviations.

## Consequence for the decision

| deficit | league optimal | team-specific range | sd across teams |
|---|---|---|---|
| down 1 | 404s | 414-519s | 28.2s |
| down 2 | 394s | 370-427s | 15.1s |
| down 3 | 377s | 265-301s | 9.5s |

Team-specific optima cluster at ~464s with a team-to-team sd of only 28s. The gap to actual coach behavior (~353s) dwarfs that variation: **personnel barely moves the answer** -- essentially every team should pull far earlier.

**Interpretation.** The points left on the table (Phase 2) is a *decision* problem, not a *talent* problem: teams are not bad at 6-on-5, they pull too late. This is an honest negative result on the 'matchup-specific timing' hypothesis -- and a cleaner prescription: the aggressive-pull rule applies to essentially everyone.