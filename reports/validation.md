# Phase 2: validation & cost analysis

## Held-out validation (train [2022, 2023] -> test [2024])

Posterior-predictive 95% coverage on the held-out season: **12/12 = 100%** of rate cells. The Gamma-Poisson rates fit on the training seasons predict the test season's goal counts with no systematic miscalibration.

## Points left on the table

League average: **0.54 expected standings points per team per season** forgone by pulling later than optimal. Note the split between *decision quality* (`cost_per_sit_down1`) and *exposure* (how often a team trails): a good team can leave few total points simply by rarely being behind, not by pulling well.

| team | pulls | down-1 pull (s) | cost/situation (down 1) | points left/season |
|---|---|---|---|---|
| NSH | 41 | 69 | 0.099 | 0.869 |
| MTL | 42 | 102 | 0.075 | 0.779 |
| CHI | 39 | 105 | 0.073 | 0.766 |
| FLA | 53 | 120 | 0.060 | 0.687 |
| OTT | 39 | 100 | 0.074 | 0.685 |
| WPG | 39 | 96 | 0.077 | 0.680 |
| EDM | 27 | 75 | 0.093 | 0.666 |
| WSH | 45 | 109 | 0.067 | 0.657 |
| NYI | 38 | 107 | 0.070 | 0.643 |
| COL | 33 | 102 | 0.072 | 0.598 |
| STL | 33 | 90 | 0.081 | 0.587 |
| SEA | 34 | 100 | 0.074 | 0.585 |
| DET | 34 | 88 | 0.082 | 0.575 |
| ARI | 30 | 98 | 0.076 | 0.545 |
| CBJ | 38 | 106 | 0.070 | 0.536 |
| PIT | 29 | 117 | 0.063 | 0.533 |
| VAN | 39 | 140 | 0.050 | 0.498 |
| BUF | 37 | 121 | 0.059 | 0.491 |
| VGK | 30 | 111 | 0.067 | 0.481 |
| SJS | 38 | 140 | 0.049 | 0.476 |
| PHI | 39 | 148 | 0.046 | 0.475 |
| LAK | 24 | 102 | 0.073 | 0.471 |
| TBL | 36 | 150 | 0.048 | 0.460 |
| CAR | 31 | 120 | 0.061 | 0.455 |
| DAL | 34 | 124 | 0.057 | 0.451 |
| NJD | 32 | 132 | 0.054 | 0.414 |
| CGY | 24 | 111 | 0.065 | 0.411 |
| ANA | 23 | 113 | 0.066 | 0.407 |
| NYR | 30 | 148 | 0.048 | 0.391 |
| MIN | 35 | 134 | 0.052 | 0.386 |
| TOR | 28 | 157 | 0.041 | 0.307 |
| BOS | 18 | 118 | 0.061 | 0.288 |