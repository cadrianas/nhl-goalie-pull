# Baseline results (Phase 1)

Conjugate Gamma-Poisson rates + backward-induction optimal stopping.
Utility = expected NHL standings points (reg win 2, OT/tie 1.5, reg loss 0).

## Optimal vs. actual pull time (seconds remaining)

| deficit | model optimal (90% CI) | actual coach median | E[points] gain | games |
|---|---|---|---|---|
| down 1 | 404s (342-498) | 119s | +0.189 | 1628 |
| down 2 | 394s (339-459) | 163s | +0.096 | 1389 |
| down 3 | 377s (304-462) | 252s | +0.039 | 376 |

## Fitted rates (per 60s)
```
state deficit  dir       goals   exp(min)   rate/60s   95% CI/60s
  in      1  against     832   19931.1     0.042   [0.039, 0.045]
  in      1      for     533   19931.1     0.027   [0.025, 0.029]
  in      2  against     329   14172.6     0.023   [0.021, 0.026]
  in      2      for     605   14172.6     0.043   [0.039, 0.046]
  in      3  against      96    6373.7     0.015   [0.012, 0.019]
  in      3      for     470    6373.7     0.074   [0.067, 0.081]
 out      1  against     654    1657.7     0.395   [0.366, 0.426]
 out      1      for     236    1657.7     0.143   [0.126, 0.162]
 out      2  against     669    1608.6     0.417   [0.386, 0.449]
 out      2      for     232    1608.6     0.145   [0.127, 0.164]
 out      3  against     159     471.7     0.340   [0.290, 0.395]
 out      3      for      88     471.7     0.189   [0.152, 0.231]
```