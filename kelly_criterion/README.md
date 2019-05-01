The Kelly Criterion shows you how to optimally size a bet to minimize 
the risk of losing your bank-roll while maximizing the rate at which you can
exploit an edge you may have in any particular probablistic game.


It is given by:

```
bet_fraction = (odds * P_win - P_lose)/odds

bet_size = bet_fraction * bankroll_size
```

