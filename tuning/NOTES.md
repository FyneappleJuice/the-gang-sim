# Tuning notes

## Run 1 — tablereader, 4 players (stopped early, 12 of 20 configs)

Device: Steam Frame (ARM, 5 usable cores), ~2.5 CPU-s per game.
Command: `python -m thegang.tune --agent tablereader -n 4 --configs 20 --games 120`
All configs played the same 120 seeds (seed 20000). ±~9 pts noise on win rate per row.
The recheck on fresh seeds never ran, so nothing below is confirmed yet.

Reference: chipreader won 32.5% of games at 4 players (400 games, seed 1000).

```
[1/20] hand 40.6% win 35.0%  defaults
[2/20] hand 38.2% win 26.7%  {'beta': 2.0, 'history_weight': 1.0, 'displaced_weight': 0.5, 'gap_cap': 3, 'switch_margin': 0.1, 'steal_margin': 0.05, 'max_steals': 1}
[3/20] hand 36.1% win 28.3%  {'beta': 0.5, 'history_weight': 1.0, 'displaced_weight': 1.0, 'gap_cap': 2, 'switch_margin': 0.06, 'steal_margin': 0.05, 'max_steals': 0}
[4/20] hand 37.5% win 25.8%  {'beta': 2.0, 'history_weight': 1.5, 'displaced_weight': 0.5, 'gap_cap': 3, 'switch_margin': 0.15, 'steal_margin': 0.3, 'max_steals': 1}
[5/20] hand 45.8% win 40.8%  {'beta': 1.0, 'history_weight': 1.0, 'displaced_weight': 0.5, 'gap_cap': 3, 'switch_margin': 0.15, 'steal_margin': 0.2, 'max_steals': 2}
[6/20] hand 38.2% win 28.3%  {'beta': 1.0, 'history_weight': 1.5, 'displaced_weight': 0.0, 'gap_cap': 2, 'switch_margin': 0.06, 'steal_margin': 0.3, 'max_steals': 2}
[7/20] hand 35.6% win 23.3%  {'beta': 4.0, 'history_weight': 1.5, 'displaced_weight': 0.0, 'gap_cap': 1, 'switch_margin': 0.1, 'steal_margin': 0.1, 'max_steals': 2}
[8/20] hand 56.0% win 59.2%  {'beta': 1.0, 'history_weight': 0.3, 'displaced_weight': 1.0, 'gap_cap': 2, 'switch_margin': 0.03, 'steal_margin': 0.2, 'max_steals': 1}
[9/20] hand 45.1% win 42.5%  {'beta': 1.0, 'history_weight': 0.6, 'displaced_weight': 0.0, 'gap_cap': 3, 'switch_margin': 0.0, 'steal_margin': 0.1, 'max_steals': 1}
[10/20] hand 30.8% win 20.0%  {'beta': 1.0, 'history_weight': 1.5, 'displaced_weight': 0.0, 'gap_cap': 3, 'switch_margin': 0.03, 'steal_margin': 0.2, 'max_steals': 0}
[11/20] hand 40.2% win 34.2%  {'beta': 1.0, 'history_weight': 1.5, 'displaced_weight': 0.5, 'gap_cap': 1, 'switch_margin': 0.06, 'steal_margin': 0.0, 'max_steals': 2}
[12/20] hand 29.7% win 14.2%  {'beta': 1.5, 'history_weight': 1.5, 'displaced_weight': 0.5, 'gap_cap': 1, 'switch_margin': 0.15, 'steal_margin': 0.3, 'max_steals': 0}
```

### Observations
- Best: #8 (beta 1.0, history_weight 0.3, displaced_weight 1.0, gap_cap 2,
  switch_margin 0.03, steal_margin 0.2, max_steals 1): 59.2% wins vs 35.0% for defaults.
- history_weight 1.5 was bad in all six configs that used it (29.7–40.2% hands).
  Likely cause: the street-strength proxies for earlier streets are crude, so
  leaning hard on old-round chips misleads the bot. Low values (0.3) looked best.
- max_steals 0 looked bad (#3, #10, #12).

### Next step (on the PC)
Search around #8, then the tuner re-checks the top configs on fresh seeds:

```
python -m thegang.tune --agent tablereader -n 4 --configs 40 --games 200 --out tuning/run2_4p.json \
  --base '{"beta": 1.0, "history_weight": 0.3, "displaced_weight": 1.0, "gap_cap": 2, "switch_margin": 0.03, "steal_margin": 0.2, "max_steals": 1}'
```
Then: put the winning params into `TableReaderAgent.__init__` defaults, and
benchmark against chipreader at 3/4/5/6 players with fresh seeds, e.g.
`python -m thegang -n 5 -a tablereader -g 400 --seed 1000`.
Also worth trying history_weight 0.0–0.3 (add 0.0 / 0.1 to the space).
