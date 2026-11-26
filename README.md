# The Gang – simulation environment

Pure-Python (3.10+, standard library only) simulator for *The Gang*, the cooperative
Texas Hold'em game: rules engine, hand evaluator, bot agents, a single-seat RL
environment and a batch runner.

## Quick start

```bash
python3 -m thegang --show -n 4                  # one game, every move printed
python3 -m thegang -n 4 -a chipreader -g 500    # win-rate estimate (uses all cores)
python3 -m thegang -n 3 -a heuristic,heuristic,chipreader -g 200   # mixed seats
python3 -m unittest discover -s tests           # tests
```

## Layout

| file | what |
|---|---|
| `thegang/cards.py`  | cards (ints 0-51), 7-card evaluator returning comparable tuples |
| `thegang/engine.py` | `GangGame` state machine: `current`, `legal_actions()`, `view(seat)`, `apply(action)` |
| `thegang/agents.py` | `Agent` base, `RandomAgent`, `HeuristicAgent`, `ChipReaderAgent` |
| `thegang/tablereader.py` | `TableReaderAgent`: reads every chip move live, places itself relative to teammates |
| `thegang/tune.py`   | random-search parameter tuner (`python -m thegang.tune --agent tablereader -n 4`) |
| `thegang/env.py`    | `GangEnv`: control one seat, teammates are bots (reset/step API) |
| `thegang/sim.py`    | `play_game`, `run` (multiprocess batch stats) |

## Rules as modelled

- 2-6 players (3-6 official). Four chip rounds: white (preflop), yellow (flop), orange (turn), red (river), each with chips 1..N.
- On your turn: `take(star)` from the centre, swap yours for a centre chip, steal another
  player's chip (they're left chipless), or `PASS` (only if you hold a chip).
- A round ends when everyone holds a chip and all N players pass in a row. After
  `max_actions_per_round` takes (default 6·N), stealing/swapping is locked so bots can't loop forever.
- Showdown in red-chip order; success if each hand ≥ the previous (ties fine). 3 vaults win, 3 alarms lose.
- The real game is simultaneous; here it's turn-based in seat order (start seat rotates).
- Not modelled yet: the advanced Challenge / Specialist cards.

## Writing a bot

```python
from thegang import Agent, PASS, take, run

class MyBot(Agent):
    def act(self, view):       # view: PlayerView (hole, board, chips, chip_history, legal_actions, ...)
        return view.legal_actions[0]
```

Register it in `agents.AGENTS` to use it from the CLI.

## Baselines (400 games each, seed 1000)

| agent | 3p win | 4p win | 6p win |
|---|---|---|---|
| random     | 4.8%  | 0.5%  | 0%   |
| heuristic  | 58.2% | 23.0% | 2.2% |
| chipreader | 71.0% | 32.5% | 2.0% |

- `heuristic`: Monte Carlo estimate of own rank → closest chip. Ignores teammates.
- `chipreader`: also reweights sampled deals by consistency with teammates' chips from *completed* rounds.
- `tablereader`:
  1. treats every live chip as evidence (including the chip a player just had stolen) and
     re-thinks after every move;
  2. for each chip it could take, estimates P(I'm correctly placed there | everyone else's
     current claims, chipless players filling the gaps) and takes the best one — so if three
     teammates fight over chip 1 it moves up to make room;
  3. judges teammates' flop/turn claims with a draw-aware strength proxy;
  4. parameters tuned with `thegang.tune`.
