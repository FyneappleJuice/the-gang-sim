"""Play full games with a set of agents and collect statistics."""

from __future__ import annotations

import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from typing import Optional, Sequence

from .agents import Agent, make_agent
from .engine import GangGame, HandResult


def play_game(agents: Sequence[Agent], seed: Optional[int] = None, verbose: bool = False,
              **game_kwargs) -> GangGame:
    game = GangGame(len(agents), seed=seed, **game_kwargs)
    while not game.over:
        seat = game.current
        view = game.view(seat)
        action = agents[seat].act(view)
        if verbose:
            print(f"  [{game.hand_number + 1}.{'WYOR'[game.round]}] seat {seat} "
                  f"(chip {view.my_chip}): {action}")
        result = game.apply(action)
        if result is not None:
            for a in agents:
                a.on_hand_end(result)
            if verbose:
                print(result.describe())
                print(f"  vaults {game.vaults} / alarms {game.alarms}\n")
    return game


@dataclass
class Stats:
    games: int = 0
    wins: int = 0
    hands: int = 0
    hand_successes: int = 0
    seconds: float = 0.0

    def add(self, game: GangGame) -> None:
        self.games += 1
        self.wins += game.won
        for r in game.results:
            self.hands += 1
            self.hand_successes += r.success

    def merge(self, o: "Stats") -> None:
        self.games += o.games
        self.wins += o.wins
        self.hands += o.hands
        self.hand_successes += o.hand_successes
        self.seconds += o.seconds

    def summary(self) -> str:
        import math
        p = self.wins / self.games if self.games else 0
        se = math.sqrt(p * (1 - p) / self.games) if self.games else 0
        hp = self.hand_successes / self.hands if self.hands else 0
        return (f"games {self.games}: win rate {p:.1%} (±{1.96 * se:.1%}), "
                f"per-hand success {hp:.1%} over {self.hands} hands, "
                f"{self.seconds / max(self.games, 1) * 1000:.0f} cpu-ms/game")


def _run_batch(args) -> Stats:
    agent_names, n_games, seed0, game_kwargs, agent_kwargs = args
    stats = Stats()
    t = time.perf_counter()
    for g in range(n_games):
        seed = seed0 + g
        agents = [make_agent(name, seed=seed * 101 + i, **agent_kwargs) for i, name in enumerate(agent_names)]
        stats.add(play_game(agents, seed=seed, **game_kwargs))
    stats.seconds = time.perf_counter() - t
    return stats


def run(agent_names: Sequence[str], games: int, seed: int = 0, workers: int = 1,
        agent_kwargs: Optional[dict] = None, **game_kwargs) -> Stats:
    """Play `games` games; agent_names[i] controls seat i. `agent_kwargs` go to every agent."""
    agent_kwargs = agent_kwargs or {}
    if workers <= 1:
        return _run_batch((list(agent_names), games, seed, game_kwargs, agent_kwargs))
    chunk = -(-games // workers)
    jobs = [(list(agent_names), min(chunk, games - i * chunk), seed + i * chunk, game_kwargs,
             agent_kwargs)
            for i in range(workers) if games - i * chunk > 0]
    total = Stats()
    with ProcessPoolExecutor(workers) as ex:
        for s in ex.map(_run_batch, jobs):
            total.merge(s)
    return total
