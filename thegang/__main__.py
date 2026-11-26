"""CLI: python -m thegang --players 4 --agent heuristic --games 200"""

import argparse
import os

from .agents import AGENTS, make_agent
from .sim import play_game, run


def main() -> None:
    p = argparse.ArgumentParser(prog="thegang", description="Simulate The Gang")
    p.add_argument("--players", "-n", type=int, default=4)
    p.add_argument("--agent", "-a", default="heuristic",
                   help=f"agent for every seat, or comma list per seat ({', '.join(AGENTS)})")
    p.add_argument("--games", "-g", type=int, default=100)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--workers", "-w", type=int, default=os.cpu_count() or 1)
    p.add_argument("--show", action="store_true", help="play one game and print every move")
    a = p.parse_args()

    names = a.agent.split(",")
    if len(names) == 1:
        names = names * a.players
    if len(names) != a.players:
        p.error("need one agent per seat")

    if a.show:
        agents = [make_agent(n, seed=a.seed * 101 + i) for i, n in enumerate(names)]
        game = play_game(agents, seed=a.seed, verbose=True)
        print("TEAM WINS" if game.won else "TEAM LOSES")
        return

    stats = run(names, a.games, seed=a.seed, workers=a.workers)
    print(f"{a.players} players, seats: {','.join(names)}")
    print(stats.summary())


if __name__ == "__main__":
    main()
