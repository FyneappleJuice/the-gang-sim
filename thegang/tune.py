"""Random-search tuner for agent parameters.

    python -m thegang.tune --agent tablereader -n 4 --configs 16 --games 120

Every config plays the same seeds (common random numbers), so differences are
mostly due to the parameters. Score = per-hand success rate (much less noisy
than game win rate; win rate rises monotonically with it). The top configs
are then re-checked on fresh seeds.
"""

from __future__ import annotations

import argparse
import json
import os
import random

from .agents import AGENTS
from .sim import run

SPACES = {
    "tablereader": {
        "beta": [0.5, 1.0, 1.5, 2.0, 3.0, 4.0],
        "history_weight": [0.3, 0.6, 1.0, 1.5],
        "displaced_weight": [0.0, 0.25, 0.5, 1.0],
        "gap_cap": [1, 2, 3],
        "switch_margin": [0.0, 0.03, 0.06, 0.1, 0.15],
        "steal_margin": [0.0, 0.05, 0.1, 0.2, 0.3],
        "max_steals": [0, 1, 2, 3],
    },
    "chipreader": {
        "beta": [0.5, 1.0, 1.5, 2.0, 3.0],
        "steal_gap": [1.0, 1.5, 2.0, 3.0],
        "max_steals": [0, 1, 2],
    },
}


def evaluate_config(agent, n, params, games, seed, workers):
    s = run([agent] * n, games, seed=seed, workers=workers, agent_kwargs=params)
    return s.hand_successes / s.hands, s.wins / s.games, s


def main():
    p = argparse.ArgumentParser(prog="thegang.tune")
    p.add_argument("--agent", default="tablereader", choices=sorted(SPACES))
    p.add_argument("--players", "-n", type=int, default=4)
    p.add_argument("--configs", type=int, default=16)
    p.add_argument("--games", type=int, default=120)
    p.add_argument("--recheck", type=int, default=3, help="re-test this many top configs")
    p.add_argument("--recheck-games", type=int, default=300)
    p.add_argument("--seed", type=int, default=20000)
    p.add_argument("--workers", "-w", type=int, default=os.cpu_count() or 1)
    p.add_argument("--out", default="tuning_results.json")
    a = p.parse_args()

    space = SPACES[a.agent]
    rng = random.Random(a.seed)
    defaults = {}
    configs = [defaults]  # always include the current defaults
    while len(configs) < a.configs:
        c = {k: rng.choice(v) for k, v in space.items()}
        if c not in configs:
            configs.append(c)

    results = []
    for i, c in enumerate(configs):
        hand, win, _ = evaluate_config(a.agent, a.players, c, a.games, a.seed, a.workers)
        results.append({"params": c, "hand": hand, "win": win})
        print(f"[{i + 1}/{len(configs)}] hand {hand:.1%} win {win:.1%}  {c or 'defaults'}", flush=True)

    results.sort(key=lambda r: -r["hand"])
    print("\nRe-checking top configs on fresh seeds:")
    final = []
    for r in results[:a.recheck] + ([] if any(not r["params"] for r in results[:a.recheck])
                                    else [x for x in results if not x["params"]]):
        hand, win, s = evaluate_config(a.agent, a.players, r["params"], a.recheck_games,
                                       a.seed + 100000, a.workers)
        final.append({"params": r["params"], "hand": hand, "win": win})
        print(f"  hand {hand:.1%} win {win:.1%}  {r['params'] or 'defaults'}", flush=True)
    final.sort(key=lambda r: -r["hand"])
    print(f"\nBest: {final[0]['params'] or 'defaults'}")
    with open(a.out, "w") as f:
        json.dump({"agent": a.agent, "players": a.players, "search": results, "recheck": final}, f,
                  indent=1)


if __name__ == "__main__":
    main()
