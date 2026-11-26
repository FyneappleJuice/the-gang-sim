"""Single-seat environment for RL / custom-policy experiments.

One seat is controlled by the caller; the other seats are played by teammate
agents. Every call to `step` advances the game until it is the controlled
seat's turn again (or the game ends).

    env = GangEnv(n_players=4, seed=0)
    view = env.reset()
    while not env.done:
        view, reward, done, info = env.step(my_policy(view))

Reward: +1 when a vault is cracked, -1 when an alarm trips, plus
`win_bonus` / -`win_bonus` when the game ends. `info["results"]` holds the
HandResults finished during that step.
"""

from __future__ import annotations

from typing import Callable, Optional

from .agents import Agent, HeuristicAgent
from .engine import Action, GangGame, PlayerView


class GangEnv:
    def __init__(
        self,
        n_players: int = 4,
        seat: int = 0,
        teammate_factory: Optional[Callable[[int], Agent]] = None,
        seed: Optional[int] = None,
        win_bonus: float = 0.0,
        **game_kwargs,
    ):
        self.n = n_players
        self.seat = seat
        self.factory = teammate_factory or (lambda s: HeuristicAgent(seed=s))
        self.seed = seed
        self.win_bonus = win_bonus
        self.game_kwargs = game_kwargs
        self._episode = 0
        self.game: Optional[GangGame] = None

    @property
    def done(self) -> bool:
        return self.game is None or self.game.over

    def reset(self) -> Optional[PlayerView]:
        seed = None if self.seed is None else self.seed + self._episode
        self._episode += 1
        self.game = GangGame(self.n, seed=seed, **self.game_kwargs)
        self.teammates = {s: self.factory(s if seed is None else seed * 31 + s)
                          for s in range(self.n) if s != self.seat}
        self._advance()
        return self.game.view(self.seat) if not self.done else None

    def legal_actions(self) -> list[Action]:
        return self.game.legal_actions(self.seat)

    def step(self, action: Action):
        if self.done:
            raise RuntimeError("call reset() first")
        if self.game.current != self.seat:
            raise RuntimeError("not the controlled seat's turn")
        results = []
        r = self.game.apply(action)
        if r is not None:
            results.append(r)
        results += self._advance()
        reward = sum(1.0 if x.success else -1.0 for x in results)
        if self.game.over and self.win_bonus:
            reward += self.win_bonus if self.game.won else -self.win_bonus
        view = None if self.done else self.game.view(self.seat)
        return view, reward, self.done, {"results": results}

    def _advance(self):
        results = []
        while not self.game.over and self.game.current != self.seat:
            cur = self.game.current
            r = self.game.apply(self.teammates[cur].act(self.game.view(cur)))
            if r is not None:
                results.append(r)
                for a in self.teammates.values():
                    a.on_hand_end(r)
        return results
