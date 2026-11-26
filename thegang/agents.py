"""Agents. Subclass `Agent` and implement `act(view) -> Action`."""

from __future__ import annotations

import math
import random
from typing import Optional

from .cards import evaluate
from .engine import BOARD_SIZES, PASS, Action, PlayerView, take


class Agent:
    name = "agent"

    def act(self, view: PlayerView) -> Action:
        raise NotImplementedError

    def on_hand_end(self, result) -> None:
        """Optional hook: called for every seat after each showdown."""


class RandomAgent(Agent):
    """Uniformly random legal moves, but passes with probability `pass_prob` when it can."""
    name = "random"

    def __init__(self, pass_prob: float = 0.8, seed: Optional[int] = None):
        self.pass_prob = pass_prob
        self.rng = random.Random(seed)

    def act(self, view: PlayerView) -> Action:
        legal = list(view.legal_actions)
        if PASS in legal and self.rng.random() < self.pass_prob:
            return PASS
        takes = [a for a in legal if a.kind == "take"]
        return self.rng.choice(takes) if takes else PASS


# --------------------------------------------------------------- equity

_preflop_cache: dict = {}


def expected_beaten(hole, board, n_players: int, sims: int, rng: random.Random) -> float:
    """Monte Carlo: expected number of the other n-1 players whose final hand
    this hole+board beats (ties count half), with everyone else's cards and
    the rest of the board drawn at random."""
    if not board:
        hi, lo = sorted(((c >> 2) for c in hole), reverse=True)
        key = (hi, lo, (hole[0] & 3) == (hole[1] & 3), n_players)
        if key not in _preflop_cache:
            _preflop_cache[key] = _mc(hole, board, n_players, max(sims, 1500), rng)
        return _preflop_cache[key]
    return _mc(hole, board, n_players, sims, rng)


def _mc(hole, board, n_players, sims, rng) -> float:
    known = set(hole) | set(board)
    stub = [c for c in range(52) if c not in known]
    need_board = 5 - len(board)
    need = need_board + 2 * (n_players - 1)
    total = 0.0
    hole = tuple(hole)
    board = tuple(board)
    for _ in range(sims):
        draw = rng.sample(stub, need)
        full = board + tuple(draw[:need_board])
        me = evaluate(hole + full)
        k = need_board
        for _ in range(n_players - 1):
            other = evaluate((draw[k], draw[k + 1]) + full)
            k += 2
            if me > other:
                total += 1.0
            elif me == other:
                total += 0.5
    return total / sims


class HeuristicAgent(Agent):
    """Estimates where its hand should rank and grabs the matching chip.

    position = 1 + E[# players beaten], a float in [1, N]. It prefers the
    chip closest to that position, swaps with the centre when a closer chip is
    free, and steals only when it would otherwise be `steal_gap` or more away
    (at most `max_steals` per round). It ignores what other players' chips
    imply; see `ChipReaderAgent` for a version that uses that information.
    """
    name = "heuristic"

    def __init__(self, sims: int = 150, steal_gap: float = 1.5, max_steals: int = 1,
                 seed: Optional[int] = None):
        self.sims = sims
        self.steal_gap = steal_gap
        self.max_steals = max_steals
        self.rng = random.Random(seed)
        self._cache_key = None
        self._pos = 0.0
        self._steals = 0

    def position(self, view: PlayerView) -> float:
        key = (view.hand_number, view.round, view.seat)
        if key != self._cache_key:
            self._cache_key = key
            self._steals = 0
            self._pos = self.estimate_position(view)
        return self._pos

    def estimate_position(self, view: PlayerView) -> float:
        return 1.0 + expected_beaten(view.hole, view.board, view.n_players, self.sims, self.rng)

    def act(self, view: PlayerView) -> Action:
        pos = self.position(view)
        legal = set(view.legal_actions)
        mine = view.my_chip
        n = view.n_players
        target = min(range(1, n + 1), key=lambda s: (abs(s - pos), s))
        centre = [s for s in view.centre_stars() if take(s) in legal]

        def dist(s):
            return abs(s - pos)

        best_centre = min(centre, key=lambda s: (dist(s), s)) if centre else None

        if mine is None:
            if (take(target) in legal and view.holder(target) is not None
                    and self._steals < self.max_steals
                    and (best_centre is None or dist(best_centre) >= self.steal_gap)):
                self._steals += 1
                return take(target)
            return take(best_centre)

        if mine == target:
            return PASS
        if best_centre is not None and dist(best_centre) < dist(mine) - 1e-9:
            return take(best_centre)
        if (take(target) in legal and view.holder(target) is not None
                and self._steals < self.max_steals and dist(mine) >= self.steal_gap):
            self._steals += 1
            return take(target)
        return PASS


def preflop_score(hole) -> float:
    """Cheap preflop strength proxy (higher is better)."""
    a, b = sorted(((c >> 2) + 2 for c in hole), reverse=True)
    if a == b:
        return 30 + 2 * a
    score = a + 0.5 * b
    if (hole[0] & 3) == (hole[1] & 3):
        score += 2
    gap = a - b
    if gap <= 4:
        score += 2 - 0.5 * (gap - 1)
    return score


def _proxy(hole, board_r):
    return preflop_score(hole) if len(board_r) < 3 else evaluate(tuple(hole) + tuple(board_r))


class ChipReaderAgent(HeuristicAgent):
    """Heuristic agent that also reads teammates' chips.

    Same Monte Carlo as `HeuristicAgent`, but each sampled deal is weighted by
    how consistent it is with the chips everyone claimed in the completed
    rounds of this hand: for every pair of players (involving a teammate)
    whose claimed order disagrees with the order of their hand strength at
    that street in the sample, the weight is multiplied by exp(-beta).
    Strength at a street is a proxy: a preflop formula, or the made hand.
    """
    name = "chipreader"

    def __init__(self, beta: float = 1.0, sims: int = 250, **kw):
        super().__init__(sims=sims, **kw)
        self.beta = beta

    def estimate_position(self, view: PlayerView) -> float:
        if not view.chip_history:
            return super().estimate_position(view)
        n = view.n_players
        me_seat = view.seat
        others = [s for s in range(n) if s != me_seat]
        hole = tuple(view.hole)
        board = tuple(view.board)
        known = set(hole) | set(board)
        stub = [c for c in range(52) if c not in known]
        need_board = 5 - len(board)
        need = need_board + 2 * len(others)
        penalties = [math.exp(-self.beta * k) for k in range(n * n)]
        boards_r = [board[:BOARD_SIZES[r]] for r in range(len(view.chip_history))]
        my_proxy = [_proxy(hole, b) for b in boards_r]
        wsum = 0.0
        total = 0.0
        for _ in range(self.sims):
            draw = self.rng.sample(stub, need)
            full = board + tuple(draw[:need_board])
            holes = {me_seat: hole}
            k = need_board
            for s in others:
                holes[s] = (draw[k], draw[k + 1])
                k += 2
            disc = 0
            for r, claims in enumerate(view.chip_history):
                prox = {s: _proxy(holes[s], boards_r[r]) for s in others}
                prox[me_seat] = my_proxy[r]
                for i in range(n):
                    for j in range(i + 1, n):
                        ci, cj = claims[i], claims[j]
                        pi, pj = prox[i], prox[j]
                        if (ci < cj and pi > pj) or (ci > cj and pi < pj):
                            disc += 1
            w = penalties[min(disc, len(penalties) - 1)]
            me = evaluate(hole + full)
            beaten = 0.0
            for s in others:
                o = evaluate(holes[s] + full)
                if me > o:
                    beaten += 1.0
                elif me == o:
                    beaten += 0.5
            total += w * beaten
            wsum += w
        return 1.0 + (total / wsum if wsum > 0 else (n - 1) / 2)


AGENTS = {
    "random": RandomAgent,
    "heuristic": HeuristicAgent,
    "chipreader": ChipReaderAgent,
}

from .tablereader import TableReaderAgent  # noqa: E402  (imports from this module)

AGENTS["tablereader"] = TableReaderAgent


def make_agent(name: str, seed: Optional[int] = None, **kw) -> Agent:
    cls = AGENTS[name]
    return cls(seed=seed, **kw)
