"""TableReaderAgent: reads every chip move and places itself among its teammates.

Compared to ChipReaderAgent it:

1. Uses *live* evidence, not only completed rounds. Every chip currently held
   is a claim, and a player whose chip was just stolen still "claims"
   roughly the chip they lost (weaker evidence). The belief is recomputed
   every time the chips change, not once per round.
2. Chooses its chip by asking, for every chip it could take: "if everyone else
   keeps their current claim and the chipless players fill the remaining
   chips correctly, how likely is it that I am placed correctly?" Being
   placed correctly at chip s means: beating everyone holding a chip below
   s, losing to everyone above, and beating exactly as many chipless players
   as there are free chips below s. It takes the chip with the best chance.
   So if three teammates fight over chip 1, it infers it should sit above
   all of them and moves up to make room.
3. Judges teammates' claims at the flop/turn by a draw-aware strength proxy
   (made hand + flush/straight-draw bonus), not the made hand alone.

Efficiency: once per (hand, round) it samples `sims` possible deals
(unknown hole cards + rest of board) and precomputes each player's
street-strength proxies and how its own final hand compares with each
teammate's. Each decision only reweights those samples, which is cheap.

Weighting: for every pair of players whose claims differ, a sample whose
street-strength order disagrees with the claimed order is multiplied by
exp(-beta * w), where w grows with the gap between the two claimed chips
(capped at `gap_cap`). Earlier rounds are scaled by `history_weight`, claims
of displaced players by `displaced_weight`.
"""

from __future__ import annotations

import math
from typing import Optional

from .agents import HeuristicAgent, preflop_score
from .cards import evaluate
from .engine import BOARD_SIZES, PASS, Action, PlayerView, take

_POW = [15.0 ** -(i + 1) for i in range(6)]
# 5-rank straight windows as rank bitmasks (bit r for rank r, ace also at bit 1)
_WINDOWS = [0b11111 << (high - 4) for high in range(5, 15)]


def street_strength(hole, board, flush_bonus=(0.9, 0.5), straight_bonus=(0.6, 0.35),
                    gutshot_bonus=(0.25, 0.12)) -> float:
    """Scalar strength of hole+board at a street, comparable between players
    on the same street. Preflop uses a simple formula; post-flop it is the
    hand category plus kicker fractions plus a bonus for live draws."""
    if len(board) < 3:
        return preflop_score(hole)
    cards = tuple(hole) + tuple(board)
    sc = evaluate(cards)
    v = sc[0]
    for i, r in enumerate(sc[1:]):
        v += r * _POW[i]
    to_come = 5 - len(board)
    if to_come and sc[0] < 4:
        k = 0 if to_come == 2 else 1
        # flush draw that uses at least one hole card
        for s in {hole[0] & 3, hole[1] & 3}:
            if sum(1 for c in cards if c & 3 == s) == 4:
                v += flush_bonus[k]
                break
        mask = 0
        for c in cards:
            mask |= 1 << ((c >> 2) + 2)
        if mask & (1 << 14):
            mask |= 2
        outs = 0
        for w in _WINDOWS:
            m = mask & w
            if m != w and m.bit_count() == 4:
                outs |= w & ~m
        n_outs = outs.bit_count()
        if n_outs >= 2:
            v += straight_bonus[k]
        elif n_outs == 1:
            v += gutshot_bonus[k]
    return v


class TableReaderAgent(HeuristicAgent):
    name = "tablereader"

    def __init__(self, sims: int = 250, beta: float = 2.0, history_weight: float = 1.0,
                 displaced_weight: float = 0.5, gap_cap: int = 2, switch_margin: float = 0.03,
                 steal_margin: float = 0.1, max_steals: int = 2, seed: Optional[int] = None):
        super().__init__(sims=sims, max_steals=max_steals, seed=seed)
        self.beta = beta
        self.history_weight = history_weight
        self.displaced_weight = displaced_weight
        self.gap_cap = gap_cap
        self.switch_margin = switch_margin
        self.steal_margin = steal_margin
        self._round_key = None
        self._state_key = None
        self._weights: list[float] = []

    # ---------------------------------------------------------- sampling
    def _prepare_round(self, view: PlayerView) -> None:
        n = view.n_players
        me = view.seat
        others = [s for s in range(n) if s != me]
        hole = tuple(view.hole)
        board = tuple(view.board)
        known = set(hole) | set(board)
        stub = [c for c in range(52) if c not in known]
        need_board = 5 - len(board)
        need = need_board + 2 * len(others)
        boards_r = [board[:BOARD_SIZES[r]] for r in range(len(view.chip_history))]
        my_hist = [street_strength(hole, b) for b in boards_r]
        my_cur = street_strength(hole, board)
        cap = self.gap_cap
        hist_pairs = [[(i, j, 1 if c[i] > c[j] else -1, min(abs(c[i] - c[j]), cap))
                       for i in range(n) for j in range(i + 1, n)]
                      for c in view.chip_history]

        self._cur = []  # per sample: street strength per seat (this street)
        self._cmp = []  # per sample: +1/0/-1 = my final hand beats/ties/loses to seat j
        self._ranks = []  # per sample: final rank per seat (1..n, ties averaged)
        self._base = []  # per sample: weighted discordance with completed rounds
        for _ in range(self.sims):
            draw = self.rng.sample(stub, need)
            full = board + tuple(draw[:need_board])
            holes = [None] * n
            holes[me] = hole
            k = need_board
            for s in others:
                holes[s] = (draw[k], draw[k + 1])
                k += 2
            finals = [evaluate(holes[s] + full) for s in range(n)]
            mine = finals[me]
            self._cmp.append([(mine > f) - (mine < f) for f in finals])
            self._ranks.append([1.0 + sum(1.0 if finals[s] > finals[t] else 0.5 if finals[s] == finals[t]
                                          else 0.0 for t in range(n) if t != s) for s in range(n)])
            disc = 0.0
            for r, pairs in enumerate(hist_pairs):
                st = [my_hist[r] if s == me else street_strength(holes[s], boards_r[r])
                      for s in range(n)]
                for i, j, sign, w in pairs:
                    if (st[i] - st[j]) * sign < 0:
                        disc += w
            self._cur.append([my_cur if s == me else street_strength(holes[s], board)
                              for s in range(n)])
            self._base.append(disc)

    # ---------------------------------------------------------- beliefs
    def _claims(self, view: PlayerView) -> dict:
        """seat -> (star, evidence weight) for this round's live claims."""
        held = {h: i + 1 for i, h in enumerate(view.chips) if h is not None}
        last_held: dict[int, int] = {}  # chip a displaced player lost most recently
        owner: list = [None] * view.n_players
        for seat, act in view.round_log:
            if act.kind != "take":
                continue
            prev = owner[act.star - 1]
            if prev is not None:
                last_held[prev] = act.star
            for i, o in enumerate(owner):
                if o == seat:
                    owner[i] = None
            owner[act.star - 1] = seat
        claims = {}
        for s in range(view.n_players):
            if s in held:
                claims[s] = (held[s], 1.0)
            elif s in last_held:
                claims[s] = (last_held[s], self.displaced_weight)
        return claims

    def weights(self, view: PlayerView) -> list[float]:
        """Normalised weights of this round's samples given all chip evidence."""
        rkey = (view.hand_number, view.round, view.seat)
        if rkey != self._round_key:
            self._round_key = rkey
            self._steals = 0
            self._prepare_round(view)
            self._state_key = None
        skey = (view.chips, len(view.round_log))
        if skey == self._state_key:
            return self._weights
        self._state_key = skey

        claims = self._claims(view)
        seats = list(claims)
        pairs = []
        for a in range(len(seats)):
            for b in range(a + 1, len(seats)):
                i, j = seats[a], seats[b]
                (ci, wi), (cj, wj) = claims[i], claims[j]
                if ci != cj:
                    pairs.append((i, j, 1 if ci > cj else -1,
                                  min(wi, wj) * min(abs(ci - cj), self.gap_cap)))
        hw = self.history_weight
        logws = []
        for base, cur in zip(self._base, self._cur):
            disc = hw * base
            for i, j, sign, w in pairs:
                if (cur[i] - cur[j]) * sign < 0:
                    disc += w
            logws.append(-self.beta * disc)
        m = max(logws)
        ws = [math.exp(l - m) for l in logws]
        tot = sum(ws)
        self._weights = [w / tot for w in ws]
        return self._weights

    def expected_ranks(self, view: PlayerView) -> list[float]:
        ws = self.weights(view)
        n = view.n_players
        return [sum(w * r[s] for w, r in zip(ws, self._ranks)) for s in range(n)]

    def placement_scores(self, view: PlayerView) -> dict:
        """star -> P(I am correctly placed at that star | others' current claims)."""
        ws = self.weights(view)
        n = view.n_players
        me = view.seat
        legal = set(view.legal_actions)
        cands = [s for s in range(1, n + 1) if take(s) in legal or s == view.my_chip]
        held = {h: i + 1 for i, h in enumerate(view.chips) if h is not None and h != me}
        scores = {}
        for s in cands:
            fixed = [(j, c) for j, c in held.items() if c != s]  # holder of s is displaced
            below = [j for j, c in fixed if c < s]
            above = [j for j, c in fixed if c > s]
            taken = {c for _, c in fixed}
            unclaimed = [j for j in range(n) if j != me and j not in dict(fixed)]
            k = sum(1 for c in range(1, s) if c not in taken)  # free chips below s
            p = 0.0
            for w, cmp in zip(ws, self._cmp):
                if w < 1e-9:
                    continue
                if any(cmp[j] < 0 for j in below) or any(cmp[j] > 0 for j in above):
                    continue
                lo = hi = 0
                for j in unclaimed:
                    c = cmp[j]
                    if c > 0:
                        lo += 1
                        hi += 1
                    elif c == 0:
                        hi += 1
                if lo <= k <= hi:
                    p += w
            scores[s] = p
        return scores

    # ---------------------------------------------------------- policy
    def act(self, view: PlayerView) -> Action:
        legal = set(view.legal_actions)
        if legal == {PASS}:
            return PASS
        scores = self.placement_scores(view)
        mine = view.my_chip
        # tie-break towards my plain expected rank
        exp_me = sum(w * r[view.seat] for w, r in zip(self._weights, self._ranks))

        def key(s):
            return (scores[s], -abs(s - exp_me))

        free = [s for s in scores if s != mine and view.holder(s) is None]
        steals = [s for s in scores if s != mine and view.holder(s) is not None]
        best_free = max(free, key=key) if free else None
        best_steal = max(steals, key=key) if steals else None
        baseline = scores[mine] if mine is not None else (scores[best_free] if best_free else -1)

        if (best_steal is not None and self._steals < self.max_steals
                and scores[best_steal] >= baseline + self.steal_margin
                and (best_free is None or scores[best_steal] >= scores[best_free] + self.steal_margin)):
            self._steals += 1
            return take(best_steal)
        if mine is None:
            return take(best_free) if best_free is not None else take(best_steal)
        if best_free is not None and scores[best_free] > scores[mine] + self.switch_margin:
            return take(best_free)
        return PASS
