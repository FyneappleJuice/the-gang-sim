"""Rules engine for The Gang (cooperative Texas Hold'em).

Summary of the rules modelled here (base game, no Challenge/Specialist cards):

* 3-6 players (2 is allowed for experimentation). Everybody is on one team.
* Each *hand* (a "heist"): every player gets 2 hole cards, 5 community cards
  are revealed over the usual streets. There are four chip rounds:
      0 white  - preflop      (0 community cards visible)
      1 yellow - flop         (3 visible)
      2 orange - turn         (4 visible)
      3 red    - river        (5 visible)
  Each round has chips numbered 1..N (N = players). Each player ends the round
  holding exactly one chip: it is a public claim about how strong their final
  hand will be relative to the others (1 = weakest, N = strongest).
* Players may take a chip from the centre, swap their chip for one in the
  centre, or take a chip from another player (who must then take another).
  Talking about cards is forbidden, so chips are the only communication.
* Showdown: hands are revealed in order of red chips, lowest first. The heist
  succeeds if every hand is at least as strong as the one before it (ties may
  sit in either order). Success opens a vault, failure trips an alarm.
* The team wins on 3 vaults and loses on 3 alarms.

The real game has simultaneous, free-form chip grabbing. Here it is turned
into turns: players act in seat order, each choosing ``take(star)`` or
``pass``. A round ends when every player holds a chip and all players pass in
a row. To keep bots from looping forever, after ``max_actions_per_round``
takes the round is "locked": stealing and swapping stop, chipless players
take from the centre, and then the round closes.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Optional

from .cards import cards_str, evaluate, hand_name

ROUND_COLORS = ("white", "yellow", "orange", "red")
STREETS = ("preflop", "flop", "turn", "river")
BOARD_SIZES = (0, 3, 4, 5)
NUM_ROUNDS = 4


@dataclass(frozen=True)
class Action:
    kind: str  # "take" or "pass"
    star: Optional[int] = None

    def __str__(self) -> str:
        return "pass" if self.kind == "pass" else f"take {self.star}"


PASS = Action("pass")


def take(star: int) -> Action:
    return Action("take", star)


@dataclass(frozen=True)
class HandResult:
    hand_number: int
    success: bool
    holes: tuple  # per seat: (c1, c2)
    board: tuple
    scores: tuple  # per seat: evaluate() tuple
    chip_history: tuple  # per round: tuple of star per seat
    reveal_order: tuple  # seats ordered by red chip, lowest first
    first_failure: Optional[int]  # index into reveal_order where order broke

    def describe(self) -> str:
        lines = [f"Hand {self.hand_number + 1}: board {cards_str(self.board)}"]
        for i, seat in enumerate(self.reveal_order):
            chips = " ".join(f"{ROUND_COLORS[r][0].upper()}{self.chip_history[r][seat]}"
                             for r in range(NUM_ROUNDS))
            mark = "  <-- out of order" if self.first_failure == i else ""
            lines.append(f"  seat {seat} [{chips}] {cards_str(self.holes[seat])}  "
                         f"{hand_name(self.scores[seat])}{mark}")
        lines.append("  => " + ("VAULT CRACKED" if self.success else "ALARM"))
        return "\n".join(lines)


@dataclass(frozen=True)
class PlayerView:
    """Everything one seat is allowed to know when acting."""
    seat: int
    n_players: int
    hand_number: int
    round: int  # 0..3
    hole: tuple
    board: tuple  # visible community cards only
    chips: tuple  # chips[star-1] = seat holding it, or None if in centre
    my_chip: Optional[int]
    chip_history: tuple  # completed rounds of this hand: tuple of star per seat
    round_log: tuple  # (seat, Action) taken so far this round
    vaults: int
    alarms: int
    locked: bool
    legal_actions: tuple
    past_results: tuple  # HandResult of earlier hands (all revealed publicly)

    def holder(self, star: int) -> Optional[int]:
        return self.chips[star - 1]

    def centre_stars(self) -> list[int]:
        return [i + 1 for i, h in enumerate(self.chips) if h is None]


class GangGame:
    def __init__(
        self,
        n_players: int = 4,
        seed: Optional[int] = None,
        vaults_to_win: int = 3,
        alarms_to_lose: int = 3,
        max_actions_per_round: Optional[int] = None,
    ):
        if not 2 <= n_players <= 6:
            raise ValueError("The Gang supports 2-6 players (3-6 officially)")
        self.n = n_players
        self.rng = random.Random(seed)
        self.vaults_to_win = vaults_to_win
        self.alarms_to_lose = alarms_to_lose
        self.max_actions = max_actions_per_round or 6 * n_players
        self.vaults = 0
        self.alarms = 0
        self.hand_number = -1
        self.results: list[HandResult] = []
        self._new_hand()

    # ----------------------------------------------------------- state setup
    def _new_hand(self) -> None:
        self.hand_number += 1
        deck = list(range(52))
        self.rng.shuffle(deck)
        self.holes = tuple(tuple(sorted(deck[2 * i:2 * i + 2], reverse=True))
                           for i in range(self.n))
        self.full_board = tuple(deck[2 * self.n:2 * self.n + 5])
        self.chip_history: list[tuple] = []
        self.round = 0
        self._new_round()

    def _new_round(self) -> None:
        self.chips: list[Optional[int]] = [None] * self.n
        self.round_log: list[tuple] = []
        self.takes = 0
        self.consecutive_passes = 0
        # starting seat rotates between hands and rounds
        self.current = (self.hand_number + self.round) % self.n

    # ------------------------------------------------------------- queries
    @property
    def over(self) -> bool:
        return self.vaults >= self.vaults_to_win or self.alarms >= self.alarms_to_lose

    @property
    def won(self) -> bool:
        return self.vaults >= self.vaults_to_win

    @property
    def locked(self) -> bool:
        return self.takes >= self.max_actions

    def chip_of(self, seat: int) -> Optional[int]:
        for i, h in enumerate(self.chips):
            if h == seat:
                return i + 1
        return None

    def board(self) -> tuple:
        return self.full_board[:BOARD_SIZES[self.round]]

    def legal_actions(self, seat: Optional[int] = None) -> list[Action]:
        if self.over:
            return []
        seat = self.current if seat is None else seat
        mine = self.chip_of(seat)
        acts: list[Action] = []
        if mine is not None:
            acts.append(PASS)
        for i, h in enumerate(self.chips):
            if h == seat:
                continue
            if self.locked and (h is not None or mine is not None):
                continue
            acts.append(take(i + 1))
        return acts

    def view(self, seat: Optional[int] = None) -> PlayerView:
        seat = self.current if seat is None else seat
        return PlayerView(
            seat=seat,
            n_players=self.n,
            hand_number=self.hand_number,
            round=self.round,
            hole=self.holes[seat],
            board=self.board(),
            chips=tuple(self.chips),
            my_chip=self.chip_of(seat),
            chip_history=tuple(self.chip_history),
            round_log=tuple(self.round_log),
            vaults=self.vaults,
            alarms=self.alarms,
            locked=self.locked,
            legal_actions=tuple(self.legal_actions(seat)),
            past_results=tuple(self.results),
        )

    # ------------------------------------------------------------- actions
    def apply(self, action: Action) -> Optional[HandResult]:
        """Apply the current player's action. Returns a HandResult if a hand ended."""
        if self.over:
            raise RuntimeError("game is over")
        seat = self.current
        if action not in self.legal_actions(seat):
            raise ValueError(f"illegal action {action} for seat {seat}")
        self.round_log.append((seat, action))

        if action.kind == "pass":
            self.consecutive_passes += 1
        else:
            mine = self.chip_of(seat)
            if mine is not None:
                self.chips[mine - 1] = None  # swap: own chip returns to centre
            self.chips[action.star - 1] = seat  # from centre, or stolen
            self.takes += 1
            self.consecutive_passes = 0

        if self.consecutive_passes >= self.n and all(h is not None for h in self.chips):
            return self._end_round()
        self.current = (self.current + 1) % self.n
        return None

    def _end_round(self) -> Optional[HandResult]:
        per_seat = [0] * self.n
        for i, h in enumerate(self.chips):
            per_seat[h] = i + 1
        self.chip_history.append(tuple(per_seat))
        if self.round < NUM_ROUNDS - 1:
            self.round += 1
            self._new_round()
            return None
        return self._showdown()

    def _showdown(self) -> HandResult:
        red = self.chip_history[-1]
        order = tuple(sorted(range(self.n), key=lambda s: red[s]))
        scores = tuple(evaluate(self.holes[s] + self.full_board) for s in range(self.n))
        first_failure = None
        for i in range(1, self.n):
            if scores[order[i]] < scores[order[i - 1]]:
                first_failure = i
                break
        success = first_failure is None
        result = HandResult(
            hand_number=self.hand_number,
            success=success,
            holes=self.holes,
            board=self.full_board,
            scores=scores,
            chip_history=tuple(self.chip_history),
            reveal_order=order,
            first_failure=first_failure,
        )
        self.results.append(result)
        if success:
            self.vaults += 1
        else:
            self.alarms += 1
        if not self.over:
            self._new_hand()
        return result
