"""Cards and a fast-enough pure-Python 7-card poker hand evaluator.

Cards are ints 0..51: rank = c >> 2 (0 = deuce .. 12 = ace), suit = c & 3.
`evaluate(cards)` works on 5..7 cards and returns a tuple; larger tuples are
stronger hands, equal tuples are exact ties.
"""

from __future__ import annotations

from typing import Iterable, Sequence

RANKS = "23456789TJQKA"
SUITS = "cdhs"
HAND_NAMES = [
    "High card",
    "Pair",
    "Two pair",
    "Three of a kind",
    "Straight",
    "Flush",
    "Full house",
    "Four of a kind",
    "Straight flush",
]

FULL_DECK = tuple(range(52))


def card_rank(c: int) -> int:
    """2..14 (ace = 14)."""
    return (c >> 2) + 2


def card_suit(c: int) -> int:
    return c & 3


def card_str(c: int) -> str:
    return RANKS[c >> 2] + SUITS[c & 3]


def cards_str(cards: Iterable[int]) -> str:
    return " ".join(card_str(c) for c in cards)


def parse_card(s: str) -> int:
    s = s.strip()
    if len(s) != 2 or s[0].upper() not in RANKS or s[1].lower() not in SUITS:
        raise ValueError(f"bad card: {s!r}")
    return RANKS.index(s[0].upper()) * 4 + SUITS.index(s[1].lower())


def parse_cards(s: str) -> list[int]:
    return [parse_card(t) for t in s.split()]


def _straight_high(mask: int) -> int:
    """Highest straight in a rank bitmask (bit r set for rank r in 2..14), or 0."""
    if mask & (1 << 14):
        mask |= 1 << 1  # ace plays low in A-2-3-4-5
    for high in range(14, 4, -1):
        m = 0b11111 << (high - 4)
        if mask & m == m:
            return high
    return 0


def evaluate(cards: Sequence[int]) -> tuple:
    """Best 5-card hand value from 5..7 cards, as a comparable tuple."""
    counts = [0] * 15
    suit_masks = [0, 0, 0, 0]
    suit_counts = [0, 0, 0, 0]
    mask = 0
    for c in cards:
        r = (c >> 2) + 2
        s = c & 3
        counts[r] += 1
        suit_masks[s] |= 1 << r
        suit_counts[s] += 1
        mask |= 1 << r

    flush = None
    for s in range(4):
        if suit_counts[s] >= 5:
            sm = suit_masks[s]
            sf = _straight_high(sm)
            if sf:
                return (8, sf)
            flush = (5, *[r for r in range(14, 1, -1) if sm >> r & 1][:5])
            break

    # (count, rank) for present ranks, best first
    groups = sorted(((counts[r], r) for r in range(14, 1, -1) if counts[r]), reverse=True)
    top_count, top_rank = groups[0]

    if top_count == 4:
        kicker = max(r for _, r in groups[1:])
        return (7, top_rank, kicker)
    if top_count == 3 and len(groups) > 1 and groups[1][0] >= 2:
        return (6, top_rank, groups[1][1])
    if flush:
        return flush
    st = _straight_high(mask)
    if st:
        return (4, st)
    if top_count == 3:
        kick = sorted((r for _, r in groups[1:]), reverse=True)[:2]
        return (3, top_rank, *kick)
    if top_count == 2 and groups[1][0] == 2:
        p1, p2 = top_rank, groups[1][1]
        kicker = max(r for _, r in groups[2:])
        return (2, p1, p2, kicker)
    if top_count == 2:
        kick = sorted((r for _, r in groups[1:]), reverse=True)[:3]
        return (1, top_rank, *kick)
    return (0, *sorted((r for _, r in groups), reverse=True)[:5])


def hand_name(score: tuple) -> str:
    return HAND_NAMES[score[0]]
