import itertools
import random
import unittest

from thegang import (TableReaderAgent, PASS, GangEnv, GangGame, HeuristicAgent, RandomAgent, ChipReaderAgent,
                     evaluate, parse_cards, play_game, take)


def ev(s):
    return evaluate(parse_cards(s))


class EvaluatorTests(unittest.TestCase):
    def test_categories(self):
        cases = [
            ("As Ks Qs Js Ts 2d 3c", 8),
            ("5h 4h 3h 2h Ah Kd Kc", 8),   # steel wheel beats the trips/pair
            ("9c 9d 9h 9s 2c 3d 4h", 7),
            ("Kc Kd Kh 2s 2c 3d 4h", 6),
            ("Kc Kd Kh 2s 2c 2d 4h", 6),
            ("Ac 9c 7c 4c 2c Kd Kh", 5),
            ("Ah 2c 3d 4s 5h Kd Qh", 4),
            ("7c 7d 7h Ks 2c 3d 9h", 3),
            ("7c 7d 2h 2s Kc Kd 9h", 2),
            ("7c 7d 2h 3s Kc Qd 9h", 1),
            ("7c 8d 2h 3s Kc Qd 4h", 0),
        ]
        for cards, cat in cases:
            with self.subTest(cards=cards):
                self.assertEqual(ev(cards)[0], cat)

    def test_orderings(self):
        self.assertGreater(ev("6h 2c 3d 4s 5h Kd Qh"), ev("Ah 2c 3d 4s 5h Kd Qh"))  # 6-high > wheel
        self.assertGreater(ev("Kc Kd Kh 3s 3c 2d 4h"), ev("Kc Kd Kh 2s 2c 3d 4h"))
        self.assertGreater(ev("7c 7d 2h 2s Ac Qd 9h"), ev("7c 7d 2h 2s Kc Qd 9h"))  # kicker
        self.assertEqual(ev("As Kd 2c 3c 7h 9s Jd"), ev("Ac Kh 2c 3c 7h 9s Jd"))
        # three pairs: best two pairs + best kicker
        self.assertEqual(ev("Ac Ad Kc Kd Qc Qd 2h"), (2, 14, 13, 12))

    def test_matches_brute_force(self):
        rng = random.Random(1)
        for _ in range(3000):
            cards = rng.sample(range(52), 7)
            best = max(evaluate(c) for c in itertools.combinations(cards, 5))
            self.assertEqual(evaluate(cards), best, cards)


class EngineTests(unittest.TestCase):
    def test_round_flow_and_board(self):
        g = GangGame(3, seed=1)
        self.assertEqual(g.round, 0)
        self.assertEqual(len(g.board()), 0)
        start = g.current
        for i in range(3):
            g.apply(take(i + 1))
        for _ in range(3):
            g.apply(PASS)
        self.assertEqual(g.round, 1)
        self.assertEqual(len(g.board()), 3)
        hist = g.chip_history[0]
        self.assertEqual(hist[start], 1)

    def test_cannot_pass_without_chip(self):
        g = GangGame(3, seed=1)
        self.assertNotIn(PASS, g.legal_actions())
        with self.assertRaises(ValueError):
            g.apply(PASS)

    def test_steal_and_swap(self):
        g = GangGame(3, seed=1)
        a = g.current
        g.apply(take(2))
        b = g.current
        g.apply(take(2))  # steal
        self.assertIsNone(g.chip_of(a))
        self.assertEqual(g.chip_of(b), 2)
        c = g.current
        g.apply(take(3))
        g.apply(take(1))  # a takes 1
        g.apply(PASS)  # b passes
        # nothing is free and c cannot take its own chip
        self.assertNotIn(take(3), g.legal_actions(c))

    def test_round_needs_everyone_to_pass(self):
        g = GangGame(3, seed=2)
        g.apply(take(1)); g.apply(take(2)); g.apply(take(3))
        g.apply(PASS); g.apply(PASS)
        g.apply(take(2))  # steal resets the pass streak
        self.assertEqual(g.round, 0)

    def test_lock_ends_round(self):
        g = GangGame(3, seed=0, max_actions_per_round=4)
        for _ in range(4):
            legal = [a for a in g.legal_actions() if a.kind == "take"]
            g.apply(legal[0])
        self.assertTrue(g.locked)
        for _ in range(10):
            if g.round > 0:
                break
            legal = g.legal_actions()
            for a in legal:
                if a.kind == "take":
                    self.assertIsNone(g.chips[a.star - 1])
            g.apply(legal[0])
        self.assertEqual(g.round, 1)

    def _play_hand_with_red_order(self, g, red_order_fn):
        for r in range(4):
            order = red_order_fn(g) if r == 3 else list(range(g.n))
            # seat order[k] should get chip k+1
            while g.round == r:
                seat = g.current
                want = order.index(seat) + 1
                res = g.apply(PASS if g.chip_of(seat) == want else take(want))
                if res is not None:
                    return res

    def test_showdown_success_when_sorted(self):
        g = GangGame(4, seed=5)
        def sorted_order(g):
            return sorted(range(g.n), key=lambda s: evaluate(g.holes[s] + g.full_board))
        res = self._play_hand_with_red_order(g, sorted_order)
        self.assertTrue(res.success)
        self.assertEqual(g.vaults, 1)

    def test_showdown_fails_when_reversed(self):
        for seed in range(50):
            g = GangGame(4, seed=seed)
            scores = [evaluate(h + g.full_board) for h in g.holes]
            if len(set(scores)) < 4:
                continue
            res = self._play_hand_with_red_order(
                g, lambda g: sorted(range(g.n), key=lambda s: scores[s], reverse=True))
            self.assertFalse(res.success)
            self.assertEqual(res.first_failure, 1)
            self.assertEqual(g.alarms, 1)
            return
        self.fail("no seed with distinct hands")

    def test_game_ends(self):
        for n in range(2, 7):
            g = play_game([RandomAgent(seed=i) for i in range(n)], seed=n)
            self.assertTrue(g.over)
            self.assertIn(len(g.results), (3, 4, 5))
            self.assertTrue(g.vaults == 3 or g.alarms == 3)


class AgentAndEnvTests(unittest.TestCase):
    def test_agents_play_full_games(self):
        for cls in (HeuristicAgent, ChipReaderAgent, TableReaderAgent):
            g = play_game([cls(seed=i, sims=40) for i in range(4)], seed=7)
            self.assertTrue(g.over)

    def test_tablereader_reads_in_round_steals(self):
        # 5 players: A(0) takes 1, B(1) takes 3, C(2) steals 1 from A, D(3) steals 1 from C.
        # B should now believe it ranks higher than before the steals.
        ups = 0
        for seed in range(6):
            g = GangGame(5, seed=seed)
            g.apply(take(1)); g.apply(take(3))
            bot = TableReaderAgent(seed=seed, sims=300)
            before = bot.expected_ranks(g.view(1))[1]
            g.apply(take(1)); g.apply(take(1))
            after = bot.expected_ranks(g.view(1))[1]
            ups += after > before
        self.assertGreaterEqual(ups, 5)

    def test_env(self):
        env = GangEnv(n_players=4, seed=3, teammate_factory=lambda s: HeuristicAgent(seed=s, sims=40))
        view = env.reset()
        me = HeuristicAgent(seed=99, sims=40)
        total = 0.0
        steps = 0
        while not env.done:
            self.assertEqual(view.seat, 0)
            view, reward, done, info = env.step(me.act(view))
            total += reward
            steps += 1
        self.assertGreater(steps, 0)
        self.assertEqual(total, env.game.vaults - env.game.alarms)


if __name__ == "__main__":
    unittest.main()
