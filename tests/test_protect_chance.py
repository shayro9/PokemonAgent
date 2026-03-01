import unittest
from unittest.mock import MagicMock

from poke_env.battle import MoveCategory

from combat.combat_utils import (
    ProtectBelief,
    build_protect_belief,
    did_no_damage,
    protect_chance,
)


def make_move(category: MoveCategory, accuracy=1.0):
    move = MagicMock()
    move.category = category
    move.accuracy = accuracy
    return move


def make_battle(species: str, current_hp_fraction: float):
    opp = MagicMock()
    opp.species = species
    opp.current_hp_fraction = current_hp_fraction

    battle = MagicMock()
    battle.opponent_active_pokemon = opp
    return battle


class TestDidNoDamage(unittest.TestCase):

    def test_none_move_returns_false(self):
        battle = make_battle("pikachu", 0.8)
        self.assertFalse(did_no_damage(battle, {"opp_pikachu": 1.0}, None))

    def test_status_move_returns_false(self):
        battle = make_battle("pikachu", 0.8)
        move = make_move(MoveCategory.STATUS)
        self.assertFalse(did_no_damage(battle, {"opp_pikachu": 1.0}, move))

    def test_no_previous_hp_returns_false(self):
        battle = make_battle("pikachu", 0.8)
        move = make_move(MoveCategory.PHYSICAL)
        self.assertFalse(did_no_damage(battle, {}, move))

    def test_physical_hp_dropped_returns_false(self):
        battle = make_battle("pikachu", 0.6)
        move = make_move(MoveCategory.PHYSICAL)
        self.assertFalse(did_no_damage(battle, {"opp_pikachu": 0.8}, move))

    def test_physical_hp_unchanged_returns_true(self):
        battle = make_battle("pikachu", 0.8)
        move = make_move(MoveCategory.PHYSICAL)
        self.assertTrue(did_no_damage(battle, {"opp_pikachu": 0.8}, move))

    def test_physical_hp_increased_returns_true(self):
        battle = make_battle("pikachu", 0.9)
        move = make_move(MoveCategory.PHYSICAL)
        self.assertTrue(did_no_damage(battle, {"opp_pikachu": 0.8}, move))


class TestProtectBelief(unittest.TestCase):

    def test_posterior_matches_model(self):
        belief = ProtectBelief(accuracy=0.9, last_chance=1 / 3, protect_attempt_prior=0.5)

        qp = 0.5 * (1 / 3)
        m = 0.1
        expected = qp / (qp + m * (1 - qp))

        self.assertAlmostEqual(belief.posterior_protect_success_given_no_damage(), expected)

    def test_expected_next_matches_model(self):
        belief = ProtectBelief(accuracy=0.9, last_chance=1 / 3, protect_attempt_prior=0.5)
        posterior = belief.posterior_protect_success_given_no_damage()
        expected = posterior * ((1 / 3) / 3) + (1 - posterior) * 1.0

        self.assertAlmostEqual(belief.expected_next_protect_chance_given_no_damage(), expected)

    def test_build_protect_belief_clips_inputs(self):
        move = make_move(MoveCategory.PHYSICAL, accuracy=1.2)
        belief = build_protect_belief(move, last_chance=-0.5, protect_attempt_prior=2.0)

        self.assertAlmostEqual(belief.accuracy, 1.0)
        self.assertAlmostEqual(belief.last_chance, 0.0)
        self.assertAlmostEqual(belief.protect_attempt_prior, 1.0)


class TestProtectChance(unittest.TestCase):

    def test_dealt_damage_resets(self):
        move = make_move(MoveCategory.PHYSICAL, accuracy=1.0)
        self.assertAlmostEqual(protect_chance(move, 1 / 9, no_damage=False), 1.0)

    def test_perfect_accuracy_certain_attempt_matches_chain(self):
        move = make_move(MoveCategory.PHYSICAL, accuracy=1.0)
        self.assertAlmostEqual(protect_chance(move, 1 / 3, no_damage=True, protect_attempt_prior=1.0), 1 / 9)

    def test_zero_accuracy_defaults_to_reset(self):
        move = make_move(MoveCategory.PHYSICAL, accuracy=0.0)
        self.assertAlmostEqual(protect_chance(move, 1 / 3, no_damage=True, protect_attempt_prior=0.5), 23 / 27)

    def test_partial_accuracy_with_prior(self):
        move = make_move(MoveCategory.PHYSICAL, accuracy=0.8)
        chance = protect_chance(move, last_chance=1 / 3, no_damage=True, protect_attempt_prior=0.4)

        qp = 0.4 * (1 / 3)
        m = 0.2
        posterior = qp / (qp + m * (1 - qp))
        expected = posterior * ((1 / 3) / 3) + (1 - posterior) * 1.0

        self.assertAlmostEqual(chance, expected)


if __name__ == "__main__":
    unittest.main()
