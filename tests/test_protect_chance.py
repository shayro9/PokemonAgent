import unittest
from unittest.mock import MagicMock
from poke_env.battle import MoveCategory
from env.combat_utils import did_no_damage, protect_chance


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
        """Opponent healed — still counts as no damage from us."""
        battle = make_battle("pikachu", 0.9)
        move = make_move(MoveCategory.PHYSICAL)
        self.assertTrue(did_no_damage(battle, {"opp_pikachu": 0.8}, move))

    def test_special_hp_dropped_returns_false(self):
        battle = make_battle("gengar", 0.3)
        move = make_move(MoveCategory.SPECIAL)
        self.assertFalse(did_no_damage(battle, {"opp_gengar": 0.5}, move))

    def test_special_hp_unchanged_returns_true(self):
        battle = make_battle("gengar", 0.5)
        move = make_move(MoveCategory.SPECIAL)
        self.assertTrue(did_no_damage(battle, {"opp_gengar": 0.5}, move))


class TestProtectChance(unittest.TestCase):

    def test_dealt_damage_resets(self):
        move = make_move(MoveCategory.PHYSICAL, accuracy=1.0)
        self.assertAlmostEqual(protect_chance(move, 1.0, no_damage=False), 1.0)

    def test_dealt_damage_resets_from_low_chance(self):
        move = make_move(MoveCategory.PHYSICAL, accuracy=0.9)
        self.assertAlmostEqual(protect_chance(move, 1 / 9, no_damage=False), 1.0)

    def test_perfect_accuracy_first_protect(self):
        move = make_move(MoveCategory.PHYSICAL, accuracy=1.0)
        self.assertAlmostEqual(protect_chance(move, 1.0, no_damage=True), 1 / 3)

    def test_perfect_accuracy_second_protect(self):
        move = make_move(MoveCategory.PHYSICAL, accuracy=1.0)
        self.assertAlmostEqual(protect_chance(move, 1 / 3, no_damage=True), 1 / 9)

    def test_perfect_accuracy_third_protect(self):
        move = make_move(MoveCategory.PHYSICAL, accuracy=1.0)
        self.assertAlmostEqual(protect_chance(move, 1 / 9, no_damage=True), 1 / 27)

    def test_zero_accuracy_first_protect(self):
        """Always miss → blend of scenario 2 and 3."""
        move = make_move(MoveCategory.PHYSICAL, accuracy=0.0)
        last_chance = 1.0
        expected = last_chance * (last_chance / 3) + (1 - last_chance) * 1.0
        self.assertAlmostEqual(protect_chance(move, last_chance, no_damage=True), expected)

    def test_zero_accuracy_low_protect_chance(self):
        move = make_move(MoveCategory.PHYSICAL, accuracy=0.0)
        last_chance = 1 / 9
        expected = last_chance * (last_chance / 3) + (1 - last_chance) * 1.0
        self.assertAlmostEqual(protect_chance(move, last_chance, no_damage=True), expected)

    def test_partial_accuracy_first_protect(self):
        move = make_move(MoveCategory.PHYSICAL, accuracy=0.8)
        last_chance = 1.0
        p_hit, p_miss = 0.8, 0.2
        expected = (p_hit * last_chance + p_miss * last_chance) * (last_chance / 3) \
                   + (p_miss * (1 - last_chance)) * 1.0
        self.assertAlmostEqual(protect_chance(move, last_chance, no_damage=True), expected)

    def test_partial_accuracy_second_protect(self):
        move = make_move(MoveCategory.PHYSICAL, accuracy=0.8)
        last_chance = 1 / 3
        p_hit, p_miss = 0.8, 0.2
        expected = (p_hit * last_chance + p_miss * last_chance) * (last_chance / 3) \
                   + (p_miss * (1 - last_chance)) * 1.0
        self.assertAlmostEqual(protect_chance(move, last_chance, no_damage=True), expected)

    def test_miss_chance_prevents_full_zero(self):
        """With any miss chance, protect never fully reaches 0."""
        move = make_move(MoveCategory.PHYSICAL, accuracy=0.9)
        result = protect_chance(move, 1 / 27, no_damage=True)
        self.assertGreater(result, 0.0)

    def test_special_move_accuracy(self):
        """Special moves should work the same as physical."""
        move = make_move(MoveCategory.SPECIAL, accuracy=0.7)
        last_chance = 1.0
        p_hit, p_miss = 0.7, 0.3
        expected = (p_hit * last_chance + p_miss * last_chance) * (last_chance / 3) \
                   + (p_miss * (1 - last_chance)) * 1.0
        self.assertAlmostEqual(protect_chance(move, last_chance, no_damage=True), expected)


if __name__ == "__main__":
    unittest.main()
