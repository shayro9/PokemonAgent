"""
Tests for env/action_mask_gen_1.py — ActionMaskGen1.

Covers:
  1. Initial state (all False, correct length)
  2. set() / get_mask() round-trip
  3. reset() restores all-False
  4. ACTION_SWITCH_RANGE and ACTION_MOVE_RANGE constants
  5. describe() without battle argument
  6. describe() with battle mock
  7. Mask dtype is bool
"""

import unittest
from unittest.mock import MagicMock
import numpy as np

from env.action_mask_gen_1 import ActionMaskGen1


class TestActionMaskInit(unittest.TestCase):

    def test_initial_mask_all_false(self):
        mask = ActionMaskGen1()
        self.assertTrue(np.all(~mask.get_mask()))

    def test_initial_mask_length(self):
        mask = ActionMaskGen1()
        self.assertEqual(len(mask.get_mask()), ActionMaskGen1.ACTION_SPACE)

    def test_initial_mask_dtype_bool(self):
        mask = ActionMaskGen1()
        self.assertEqual(mask.get_mask().dtype, bool)

    def test_action_space_is_10(self):
        self.assertEqual(ActionMaskGen1.ACTION_SPACE, 10)


class TestActionMaskConstants(unittest.TestCase):

    def test_switch_range_covers_slots_0_to_5(self):
        self.assertEqual(list(ActionMaskGen1.ACTION_SWITCH_RANGE), [0, 1, 2, 3, 4, 5])

    def test_move_range_covers_slots_6_to_9(self):
        self.assertEqual(list(ActionMaskGen1.ACTION_MOVE_RANGE), [6, 7, 8, 9])

    def test_switch_and_move_ranges_non_overlapping(self):
        switches = set(ActionMaskGen1.ACTION_SWITCH_RANGE)
        moves = set(ActionMaskGen1.ACTION_MOVE_RANGE)
        self.assertFalse(switches & moves)

    def test_switch_and_move_ranges_cover_full_action_space(self):
        all_actions = set(ActionMaskGen1.ACTION_SWITCH_RANGE) | set(ActionMaskGen1.ACTION_MOVE_RANGE)
        self.assertEqual(all_actions, set(range(ActionMaskGen1.ACTION_SPACE)))


class TestActionMaskSetGet(unittest.TestCase):

    def test_set_and_get_round_trip(self):
        mask = ActionMaskGen1()
        values = [1, 0, 0, 0, 0, 0, 1, 1, 0, 0]
        mask.set(values)
        result = mask.get_mask()
        np.testing.assert_array_equal(result, np.array(values, dtype=bool))

    def test_set_all_true(self):
        mask = ActionMaskGen1()
        mask.set([1] * ActionMaskGen1.ACTION_SPACE)
        self.assertTrue(np.all(mask.get_mask()))

    def test_set_all_false(self):
        mask = ActionMaskGen1()
        mask.set([1] * ActionMaskGen1.ACTION_SPACE)
        mask.set([0] * ActionMaskGen1.ACTION_SPACE)
        self.assertTrue(np.all(~mask.get_mask()))

    def test_set_only_moves_valid(self):
        mask = ActionMaskGen1()
        values = [0, 0, 0, 0, 0, 0, 1, 1, 1, 1]
        mask.set(values)
        for i in ActionMaskGen1.ACTION_SWITCH_RANGE:
            self.assertFalse(mask.get_mask()[i])
        for i in ActionMaskGen1.ACTION_MOVE_RANGE:
            self.assertTrue(mask.get_mask()[i])

    def test_set_only_switches_valid(self):
        mask = ActionMaskGen1()
        values = [1, 1, 1, 1, 1, 1, 0, 0, 0, 0]
        mask.set(values)
        for i in ActionMaskGen1.ACTION_SWITCH_RANGE:
            self.assertTrue(mask.get_mask()[i])
        for i in ActionMaskGen1.ACTION_MOVE_RANGE:
            self.assertFalse(mask.get_mask()[i])

    def test_get_mask_returns_ndarray(self):
        mask = ActionMaskGen1()
        self.assertIsInstance(mask.get_mask(), np.ndarray)


class TestActionMaskReset(unittest.TestCase):

    def test_reset_after_set_restores_all_false(self):
        mask = ActionMaskGen1()
        mask.set([1] * ActionMaskGen1.ACTION_SPACE)
        mask.reset()
        self.assertTrue(np.all(~mask.get_mask()))

    def test_reset_idempotent(self):
        mask = ActionMaskGen1()
        mask.reset()
        mask.reset()
        self.assertTrue(np.all(~mask.get_mask()))

    def test_reset_does_not_change_length(self):
        mask = ActionMaskGen1()
        mask.reset()
        self.assertEqual(len(mask.get_mask()), ActionMaskGen1.ACTION_SPACE)


class TestActionMaskDescribe(unittest.TestCase):

    def test_describe_no_battle_returns_string(self):
        mask = ActionMaskGen1()
        result = mask.describe()
        self.assertIsInstance(result, str)

    def test_describe_contains_header(self):
        mask = ActionMaskGen1()
        result = mask.describe()
        self.assertIn("ACTION MASK", result)

    def test_describe_no_valid_actions_shows_none(self):
        mask = ActionMaskGen1()
        result = mask.describe()
        self.assertIn("(none)", result)

    def test_describe_with_active_move_shows_move(self):
        mask = ActionMaskGen1()
        mask.set([0, 0, 0, 0, 0, 0, 1, 0, 0, 0])  # only move slot 0 active
        result = mask.describe()
        self.assertIn("MOVE", result)
        self.assertNotIn("SWITCH", result.replace("--- Switches ---", ""))

    def test_describe_with_active_switch_shows_switch(self):
        mask = ActionMaskGen1()
        mask.set([1, 0, 0, 0, 0, 0, 0, 0, 0, 0])  # only switch slot 0 active
        result = mask.describe()
        self.assertIn("SWITCH", result)

    def test_describe_valid_actions_list(self):
        mask = ActionMaskGen1()
        mask.set([1, 0, 0, 0, 0, 0, 1, 1, 0, 0])
        result = mask.describe()
        self.assertIn("Valid actions: [0, 6, 7]", result)

    def test_describe_with_battle_uses_pokemon_names(self):
        mask = ActionMaskGen1()
        mask.set([1, 0, 0, 0, 0, 0, 0, 0, 0, 0])

        mon = MagicMock()
        mon.species = "Pikachu"
        battle = MagicMock()
        battle.team = {"slot_0": mon}

        result = mask.describe(battle=battle)
        self.assertIn("Pikachu", result)

    def test_describe_with_battle_uses_move_names(self):
        mask = ActionMaskGen1()
        mask.set([0, 0, 0, 0, 0, 0, 1, 0, 0, 0])

        move = MagicMock()
        move.id = "thunderbolt"
        active_mon = MagicMock()
        active_mon.moves = {"thunderbolt": move}
        battle = MagicMock()
        battle.team = {}
        battle.active_pokemon = active_mon

        result = mask.describe(battle=battle)
        self.assertIn("thunderbolt", result)

    def test_describe_without_battle_uses_placeholder_names(self):
        mask = ActionMaskGen1()
        mask.set([1, 0, 0, 0, 0, 0, 1, 0, 0, 0])
        result = mask.describe(battle=None)
        self.assertIn("Pokemon_1", result)
        self.assertIn("move_1", result)


if __name__ == "__main__":
    unittest.main()
