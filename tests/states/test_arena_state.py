import unittest
from unittest.mock import MagicMock

import numpy as np

from poke_env.battle.side_condition import SideCondition

from env.states.gen1.arena_state_gen1 import ArenaStateGen1, TURN_NORM


def make_battle_mock(
    turn: int = 0,
    my_conditions: dict = None,
    opp_conditions: dict = None,
) -> MagicMock:
    """Build a minimal mock battle object for ArenaState."""
    battle = MagicMock()
    battle.turn                     = turn
    battle.side_conditions          = my_conditions  or {}
    battle.opponent_side_conditions = opp_conditions or {}
    return battle


# ---------------------------------------------------------------------------
# Shared base
# ---------------------------------------------------------------------------

class ArenaStateBaseTest:
    """
    Mixin — subclasses must set self.arena in setUp().
    """
    arena: ArenaStateGen1

    def test_my_screens_shape(self):
        self.assertEqual(self.arena.my_screens.shape, (len(ArenaStateGen1.TRACKED_SCREENS),))

    def test_my_screens_dtype(self):
        self.assertEqual(self.arena.my_screens.dtype, np.float32)

    def test_opp_screens_shape(self):
        self.assertEqual(self.arena.opp_screens.shape, (len(ArenaStateGen1.TRACKED_SCREENS),))

    def test_opp_screens_dtype(self):
        self.assertEqual(self.arena.opp_screens.dtype, np.float32)

    def test_to_array_dtype(self):
        self.assertEqual(self.arena.to_array().dtype, np.float32)

    def test_to_array_length(self):
        self.assertEqual(len(self.arena.to_array()), self.arena.array_len())

    def test_array_len(self):
        expected = 1 + len(ArenaStateGen1.TRACKED_SCREENS) * 2
        self.assertEqual(self.arena.array_len(), expected)

    def test_screens_binary(self):
        for v in np.concatenate([self.arena.my_screens, self.arena.opp_screens]):
            self.assertIn(v, [0.0, 1.0])


# ---------------------------------------------------------------------------
# Empty state
# ---------------------------------------------------------------------------

class TestArenaStateEmpty(ArenaStateBaseTest, unittest.TestCase):

    def setUp(self):
        self.arena = ArenaStateGen1()

    def test_turn_is_zero(self):
        self.assertEqual(self.arena.turn, 0)

    def test_turn_encoded_is_zero(self):
        self.assertEqual(self.arena.turn_encoded(), 0.0)

    def test_my_screens_all_zero(self):
        np.testing.assert_array_equal(
            self.arena.my_screens,
            np.zeros(len(ArenaStateGen1.TRACKED_SCREENS))
        )

    def test_opp_screens_all_zero(self):
        np.testing.assert_array_equal(
            self.arena.opp_screens,
            np.zeros(len(ArenaStateGen1.TRACKED_SCREENS))
        )


# ---------------------------------------------------------------------------
# Turn normalization
# ---------------------------------------------------------------------------

class TestArenaStateTurn(ArenaStateBaseTest, unittest.TestCase):

    def setUp(self):
        self.arena = ArenaStateGen1(make_battle_mock(turn=15))

    def test_turn_raw(self):
        self.assertEqual(self.arena.turn, 15)

    def test_turn_encoded(self):
        self.assertAlmostEqual(self.arena.turn_encoded(), 15 / TURN_NORM)

    def test_turn_encoded_capped_at_1(self):
        arena = ArenaStateGen1(make_battle_mock(turn=999))
        self.assertEqual(arena.turn_encoded(), 1.0)

    def test_turn_encoded_in_to_array(self):
        self.assertAlmostEqual(float(self.arena.to_array()[0]), 15 / TURN_NORM)


# ---------------------------------------------------------------------------
# Reflect on my side only
# ---------------------------------------------------------------------------

class TestArenaStateMyReflect(ArenaStateBaseTest, unittest.TestCase):

    def setUp(self):
        self.arena = ArenaStateGen1(make_battle_mock(
            turn=5,
            my_conditions={SideCondition.REFLECT: 1},
        ))

    def test_my_reflect_set(self):
        reflect_idx = ArenaStateGen1.TRACKED_SCREENS.index(SideCondition.REFLECT)
        self.assertEqual(self.arena.my_screens[reflect_idx], 1.0)

    def test_my_light_screen_not_set(self):
        ls_idx = ArenaStateGen1.TRACKED_SCREENS.index(SideCondition.LIGHT_SCREEN)
        self.assertEqual(self.arena.my_screens[ls_idx], 0.0)

    def test_opp_screens_all_zero(self):
        np.testing.assert_array_equal(
            self.arena.opp_screens,
            np.zeros(len(ArenaStateGen1.TRACKED_SCREENS))
        )


# ---------------------------------------------------------------------------
# Light Screen on opponent side only
# ---------------------------------------------------------------------------

class TestArenaStateOppLightScreen(ArenaStateBaseTest, unittest.TestCase):

    def setUp(self):
        self.arena = ArenaStateGen1(make_battle_mock(
            turn=10,
            opp_conditions={SideCondition.LIGHT_SCREEN: 1},
        ))

    def test_opp_light_screen_set(self):
        ls_idx = ArenaStateGen1.TRACKED_SCREENS.index(SideCondition.LIGHT_SCREEN)
        self.assertEqual(self.arena.opp_screens[ls_idx], 1.0)

    def test_opp_reflect_not_set(self):
        reflect_idx = ArenaStateGen1.TRACKED_SCREENS.index(SideCondition.REFLECT)
        self.assertEqual(self.arena.opp_screens[reflect_idx], 0.0)

    def test_my_screens_all_zero(self):
        np.testing.assert_array_equal(
            self.arena.my_screens,
            np.zeros(len(ArenaStateGen1.TRACKED_SCREENS))
        )


# ---------------------------------------------------------------------------
# Both screens on both sides
# ---------------------------------------------------------------------------

class TestArenaStateBothScreensBothSides(ArenaStateBaseTest, unittest.TestCase):

    def setUp(self):
        self.arena = ArenaStateGen1(make_battle_mock(
            turn=3,
            my_conditions={SideCondition.REFLECT: 1, SideCondition.LIGHT_SCREEN: 1},
            opp_conditions={SideCondition.REFLECT: 1, SideCondition.LIGHT_SCREEN: 1},
        ))

    def test_my_all_screens_set(self):
        np.testing.assert_array_equal(
            self.arena.my_screens,
            np.ones(len(ArenaStateGen1.TRACKED_SCREENS), dtype=np.float32)
        )

    def test_opp_all_screens_set(self):
        np.testing.assert_array_equal(
            self.arena.opp_screens,
            np.ones(len(ArenaStateGen1.TRACKED_SCREENS), dtype=np.float32)
        )


if __name__ == "__main__":
    unittest.main()

