import json
import unittest
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np

from poke_env.battle.pokemon_type import PokemonType
from poke_env.battle.status import Status

from env.states.pokemon_state import (
    PokemonState,
    GEN1_STAT_KEYS,
    GEN1_BOOST_KEYS,
    ALL_TYPES,
    ALL_STATUSES,
    TRACKED_EFFECTS,
)

FIXTURES_DIR = Path(__file__).parent.parent / "fixtures"


def make_mock_from_fixture(filename: str) -> MagicMock:
    """Load a fixture JSON and return a mock Pokemon object."""
    with open(FIXTURES_DIR / filename) as f:
        data = json.load(f)

    p = MagicMock()
    p.current_hp_fraction = data["current_hp_fraction"]
    p.stats               = data["stats"]
    p.boosts              = data["boosts"]
    p.status              = Status[data["status"]] if data["status"] else None
    p.effects             = {}
    p.types               = tuple(PokemonType[t] for t in data["types"])
    p.stab_multiplier     = data["stab_multiplier"]
    return p


def _expected_array_len():
    hp = stab = 1
    return hp + len(GEN1_STAT_KEYS) + len(GEN1_BOOST_KEYS) + len(ALL_STATUSES) + len(TRACKED_EFFECTS) + len(ALL_TYPES) + stab


# ---------------------------------------------------------------------------
# Shared base — structure / dtype / to_array checks every fixture must pass
# ---------------------------------------------------------------------------

class PokemonStatsBaseTest:
    """
    Mixin with shared checks — does NOT inherit TestCase so unittest
    won't discover and run it directly. Subclasses inherit from both
    this mixin and unittest.TestCase.
    """
    ps: PokemonState

    def test_stats_shape(self):
        self.assertEqual(self.ps.stats.shape, (len(GEN1_STAT_KEYS),))

    def test_stats_dtype(self):
        self.assertEqual(self.ps.stats.dtype, np.float32)

    def test_stats_encoded_in_range(self):
        enc = self.ps.stats_encoded()
        self.assertTrue(np.all(enc >= 0.0) and np.all(enc <= 1.0))

    def test_boosts_shape(self):
        self.assertEqual(self.ps.boosts.shape, (len(GEN1_BOOST_KEYS),))

    def test_boosts_dtype(self):
        self.assertEqual(self.ps.boosts.dtype, np.float32)

    def test_boosts_encoded_in_range(self):
        enc = self.ps.boosts_encoded()
        self.assertTrue(np.all(enc >= -1.0) and np.all(enc <= 1.0))

    def test_status_shape(self):
        self.assertEqual(self.ps.status.shape, (len(ALL_STATUSES),))

    def test_status_dtype(self):
        self.assertEqual(self.ps.status.dtype, np.float32)

    def test_status_at_most_one_hot(self):
        self.assertIn(self.ps.status.sum(), [0.0, 1.0])

    def test_effects_shape(self):
        self.assertEqual(self.ps.effects.shape, (len(TRACKED_EFFECTS),))

    def test_effects_dtype(self):
        self.assertEqual(self.ps.effects.dtype, np.float32)

    def test_types_shape(self):
        self.assertEqual(self.ps.types.shape, (len(ALL_TYPES),))

    def test_types_dtype(self):
        self.assertEqual(self.ps.types.dtype, np.float32)

    def test_stab_is_1_5(self):
        self.assertEqual(self.ps.stab, 1.5)

    def test_to_array_length(self):
        self.assertEqual(len(self.ps.to_array()), self.ps.array_len())

    def test_to_array_dtype(self):
        self.assertEqual(self.ps.to_array().dtype, np.float32)

    def test_array_len(self):
        self.assertEqual(self.ps.array_len(), _expected_array_len())


# ---------------------------------------------------------------------------
# Empty state
# ---------------------------------------------------------------------------

class TestPokemonStatsEmpty(PokemonStatsBaseTest, unittest.TestCase):
    """PokemonStats with no pokemon — all zeros."""

    def setUp(self):
        self.ps = PokemonState(gen1=True)

    def test_hp_is_zero(self):
        self.assertEqual(self.ps.hp, 0.0)

    def test_stats_all_zero(self):
        np.testing.assert_array_equal(self.ps.stats, np.zeros(len(GEN1_STAT_KEYS)))

    def test_boosts_all_zero(self):
        np.testing.assert_array_equal(self.ps.boosts, np.zeros(len(GEN1_BOOST_KEYS)))

    def test_status_all_zero(self):
        np.testing.assert_array_equal(self.ps.status, np.zeros(len(ALL_STATUSES)))

    def test_effects_all_zero(self):
        np.testing.assert_array_equal(self.ps.effects, np.zeros(len(TRACKED_EFFECTS)))

    def test_types_all_zero(self):
        np.testing.assert_array_equal(self.ps.types, np.zeros(len(ALL_TYPES)))

if __name__ == "__main__":
    unittest.main()

