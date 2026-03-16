import json
import unittest
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np

from poke_env.battle.pokemon_type import PokemonType
from poke_env.battle.status import Status

from env.states.my_pokemon_state import MyPokemonState
from env.states.pokemon_state import (
    PokemonState,
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
    p.species             = data["species"]
    p.stats               = data["stats"]
    p.boosts              = data["boosts"]
    p.status              = Status[data["status"]] if data["status"] else None
    p.effects             = {}
    p.types               = tuple(PokemonType[t] for t in data["types"])
    p.stab_multiplier     = data["stab_multiplier"]
    return p


def _expected_array_len():
    hp = stab = 1
    return hp + len(PokemonState.STAT_KEYS) + len(PokemonState.BOOST_KEYS) + len(ALL_STATUSES) + len(TRACKED_EFFECTS) + stab


# ---------------------------------------------------------------------------
# Shared base — structure / dtype / to_array checks every fixture must pass
# ---------------------------------------------------------------------------

class PokemonStatsBaseTest:
    """
    Mixin with shared checks — does NOT inherit TestCase so unittest
    won't discover and run it directly. Subclasses inherit from both
    this mixin and unittest.TestCase.
    """
    ps: MyPokemonState

    # ── stats ────────────────────────────────────────────────────────────

    def test_stats_shape(self):
        self.assertEqual(self.ps.stats.shape, (len(PokemonState.STAT_KEYS),))

    def test_stats_dtype(self):
        self.assertEqual(self.ps.stats.dtype, np.float32)

    def test_stats_encoded_in_range(self):
        enc = self.ps.stats_encoded()
        self.assertTrue(np.all(enc >= 0.0) and np.all(enc <= 1.0))

    # ── boosts ───────────────────────────────────────────────────────────

    def test_boosts_shape(self):
        self.assertEqual(self.ps.boosts.shape, (len(PokemonState.BOOST_KEYS),))

    def test_boosts_dtype(self):
        self.assertEqual(self.ps.boosts.dtype, np.float32)

    def test_boosts_encoded_in_range(self):
        enc = self.ps.boosts_encoded()
        self.assertTrue(np.all(enc >= -1.0) and np.all(enc <= 1.0))

    # ── status ───────────────────────────────────────────────────────────

    def test_status_shape(self):
        self.assertEqual(self.ps.status.shape, (len(ALL_STATUSES),))

    def test_status_dtype(self):
        self.assertEqual(self.ps.status.dtype, np.float32)

    def test_status_at_most_one_hot(self):
        self.assertIn(self.ps.status.sum(), [0.0, 1.0])

    # ── effects ──────────────────────────────────────────────────────────

    def test_effects_shape(self):
        self.assertEqual(self.ps.effects.shape, (len(TRACKED_EFFECTS),))

    def test_effects_dtype(self):
        self.assertEqual(self.ps.effects.dtype, np.float32)

    # ── stab: raw multiplier stored on object; normalised inside to_array ─

    def test_stab_is_1_5(self):
        self.assertEqual(self.ps.stab, 1.5 / 2.0)

    # ── to_array ─────────────────────────────────────────────────────────

    def test_to_array_length(self):
        self.assertEqual(len(self.ps.to_array()), self.ps.array_len())

    def test_to_array_dtype(self):
        self.assertEqual(self.ps.to_array().dtype, np.float32)

    def test_array_len(self):
        self.assertEqual(self.ps.array_len(), _expected_array_len())


# ---------------------------------------------------------------------------
# Empty state (no pokemon)
# ---------------------------------------------------------------------------

class TestPokemonStatsEmpty(PokemonStatsBaseTest, unittest.TestCase):
    """MyPokemonState with no pokemon — all zeros (except stab = 1.5)."""

    def setUp(self):
        self.ps = MyPokemonState()

    def test_hp_is_zero(self):
        self.assertEqual(self.ps.hp, 0.0)

    def test_species_is_none(self):
        self.assertEqual(self.ps.species, "none")

    def test_stats_all_zero(self):
        np.testing.assert_array_equal(self.ps.stats, np.zeros(len(PokemonState.STAT_KEYS)))

    def test_boosts_all_zero(self):
        np.testing.assert_array_equal(self.ps.boosts, np.zeros(len(PokemonState.BOOST_KEYS)))

    def test_status_all_zero(self):
        np.testing.assert_array_equal(self.ps.status, np.zeros(len(ALL_STATUSES)))

    def test_effects_all_zero(self):
        np.testing.assert_array_equal(self.ps.effects, np.zeros(len(TRACKED_EFFECTS)))

    def test_stab_is_0_75(self):
        self.assertEqual(self.ps.stab, 0.75)


if __name__ == "__main__":
    unittest.main()