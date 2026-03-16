import json
import unittest
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np

from poke_env.battle.pokemon_type import PokemonType
from poke_env.battle.status import Status

from env.pokemon_stats import (
    PokemonStats,
    GEN1_STAT_KEYS,
    GEN1_BOOST_KEYS,
    ALL_TYPES,
    ALL_STATUSES,
    TRACKED_EFFECTS,
)

FIXTURES_DIR = Path(__file__).parent / "fixtures"


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
    ps: PokemonStats

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
        self.ps = PokemonStats(gen1=True)

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


# ---------------------------------------------------------------------------
# Starmie  –  Water/Psychic, PAR, full HP, no boosts
# ---------------------------------------------------------------------------

class TestPokemonStatsStarmie(PokemonStatsBaseTest, unittest.TestCase):

    def setUp(self):
        self.ps = PokemonStats(gen1=True, pokemon=make_mock_from_fixture("gen1_starmie.json"))

    def test_hp(self):
        self.assertEqual(self.ps.hp, 1.0)

    def test_stats_raw_values(self):
        np.testing.assert_array_equal(
            self.ps.stats,
            np.array([323, 226, 226, 284, 295], dtype=np.float32)
        )

    def test_boosts_all_zero(self):
        np.testing.assert_array_equal(self.ps.boosts, np.zeros(len(GEN1_BOOST_KEYS)))

    def test_status_par(self):
        self.assertEqual(self.ps.status[ALL_STATUSES.index(Status.PAR)], 1.0)

    def test_type_water(self):
        self.assertEqual(self.ps.types[ALL_TYPES.index(PokemonType.WATER)], 1.0)

    def test_type_psychic(self):
        self.assertEqual(self.ps.types[ALL_TYPES.index(PokemonType.PSYCHIC)], 1.0)

    def test_types_two_set(self):
        self.assertEqual(self.ps.types.sum(), 2.0)


# ---------------------------------------------------------------------------
# Tauros  –  Normal, no status, full HP, +2 atk / -2 spa+spd / +1 spe
# ---------------------------------------------------------------------------

class TestPokemonStatsTauros(PokemonStatsBaseTest, unittest.TestCase):

    def setUp(self):
        self.ps = PokemonStats(gen1=True, pokemon=make_mock_from_fixture("gen1_tauros.json"))

    def test_hp(self):
        self.assertEqual(self.ps.hp, 1.0)

    def test_stats_raw_values(self):
        np.testing.assert_array_equal(
            self.ps.stats,
            np.array([303, 298, 228, 158, 318], dtype=np.float32)
        )

    def test_no_status(self):
        self.assertEqual(self.ps.status.sum(), 0.0)

    def test_type_normal_only(self):
        self.assertEqual(self.ps.types[ALL_TYPES.index(PokemonType.NORMAL)], 1.0)
        self.assertEqual(self.ps.types.sum(), 1.0)

    def test_atk_boost(self):
        self.assertEqual(self.ps.boosts[GEN1_BOOST_KEYS.index("atk")], 2.0)

    def test_spa_boost(self):
        self.assertEqual(self.ps.boosts[GEN1_BOOST_KEYS.index("spa")], -2.0)

    def test_spe_boost(self):
        self.assertEqual(self.ps.boosts[GEN1_BOOST_KEYS.index("spe")], 1.0)

    def test_spa_spd_equal_in_fixture(self):
        # In Gen 1 special is one stat — spa and spd boosts must always match
        with open(FIXTURES_DIR / "gen1_tauros.json") as f:
            raw = json.load(f)
        self.assertEqual(raw["boosts"]["spa"], raw["boosts"]["spd"])


# ---------------------------------------------------------------------------
# Chansey  –  Normal, BRN, half HP, +3 def
# ---------------------------------------------------------------------------

class TestPokemonStatsChansey(PokemonStatsBaseTest, unittest.TestCase):

    def setUp(self):
        self.ps = PokemonStats(gen1=True, pokemon=make_mock_from_fixture("gen1_chansey.json"))

    def test_hp_fraction(self):
        self.assertAlmostEqual(self.ps.hp, 0.5)

    def test_stats_raw_values(self):
        np.testing.assert_array_equal(
            self.ps.stats,
            np.array([703, 35, 105, 195, 188], dtype=np.float32)
        )

    def test_status_brn(self):
        self.assertEqual(self.ps.status[ALL_STATUSES.index(Status.BRN)], 1.0)

    def test_def_boost(self):
        self.assertEqual(self.ps.boosts[GEN1_BOOST_KEYS.index("def")], 3.0)

    def test_other_boosts_zero(self):
        def_idx = GEN1_BOOST_KEYS.index("def")
        np.testing.assert_array_equal(
            np.delete(self.ps.boosts, def_idx),
            np.zeros(len(GEN1_BOOST_KEYS) - 1)
        )

    def test_type_normal_only(self):
        self.assertEqual(self.ps.types[ALL_TYPES.index(PokemonType.NORMAL)], 1.0)
        self.assertEqual(self.ps.types.sum(), 1.0)


# ---------------------------------------------------------------------------
# Alakazam  –  Psychic, SLP, 75% HP, +2 special (spa==spd)
# ---------------------------------------------------------------------------

class TestPokemonStatsAlakazam(PokemonStatsBaseTest, unittest.TestCase):

    def setUp(self):
        self.ps = PokemonStats(gen1=True, pokemon=make_mock_from_fixture("gen1_alakazam.json"))

    def test_hp_fraction(self):
        self.assertAlmostEqual(self.ps.hp, 0.75)

    def test_stats_raw_values(self):
        np.testing.assert_array_equal(
            self.ps.stats,
            np.array([273, 146, 136, 328, 328], dtype=np.float32)
        )

    def test_status_slp(self):
        self.assertEqual(self.ps.status[ALL_STATUSES.index(Status.SLP)], 1.0)

    def test_spa_boost(self):
        self.assertEqual(self.ps.boosts[GEN1_BOOST_KEYS.index("spa")], 2.0)

    def test_spa_spd_equal_in_fixture(self):
        with open(FIXTURES_DIR / "gen1_alakazam.json") as f:
            raw = json.load(f)
        self.assertEqual(raw["boosts"]["spa"], raw["boosts"]["spd"])

    def test_type_psychic_only(self):
        self.assertEqual(self.ps.types[ALL_TYPES.index(PokemonType.PSYCHIC)], 1.0)
        self.assertEqual(self.ps.types.sum(), 1.0)



if __name__ == "__main__":
    unittest.main()

