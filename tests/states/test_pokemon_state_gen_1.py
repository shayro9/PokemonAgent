import json
import unittest

import numpy as np

from poke_env.battle.status import Status

from env.states.my_pokemon_state import MyPokemonState
from env.states.pokemon_state import (
    PokemonState,
    ALL_STATUSES,
)
from tests.states.test_pokemon_state import PokemonStatsBaseTest, FIXTURES_DIR, make_mock_from_fixture


# ---------------------------------------------------------------------------
# Starmie  –  Water/Psychic, PAR, full HP, no boosts
# ---------------------------------------------------------------------------

class TestPokemonStatsStarmie(PokemonStatsBaseTest, unittest.TestCase):

    def setUp(self):
        self.ps = MyPokemonState(pokemon=make_mock_from_fixture("gen1_starmie.json"))

    def test_hp(self):
        self.assertEqual(self.ps.hp, 1.0)

    def test_species(self):
        self.assertEqual(self.ps.species, "starmie")

    def test_stats_raw_values(self):
        np.testing.assert_array_equal(
            self.ps.stats,
            np.array([323, 226, 226, 284, 295], dtype=np.float32)
        )

    def test_boosts_all_zero(self):
        np.testing.assert_array_equal(self.ps.boosts, np.zeros(len(PokemonState.BOOST_KEYS)))

    def test_status_par(self):
        self.assertEqual(self.ps.status[ALL_STATUSES.index(Status.PAR)], 1.0)



# ---------------------------------------------------------------------------
# Tauros  –  Normal, no status, full HP, +2 atk / -2 spa+spd / +1 spe
# ---------------------------------------------------------------------------

class TestPokemonStatsTauros(PokemonStatsBaseTest, unittest.TestCase):

    def setUp(self):
        self.ps = MyPokemonState(pokemon=make_mock_from_fixture("gen1_tauros.json"))

    def test_hp(self):
        self.assertEqual(self.ps.hp, 1.0)

    def test_species(self):
        self.assertEqual(self.ps.species, "tauros")

    def test_stats_raw_values(self):
        np.testing.assert_array_equal(
            self.ps.stats,
            np.array([303, 298, 228, 158, 318], dtype=np.float32)
        )

    def test_no_status(self):
        self.assertEqual(self.ps.status.sum(), 0.0)

    def test_atk_boost(self):
        self.assertEqual(self.ps.boosts[PokemonState.BOOST_KEYS.index("atk")], 2.0)

    def test_spa_boost(self):
        self.assertEqual(self.ps.boosts[PokemonState.BOOST_KEYS.index("spa")], -2.0)

    def test_spe_boost(self):
        self.assertEqual(self.ps.boosts[PokemonState.BOOST_KEYS.index("spe")], 1.0)

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
        self.ps = MyPokemonState(pokemon=make_mock_from_fixture("gen1_chansey.json"))

    def test_hp_fraction(self):
        self.assertAlmostEqual(self.ps.hp, 0.5)

    def test_species(self):
        self.assertEqual(self.ps.species, "chansey")

    def test_stats_raw_values(self):
        np.testing.assert_array_equal(
            self.ps.stats,
            np.array([703, 35, 105, 195, 188], dtype=np.float32)
        )

    def test_status_brn(self):
        self.assertEqual(self.ps.status[ALL_STATUSES.index(Status.BRN)], 1.0)

    def test_def_boost(self):
        self.assertEqual(self.ps.boosts[PokemonState.BOOST_KEYS.index("def")], 3.0)

    def test_other_boosts_zero(self):
        def_idx = PokemonState.BOOST_KEYS.index("def")
        np.testing.assert_array_equal(
            np.delete(self.ps.boosts, def_idx),
            np.zeros(len(PokemonState.BOOST_KEYS) - 1)
        )


# ---------------------------------------------------------------------------
# Alakazam  –  Psychic, SLP, 75% HP, +2 special (spa==spd)
# ---------------------------------------------------------------------------

class TestPokemonStatsAlakazam(PokemonStatsBaseTest, unittest.TestCase):

    def setUp(self):
        self.ps = MyPokemonState(pokemon=make_mock_from_fixture("gen1_alakazam.json"))

    def test_hp_fraction(self):
        self.assertAlmostEqual(self.ps.hp, 0.75)

    def test_species(self):
        self.assertEqual(self.ps.species, "alakazam")

    def test_stats_raw_values(self):
        np.testing.assert_array_equal(
            self.ps.stats,
            np.array([273, 146, 136, 328, 328], dtype=np.float32)
        )

    def test_status_slp(self):
        self.assertEqual(self.ps.status[ALL_STATUSES.index(Status.SLP)], 1.0)

    def test_spa_boost(self):
        self.assertEqual(self.ps.boosts[PokemonState.BOOST_KEYS.index("spa")], 2.0)

    def test_spa_spd_equal_in_fixture(self):
        with open(FIXTURES_DIR / "gen1_alakazam.json") as f:
            raw = json.load(f)
        self.assertEqual(raw["boosts"]["spa"], raw["boosts"]["spd"])


if __name__ == "__main__":
    unittest.main()