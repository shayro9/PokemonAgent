"""
Tests unique to MyPokemonStateGen1 — what it adds on top of PokemonState:

  1. to_array() layout  — correct slice positions and normalization by STAT_NORM/BOOST_NORM/STAB_NORM
  2. array_len()        — formula matches actual to_array() length
  3. describe()         — Stats line shows all keys, Boosts line shows non-zero stages or "none"

Everything else (hp, species, status, boosts field, effects, types, stab field, class constants)
is already covered by test_pokemon_state.py and is NOT repeated here.
"""
import json
import unittest
import numpy as np

from env.states.my_pokemon_state_gen_1 import MyPokemonStateGen1
from poke_env.battle.status import Status
from poke_env.battle.effect import Effect
from env.states.state_utils import (
    GEN1_STAT_KEYS,
    GEN1_BOOST_KEYS,
    GEN1_TRACKED_EFFECTS,
    ALL_STATUSES,
    STAT_NORM,
    BOOST_NORM,
    STAB_NORM,
)
from tests.states.test_pokemon_state import FIXTURES_DIR, make_mock_from_fixture
from unittest.mock import MagicMock

# ---------------------------------------------------------------------------
# to_array() slice positions
# Layout: [hp | stats | boosts | status | effects | stab]
# ---------------------------------------------------------------------------
_N_STATS    = len(GEN1_STAT_KEYS)
_N_BOOSTS   = len(GEN1_BOOST_KEYS)
_N_STATUSES = len(ALL_STATUSES)
_N_EFFECTS  = len(GEN1_TRACKED_EFFECTS)
_HP_IDX     = 0
_STATS_S    = slice(1, 1 + _N_STATS)
_BOOSTS_S   = slice(1 + _N_STATS, 1 + _N_STATS + _N_BOOSTS)
_STATUS_S   = slice(1 + _N_STATS + _N_BOOSTS, 1 + _N_STATS + _N_BOOSTS + _N_STATUSES)
_EFFECTS_S  = slice(1 + _N_STATS + _N_BOOSTS + _N_STATUSES,
                    1 + _N_STATS + _N_BOOSTS + _N_STATUSES + _N_EFFECTS)
_STAB_IDX   = -1


# ---------------------------------------------------------------------------
# to_array() integrity — no NaN/Inf, length matches array_len()
# Tested on every fixture to catch regressions across different inputs
# ---------------------------------------------------------------------------

class TestMyPokemonStateGen1ArrayIntegrity(unittest.TestCase):

    FIXTURES = [
        "gen1_starmie.json",   # PAR, full hp, no boosts
        "gen1_chansey.json",   # BRN, half hp, +3 def, hp stat > STAT_NORM
        "gen1_tauros.json",    # no status, mixed positive/negative boosts
        "gen1_alakazam.json",  # SLP, 75% hp, +2 spa
    ]

    def _make(self, fixture):
        return MyPokemonStateGen1(pokemon=make_mock_from_fixture(fixture))

    def test_length_matches_array_len(self):
        for f in self.FIXTURES:
            ps = self._make(f)
            self.assertEqual(len(ps.to_array()), ps.array_len(), msg=f)

    def test_no_nan(self):
        for f in self.FIXTURES:
            self.assertFalse(np.any(np.isnan(self._make(f).to_array())), msg=f)

    def test_no_inf(self):
        for f in self.FIXTURES:
            self.assertFalse(np.any(np.isinf(self._make(f).to_array())), msg=f)

    def test_empty_state_length_matches_array_len(self):
        ps = MyPokemonStateGen1()
        self.assertEqual(len(ps.to_array()), ps.array_len())


# ---------------------------------------------------------------------------
# to_array() stats slice — normalised by STAT_NORM, clamped to [0, 1]
# Starmie: all stats within range → normalised correctly
# Chansey: hp=703 > STAT_NORM=600 → clamped to 1.0
# ---------------------------------------------------------------------------

class TestMyPokemonStateGen1StatsSlice(unittest.TestCase):

    def test_stats_normalised_correctly(self):
        # Starmie: hp=323, atk=226, def=226, spc=284, spe=295
        with open(FIXTURES_DIR / "gen1_starmie.json") as f:
            raw = json.load(f)
        arr = MyPokemonStateGen1(pokemon=make_mock_from_fixture("gen1_starmie.json")).to_array()
        for i, k in enumerate(GEN1_STAT_KEYS):
            expected = min(raw["stats"][k] / STAT_NORM, 1.0)
            self.assertAlmostEqual(float(arr[_STATS_S][i]), expected, places=5, msg=f"stats[{k}]")

    def test_stat_above_norm_clamped_to_one(self):
        # Chansey: hp=703 / 600 > 1.0 → must clamp to 1.0
        arr = MyPokemonStateGen1(pokemon=make_mock_from_fixture("gen1_chansey.json")).to_array()
        self.assertAlmostEqual(float(arr[_STATS_S][0]), 1.0, places=5)

    def test_stat_within_norm_normalised_correctly(self):
        # Chansey: atk=35 / 600 ≈ 0.0583
        arr = MyPokemonStateGen1(pokemon=make_mock_from_fixture("gen1_chansey.json")).to_array()
        self.assertAlmostEqual(float(arr[_STATS_S][1]), 35 / STAT_NORM, places=5)

    def test_stats_slice_is_all_zeros_for_empty_state(self):
        arr = MyPokemonStateGen1().to_array()
        np.testing.assert_array_equal(arr[_STATS_S], np.zeros(_N_STATS))


# ---------------------------------------------------------------------------
# to_array() boosts slice — normalised by BOOST_NORM, symmetric [-1, 1]
# Tauros: atk=+2, spa=-2, spe=+1
# ---------------------------------------------------------------------------

class TestMyPokemonStateGen1BoostsSlice(unittest.TestCase):

    def setUp(self):
        self.arr = MyPokemonStateGen1(pokemon=make_mock_from_fixture("gen1_tauros.json")).to_array()

    def test_positive_boost_normalised(self):
        # atk=+2 → 2/6 ≈ 0.333
        atk_idx = GEN1_BOOST_KEYS.index("atk")
        self.assertAlmostEqual(float(self.arr[_BOOSTS_S][atk_idx]), 2 / BOOST_NORM, places=5)

    def test_negative_boost_normalised(self):
        # boosts use symmetric=True so -2/6 ≈ -0.333 is preserved
        spa_idx = GEN1_BOOST_KEYS.index("spa")
        self.assertAlmostEqual(float(self.arr[_BOOSTS_S][spa_idx]), -2 / BOOST_NORM, places=5)

    def test_zero_boost_is_zero(self):
        def_idx = GEN1_BOOST_KEYS.index("def")
        self.assertAlmostEqual(float(self.arr[_BOOSTS_S][def_idx]), 0.0, places=5)

    def test_boosts_slice_all_zeros_for_no_boosts(self):
        # Starmie has no boosts
        arr = MyPokemonStateGen1(pokemon=make_mock_from_fixture("gen1_starmie.json")).to_array()
        np.testing.assert_array_almost_equal(arr[_BOOSTS_S], np.zeros(_N_BOOSTS))


# ---------------------------------------------------------------------------
# to_array() — None pokemon: full array verification
# Every slice must be zero except stab which defaults to 1.5/STAB_NORM
# ---------------------------------------------------------------------------

class TestMyPokemonStateGen1ToArrayEmpty(unittest.TestCase):

    def setUp(self):
        self.arr = MyPokemonStateGen1().to_array()

    def test_output_dtype_float32(self):
        self.assertEqual(self.arr.dtype, np.float32)

    def test_hp_is_zero(self):
        self.assertEqual(float(self.arr[_HP_IDX]), 0.0)

    def test_stats_all_zero(self):
        np.testing.assert_array_equal(self.arr[_STATS_S], np.zeros(_N_STATS))

    def test_boosts_all_zero(self):
        np.testing.assert_array_equal(self.arr[_BOOSTS_S], np.zeros(_N_BOOSTS))

    def test_status_all_zero(self):
        np.testing.assert_array_equal(self.arr[_STATUS_S], np.zeros(_N_STATUSES))

    def test_effects_all_zero(self):
        np.testing.assert_array_equal(self.arr[_EFFECTS_S], np.zeros(_N_EFFECTS))

    def test_stab_is_default_normalised(self):
        self.assertAlmostEqual(float(self.arr[_STAB_IDX]), 1.5 / STAB_NORM, places=5)


# ---------------------------------------------------------------------------
# to_array() — pokemon given: every slice contains the right normalised value
# ---------------------------------------------------------------------------

class TestMyPokemonStateGen1ToArrayPopulated(unittest.TestCase):

    def test_hp_at_correct_position(self):
        # Chansey: hp_fraction=0.5
        arr = MyPokemonStateGen1(pokemon=make_mock_from_fixture("gen1_chansey.json")).to_array()
        self.assertAlmostEqual(float(arr[_HP_IDX]), 0.5, places=5)

    def test_hp_full(self):
        # Starmie: hp_fraction=1.0
        arr = MyPokemonStateGen1(pokemon=make_mock_from_fixture("gen1_starmie.json")).to_array()
        self.assertAlmostEqual(float(arr[_HP_IDX]), 1.0, places=5)

    def test_status_slice_correct_position(self):
        # Chansey: BRN → correct index in status slice is 1.0, rest 0
        arr = MyPokemonStateGen1(pokemon=make_mock_from_fixture("gen1_chansey.json")).to_array()
        brn_idx = ALL_STATUSES.index(Status.BRN)
        self.assertEqual(float(arr[_STATUS_S][brn_idx]), 1.0)
        self.assertEqual(float(arr[_STATUS_S].sum()), 1.0)

    def test_status_slice_no_status_all_zero(self):
        # Tauros: no status
        arr = MyPokemonStateGen1(pokemon=make_mock_from_fixture("gen1_tauros.json")).to_array()
        self.assertEqual(float(arr[_STATUS_S].sum()), 0.0)

    def test_effects_slice_active_effect(self):
        # Inject CONFUSION into Starmie mock
        mock = make_mock_from_fixture("gen1_starmie.json")
        mock.effects = {Effect.CONFUSION: MagicMock()}
        arr = MyPokemonStateGen1(pokemon=mock).to_array()
        confusion_idx = GEN1_TRACKED_EFFECTS.index(Effect.CONFUSION)
        self.assertEqual(float(arr[_EFFECTS_S][confusion_idx]), 1.0)

    def test_effects_slice_no_effects_all_zero(self):
        arr = MyPokemonStateGen1(pokemon=make_mock_from_fixture("gen1_starmie.json")).to_array()
        np.testing.assert_array_equal(arr[_EFFECTS_S], np.zeros(_N_EFFECTS))

    def test_all_values_bounded(self):
        # Every element must be in [-1, 1] — boosts are symmetric, rest are [0,1]
        for fixture in ["gen1_starmie.json", "gen1_chansey.json",
                        "gen1_tauros.json", "gen1_alakazam.json"]:
            arr = MyPokemonStateGen1(pokemon=make_mock_from_fixture(fixture)).to_array()
            self.assertTrue(np.all(arr >= -1.0) and np.all(arr <= 1.0), msg=fixture)

    def test_output_dtype_float32(self):
        arr = MyPokemonStateGen1(pokemon=make_mock_from_fixture("gen1_starmie.json")).to_array()
        self.assertEqual(arr.dtype, np.float32)


# ---------------------------------------------------------------------------
# to_array() stab scalar — normalised by STAB_NORM
# ---------------------------------------------------------------------------

class TestMyPokemonStateGen1StabSlice(unittest.TestCase):

    def test_stab_normalised(self):
        # All fixtures have stab_multiplier=1.5
        arr = MyPokemonStateGen1(pokemon=make_mock_from_fixture("gen1_starmie.json")).to_array()
        self.assertAlmostEqual(float(arr[_STAB_IDX]), 1.5 / STAB_NORM, places=5)


# ---------------------------------------------------------------------------
# describe() — Stats line and Boosts line
# ---------------------------------------------------------------------------

class TestMyPokemonStateGen1Describe(unittest.TestCase):

    def test_stats_line_present(self):
        ps = MyPokemonStateGen1(pokemon=make_mock_from_fixture("gen1_starmie.json"))
        self.assertIn("Stats", ps.describe())

    def test_stats_line_contains_all_keys(self):
        ps = MyPokemonStateGen1(pokemon=make_mock_from_fixture("gen1_starmie.json"))
        for k in GEN1_STAT_KEYS:
            self.assertIn(k, ps.describe(), msg=f"Key '{k}' missing from Stats line")

    def test_boosts_line_shows_none_when_no_boosts(self):
        # Starmie: all boosts zero
        ps = MyPokemonStateGen1(pokemon=make_mock_from_fixture("gen1_starmie.json"))
        lines = {l.split(":")[0].strip(): l for l in ps.describe().splitlines()}
        self.assertIn("none", lines.get("Boosts", ""))

    def test_boosts_line_shows_active_boost(self):
        # Tauros: atk=+2 must appear
        ps = MyPokemonStateGen1(pokemon=make_mock_from_fixture("gen1_tauros.json"))
        lines = {l.split(":")[0].strip(): l for l in ps.describe().splitlines()}
        self.assertNotIn("none", lines.get("Boosts", ""))
        self.assertIn("atk", lines.get("Boosts", ""))

    def test_boosts_line_shows_spa_boost(self):
        # Alakazam: spa=+2 must appear
        ps = MyPokemonStateGen1(pokemon=make_mock_from_fixture("gen1_alakazam.json"))
        lines = {l.split(":")[0].strip(): l for l in ps.describe().splitlines()}
        self.assertIn("spa", lines.get("Boosts", ""))


if __name__ == "__main__":
    unittest.main()
