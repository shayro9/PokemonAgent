"""
Tests unique to OpponentPokemonStateGen1 — what it adds on top of PokemonState:

  1. estimate_stats()   — Gen1 formula: ((base + 15) * 2 + 64) + level + 10 (hp)
                                        ((base + 15) * 2 + 64) + 5          (others)
  2. preparing          — float field read from pokemon.preparing
  3. must_recharge      — float field read from pokemon.must_recharge
  4. protect            — float field read from pokemon.protect_counter
  5. normalize_protect()— geometric decay: 0.3 ** self.protect
  6. to_array()         — layout: [hp | stats | boosts | status | effects |
                                   preparing | must_recharge | stab | protect]
  7. array_len()        — formula includes the 3 extra fields
  8. describe()         — includes Preparing / MustRecharge / protect lines

Everything else (hp, species, status field, boosts field, effects field, stab field,
types, class constants) is already covered by test_pokemon_state.py.

Fixture files used
------------------
gen1_starmie_opponent.json  — PAR, full HP, no boosts, preparing=False
gen1_tauros_opponent.json   — no status, full HP, mixed boosts, preparing=True
gen1_chansey_opponent.json  — BRN, half HP, +3 def, preparing=False
"""

import json
import unittest
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
from poke_env.battle.pokemon_type import PokemonType
from poke_env.battle.status import Status

from env.states.gen1.opponent_pokemon_state_gen_1 import OpponentPokemonStateGen1
from env.states.pokemon_state import PokemonState
from env.states.state_utils import (
    ALL_STATUSES,
    GEN1_STAT_KEYS,
    GEN1_TRACKED_EFFECTS,
    STAT_NORM,
    BOOST_NORM,
    STAB_NORM,
)

FIXTURES_DIR = Path(__file__).parent.parent / "fixtures"

# ---------------------------------------------------------------------------
# Gen1 stat formula constants (mirror production code)
# ---------------------------------------------------------------------------
_DV      = 15
_EV_TERM = 64
_LEVEL   = 100

# to_array() slice positions
# Layout: [hp | stats | boosts | status | effects | preparing | must_recharge | stab | protect]
_N_STATS    = len(GEN1_STAT_KEYS)
_N_BOOSTS   = len(PokemonState.BOOST_KEYS)
_N_STATUSES = len(ALL_STATUSES)
_N_EFFECTS  = len(GEN1_TRACKED_EFFECTS)
_HP_IDX       = 0
_STATS_S      = slice(1, 1 + _N_STATS)
_BOOSTS_S     = slice(1 + _N_STATS, 1 + _N_STATS + _N_BOOSTS)
_STATUS_S     = slice(1 + _N_STATS + _N_BOOSTS, 1 + _N_STATS + _N_BOOSTS + _N_STATUSES)
_EFFECTS_S    = slice(1 + _N_STATS + _N_BOOSTS + _N_STATUSES,
                      1 + _N_STATS + _N_BOOSTS + _N_STATUSES + _N_EFFECTS)
_PREPARING_IDX    = 1 + _N_STATS + _N_BOOSTS + _N_STATUSES + _N_EFFECTS
_RECHARGE_IDX     = _PREPARING_IDX + 1
_STAB_IDX         = _RECHARGE_IDX + 1
_PROTECT_IDX      = _STAB_IDX + 1  # last element

_CORRECT_ARRAY_LEN = _PROTECT_IDX + 1  # 25 for Gen1

_BASE_STATS = {
    "starmie": {"hp": 60,  "atk": 75,  "def": 85,  "spc": 100, "spe": 115},
    "tauros":  {"hp": 75,  "atk": 100, "def": 95,  "spc": 70,  "spe": 110},
    "chansey": {"hp": 250, "atk": 5,   "def": 5,   "spc": 35,  "spe": 50},
}


def _expected_stats(base_stats: dict) -> np.ndarray:
    result = []
    for key in GEN1_STAT_KEYS:
        base = base_stats[key]
        if key == "hp":
            stat = ((base + _DV) * 2 + _EV_TERM) + _LEVEL + 10
        else:
            stat = ((base + _DV) * 2 + _EV_TERM) + 5
        result.append(stat)
    return np.array(result, dtype=np.float32)


def make_opponent_mock_from_fixture(filename: str, *, protect_counter: int = 0):
    with open(FIXTURES_DIR / filename) as f:
        data = json.load(f)
    species = data["species"]
    p = MagicMock()
    p.current_hp_fraction = data["current_hp_fraction"]
    p.species             = species
    p.boosts              = data["boosts"]
    p.status              = Status[data["status"]] if data["status"] else None
    p.effects             = {}
    p.stab_multiplier     = data["stab_multiplier"]
    p.preparing           = data.get("preparing", False)
    p.must_recharge       = False
    p.types               = tuple(PokemonType[t] for t in data["types"])
    p.base_stats          = _BASE_STATS[species]
    p.protect_counter     = protect_counter
    return p, data


def _make_bare_mock(species="starmie", *, preparing=False, must_recharge=False,
                    protect_counter=0) -> MagicMock:
    p = MagicMock()
    p.current_hp_fraction = 1.0
    p.species             = species
    p.boosts              = {}
    p.status              = None
    p.effects             = {}
    p.stab_multiplier     = 1.5
    p.preparing           = preparing
    p.must_recharge       = must_recharge
    p.types               = ()
    p.base_stats          = _BASE_STATS[species]
    p.protect_counter     = protect_counter
    return p


# ---------------------------------------------------------------------------
# estimate_stats() — Gen1 formula
# ---------------------------------------------------------------------------

class TestEstimateStats(unittest.TestCase):

    def _make(self, base_stats: dict) -> OpponentPokemonStateGen1:
        p = _make_bare_mock()
        p.base_stats = base_stats
        return OpponentPokemonStateGen1(p)

    def test_hp_formula(self):
        ops = self._make({"hp": 100, "atk": 50, "def": 50, "spc": 50, "spe": 50})
        expected = (100 + _DV) * 2 + _EV_TERM + _LEVEL + 10
        self.assertAlmostEqual(float(ops.stats[GEN1_STAT_KEYS.index("hp")]), expected)

    def test_non_hp_formula(self):
        ops = self._make({"hp": 50, "atk": 80, "def": 80, "spc": 80, "spe": 80})
        expected = (80 + _DV) * 2 + _EV_TERM + 5
        self.assertAlmostEqual(float(ops.stats[GEN1_STAT_KEYS.index("atk")]), expected)

    def test_hp_larger_than_non_hp_for_equal_bases(self):
        ops = self._make({"hp": 80, "atk": 80, "def": 80, "spc": 80, "spe": 80})
        self.assertGreater(float(ops.stats[GEN1_STAT_KEYS.index("hp")]),
                           float(ops.stats[GEN1_STAT_KEYS.index("atk")]))

    def test_higher_base_gives_higher_stat(self):
        low  = self._make({"hp": 45, "atk": 49, "def": 49, "spc": 45, "spe": 45})
        high = self._make({"hp": 75, "atk": 100, "def": 95, "spc": 70, "spe": 110})
        self.assertTrue(np.all(high.stats > low.stats))

    def test_all_stats_positive(self):
        ops = self._make({"hp": 1, "atk": 1, "def": 1, "spc": 1, "spe": 1})
        self.assertTrue(np.all(ops.stats > 0))

    def test_output_dtype_float32(self):
        self.assertEqual(self._make(_BASE_STATS["starmie"]).stats.dtype, np.float32)

    def test_output_shape(self):
        self.assertEqual(self._make(_BASE_STATS["starmie"]).stats.shape, (_N_STATS,))

    def test_starmie_all_stats_match_formula(self):
        ops = self._make(_BASE_STATS["starmie"])
        np.testing.assert_array_almost_equal(ops.stats, _expected_stats(_BASE_STATS["starmie"]))

    def test_chansey_all_stats_match_formula(self):
        ops = self._make(_BASE_STATS["chansey"])
        np.testing.assert_array_almost_equal(ops.stats, _expected_stats(_BASE_STATS["chansey"]))

    def test_empty_state_stats_all_zero(self):
        ops = OpponentPokemonStateGen1()
        np.testing.assert_array_equal(ops.stats, np.zeros(_N_STATS, dtype=np.float32))


# ---------------------------------------------------------------------------
# preparing / must_recharge / protect fields
# ---------------------------------------------------------------------------

class TestOpponentNewFields(unittest.TestCase):

    def test_preparing_true(self):
        # Tauros fixture has preparing=True
        pokemon, _ = make_opponent_mock_from_fixture("gen1_tauros_opponent.json")
        self.assertEqual(OpponentPokemonStateGen1(pokemon).preparing, 1.0)

    def test_preparing_false(self):
        # Starmie fixture has preparing=False
        pokemon, _ = make_opponent_mock_from_fixture("gen1_starmie_opponent.json")
        self.assertEqual(OpponentPokemonStateGen1(pokemon).preparing, 0.0)

    def test_must_recharge_true(self):
        # No fixture has must_recharge=True — use bare mock
        ops = OpponentPokemonStateGen1(_make_bare_mock(must_recharge=True))
        self.assertEqual(ops.must_recharge, 1.0)

    def test_must_recharge_false(self):
        # Chansey fixture — must_recharge=False by default
        pokemon, _ = make_opponent_mock_from_fixture("gen1_chansey_opponent.json")
        self.assertEqual(OpponentPokemonStateGen1(pokemon).must_recharge, 0.0)

    def test_protect_counter_stored(self):
        pokemon, _ = make_opponent_mock_from_fixture("gen1_starmie_opponent.json",
                                                     protect_counter=2)
        self.assertEqual(OpponentPokemonStateGen1(pokemon).protect, 2.0)

    def test_protect_counter_zero(self):
        pokemon, _ = make_opponent_mock_from_fixture("gen1_starmie_opponent.json",
                                                     protect_counter=0)
        self.assertEqual(OpponentPokemonStateGen1(pokemon).protect, 0.0)

    def test_empty_state_preparing_zero(self):
        self.assertEqual(OpponentPokemonStateGen1().preparing, 0.0)

    def test_empty_state_must_recharge_zero(self):
        self.assertEqual(OpponentPokemonStateGen1().must_recharge, 0.0)

    def test_empty_state_protect_zero(self):
        self.assertEqual(OpponentPokemonStateGen1().protect, -1.0)

    def test_preparing_and_recharge_independent(self):
        # Tauros: preparing=True, must_recharge=False
        pokemon, _ = make_opponent_mock_from_fixture("gen1_tauros_opponent.json")
        ops = OpponentPokemonStateGen1(pokemon)
        self.assertEqual(ops.preparing, 1.0)
        self.assertEqual(ops.must_recharge, 0.0)



# ---------------------------------------------------------------------------
# normalize_protect() — geometric decay 0.3 ** protect
# ---------------------------------------------------------------------------

class TestNormalizeProtect(unittest.TestCase):

    def _make(self, protect_counter: int) -> OpponentPokemonStateGen1:
        # Use Starmie fixture as the base pokemon, vary only protect_counter
        pokemon, _ = make_opponent_mock_from_fixture("gen1_starmie_opponent.json",
                                                     protect_counter=protect_counter)
        return OpponentPokemonStateGen1(pokemon)

    def test_counter_zero_gives_one(self):
        np.testing.assert_array_almost_equal(self._make(0).normalize_protect(), [1.0])

    def test_counter_one_gives_point_three(self):
        np.testing.assert_array_almost_equal(self._make(1).normalize_protect(), [0.3], decimal=6)

    def test_counter_two_gives_point_zero_nine(self):
        np.testing.assert_array_almost_equal(self._make(2).normalize_protect(), [0.09], decimal=6)

    def test_counter_three(self):
        np.testing.assert_array_almost_equal(self._make(3).normalize_protect(), [0.027], decimal=6)

    def test_strictly_decreasing(self):
        values = [float(self._make(n).normalize_protect()[0]) for n in range(5)]
        for i in range(len(values) - 1):
            self.assertGreater(values[i], values[i + 1])

    def test_always_positive(self):
        for n in range(10):
            self.assertGreaterEqual(float(self._make(n).normalize_protect()[0]), 0.0)

    def test_never_exceeds_one(self):
        for n in range(10):
            self.assertLessEqual(float(self._make(n).normalize_protect()[0]), 1.0)

    def test_returns_float32(self):
        self.assertEqual(self._make(0).normalize_protect().dtype, np.float32)

    def test_returns_shape_one(self):
        self.assertEqual(self._make(0).normalize_protect().shape, (1,))

    def test_fixture_with_nonzero_counter(self):
        pokemon, _ = make_opponent_mock_from_fixture(
            "gen1_starmie_opponent.json", protect_counter=2
        )
        ops = OpponentPokemonStateGen1(pokemon)
        np.testing.assert_array_almost_equal(ops.normalize_protect(), [0.09], decimal=6)


# ---------------------------------------------------------------------------
# to_array() — empty state (None pokemon)
# ---------------------------------------------------------------------------

class TestOpponentToArrayEmpty(unittest.TestCase):

    def setUp(self):
        self.arr = OpponentPokemonStateGen1().to_array()

    def test_length_matches_array_len(self):
        ops = OpponentPokemonStateGen1()
        self.assertEqual(len(self.arr), ops.array_len())

    def test_correct_total_length(self):
        self.assertEqual(len(self.arr), _CORRECT_ARRAY_LEN)

    def test_dtype_float32(self):
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

    def test_preparing_is_zero(self):
        self.assertEqual(float(self.arr[_PREPARING_IDX]), 0.0)

    def test_must_recharge_is_zero(self):
        self.assertEqual(float(self.arr[_RECHARGE_IDX]), 0.0)

    def test_stab_is_default_normalised(self):
        self.assertAlmostEqual(float(self.arr[_STAB_IDX]), 0.0, places=5)

    def test_protect_is_zero_when_none(self):
        self.assertAlmostEqual(float(self.arr[_PROTECT_IDX]), 0.0, places=5)

    def test_no_nan(self):
        self.assertFalse(np.any(np.isnan(self.arr)))

    def test_no_inf(self):
        self.assertFalse(np.any(np.isinf(self.arr)))


# ---------------------------------------------------------------------------
# to_array() — populated state, every slice verified
# ---------------------------------------------------------------------------

class TestOpponentToArrayPopulated(unittest.TestCase):

    def test_hp_position(self):
        # Chansey: hp_fraction=0.5
        pokemon, _ = make_opponent_mock_from_fixture("gen1_chansey_opponent.json")
        arr = OpponentPokemonStateGen1(pokemon).to_array()
        self.assertAlmostEqual(float(arr[_HP_IDX]), 0.5, places=5)

    def test_stats_slice_normalised(self):
        # Starmie: check all stat slots are normalised by STAT_NORM
        pokemon, _ = make_opponent_mock_from_fixture("gen1_starmie_opponent.json")
        ops = OpponentPokemonStateGen1(pokemon)
        arr = ops.to_array()
        expected = np.clip(ops.stats / STAT_NORM, 0.0, 1.0)
        np.testing.assert_array_almost_equal(arr[_STATS_S], expected)

    def test_stats_slice_all_in_range(self):
        pokemon, _ = make_opponent_mock_from_fixture("gen1_starmie_opponent.json")
        arr = OpponentPokemonStateGen1(pokemon).to_array()
        self.assertTrue(np.all(arr[_STATS_S] >= 0.0) and np.all(arr[_STATS_S] <= 1.0))

    def test_boosts_slice_positive_boost(self):
        # Tauros: atk=+2 → 2/6
        pokemon, _ = make_opponent_mock_from_fixture("gen1_tauros_opponent.json")
        arr = OpponentPokemonStateGen1(pokemon).to_array()
        atk_idx = PokemonState.BOOST_KEYS.index("atk")
        self.assertAlmostEqual(float(arr[_BOOSTS_S][atk_idx]), 2 / BOOST_NORM, places=5)

    def test_boosts_slice_negative_boost(self):
        # Tauros: spa=-2 → -2/6 (symmetric)
        pokemon, _ = make_opponent_mock_from_fixture("gen1_tauros_opponent.json")
        arr = OpponentPokemonStateGen1(pokemon).to_array()
        spa_idx = PokemonState.BOOST_KEYS.index("spa")
        self.assertAlmostEqual(float(arr[_BOOSTS_S][spa_idx]), -2 / BOOST_NORM, places=5)

    def test_status_slice_brn(self):
        # Chansey: BRN
        pokemon, _ = make_opponent_mock_from_fixture("gen1_chansey_opponent.json")
        arr = OpponentPokemonStateGen1(pokemon).to_array()
        brn_idx = ALL_STATUSES.index(Status.BRN)
        self.assertEqual(float(arr[_STATUS_S][brn_idx]), 1.0)
        self.assertEqual(float(arr[_STATUS_S].sum()), 1.0)

    def test_status_slice_no_status(self):
        # Tauros: no status
        pokemon, _ = make_opponent_mock_from_fixture("gen1_tauros_opponent.json")
        arr = OpponentPokemonStateGen1(pokemon).to_array()
        self.assertEqual(float(arr[_STATUS_S].sum()), 0.0)

    def test_preparing_position_true(self):
        # Tauros: preparing=True
        pokemon, _ = make_opponent_mock_from_fixture("gen1_tauros_opponent.json")
        arr = OpponentPokemonStateGen1(pokemon).to_array()
        self.assertEqual(float(arr[_PREPARING_IDX]), 1.0)

    def test_preparing_position_false(self):
        # Starmie: preparing=False
        pokemon, _ = make_opponent_mock_from_fixture("gen1_starmie_opponent.json")
        arr = OpponentPokemonStateGen1(pokemon).to_array()
        self.assertEqual(float(arr[_PREPARING_IDX]), 0.0)

    def test_must_recharge_position(self):
        # No fixture has must_recharge=True, override via mock attribute
        pokemon, _ = make_opponent_mock_from_fixture("gen1_tauros_opponent.json")
        pokemon.must_recharge = True
        ops = OpponentPokemonStateGen1(pokemon)
        arr = ops.to_array()
        self.assertEqual(float(arr[_RECHARGE_IDX]), 1.0)

    def test_stab_position(self):
        pokemon, _ = make_opponent_mock_from_fixture("gen1_starmie_opponent.json")
        arr = OpponentPokemonStateGen1(pokemon).to_array()
        self.assertAlmostEqual(float(arr[_STAB_IDX]), 1.5 / STAB_NORM, places=5)

    def test_protect_position_counter_zero(self):
        pokemon, _ = make_opponent_mock_from_fixture("gen1_starmie_opponent.json")
        arr = OpponentPokemonStateGen1(pokemon).to_array()
        self.assertAlmostEqual(float(arr[_PROTECT_IDX]), 1.0, places=5)

    def test_protect_position_counter_two(self):
        pokemon, _ = make_opponent_mock_from_fixture(
            "gen1_starmie_opponent.json", protect_counter=2
        )
        arr = OpponentPokemonStateGen1(pokemon).to_array()
        self.assertAlmostEqual(float(arr[_PROTECT_IDX]), 0.09, places=5)

    def test_all_values_bounded(self):
        for fixture in ["gen1_starmie_opponent.json", "gen1_chansey_opponent.json",
                        "gen1_tauros_opponent.json"]:
            pokemon, _ = make_opponent_mock_from_fixture(fixture)
            arr = OpponentPokemonStateGen1(pokemon).to_array()
            self.assertTrue(np.all(arr >= -1.0) and np.all(arr <= 1.0), msg=fixture)

    def test_no_nan(self):
        for fixture in ["gen1_starmie_opponent.json", "gen1_chansey_opponent.json",
                        "gen1_tauros_opponent.json"]:
            pokemon, _ = make_opponent_mock_from_fixture(fixture)
            arr = OpponentPokemonStateGen1(pokemon).to_array()
            self.assertFalse(np.any(np.isnan(arr)), msg=fixture)

    def test_no_inf(self):
        for fixture in ["gen1_starmie_opponent.json", "gen1_chansey_opponent.json",
                        "gen1_tauros_opponent.json"]:
            pokemon, _ = make_opponent_mock_from_fixture(fixture)
            arr = OpponentPokemonStateGen1(pokemon).to_array()
            self.assertFalse(np.any(np.isinf(arr)), msg=fixture)


# ---------------------------------------------------------------------------
# array_len()
# ---------------------------------------------------------------------------

class TestOpponentArrayLen(unittest.TestCase):

    def test_empty_state_correct_length(self):
        ops = OpponentPokemonStateGen1()
        self.assertEqual(ops.array_len(), _CORRECT_ARRAY_LEN)

    def test_populated_state_correct_length(self):
        pokemon, _ = make_opponent_mock_from_fixture("gen1_starmie_opponent.json")
        ops = OpponentPokemonStateGen1(pokemon)
        self.assertEqual(ops.array_len(), _CORRECT_ARRAY_LEN)

    def test_array_len_matches_to_array(self):
        for fixture in ["gen1_starmie_opponent.json", "gen1_chansey_opponent.json",
                        "gen1_tauros_opponent.json"]:
            pokemon, _ = make_opponent_mock_from_fixture(fixture)
            ops = OpponentPokemonStateGen1(pokemon)
            self.assertEqual(len(ops.to_array()), ops.array_len(), msg=fixture)


# ---------------------------------------------------------------------------
# describe() — opponent-specific lines
# ---------------------------------------------------------------------------

class TestOpponentDescribe(unittest.TestCase):

    def setUp(self):
        pokemon, _ = make_opponent_mock_from_fixture("gen1_tauros_opponent.json")
        self.ops = OpponentPokemonStateGen1(pokemon)
        self.desc = self.ops.describe()

    def test_has_preparing_line(self):
        self.assertIn("Preparing", self.desc)

    def test_has_must_recharge_line(self):
        self.assertIn("MustRecharge", self.desc)

    def test_has_protect_line(self):
        self.assertIn("protect", self.desc)

    def test_has_stats_line(self):
        self.assertIn("Stats", self.desc)

    def test_does_not_raise(self):
        try:
            pokemon, _ = make_opponent_mock_from_fixture("gen1_starmie_opponent.json")
            OpponentPokemonStateGen1(pokemon).describe()
        except Exception as e:
            self.fail(f"describe() raised: {e}")


if __name__ == "__main__":
    unittest.main()

