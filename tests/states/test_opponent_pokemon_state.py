"""
Tests for OpponentPokemonState.

Fixture files used
------------------
gen1_starmie_opponent.json  — PAR, full HP, no boosts, not preparing
gen1_tauros_opponent.json   — no status, full HP, boosted, preparing=True
gen1_chansey_opponent.json  — BRN, half HP, +3 def

What changed from the previous test suite
------------------------------------------
* did_protect() is gone — protect is now READ from pokemon.protect_counter
  at construction time via the protect_counter() static method.
* normalize_protect() changed from linear capping to geometric decay:
      value = 0.3 ** self.protect
  so protect=0 → 1.0, protect=1 → 0.3, protect=2 → 0.09, etc.

Known bugs in the production code (tests below expose them)
------------------------------------------------------------
BUG-1  array_len() still uses _STAT_BELIEF_DIM (12) instead of
       len(STAT_KEYS) (5 for Gen 1) and does not count must_recharge.
       Correct Gen 1 total:
         1 hp + 5 stats + 6 boosts + 7 status + 2 effects
         + 1 preparing + 1 must_recharge + 1 stab + 1 protect = 25

BUG-2  describe() references self.stats_std and self.protect_belief,
       neither of which exists, raising AttributeError.
"""

import json
import math
import unittest
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np

from poke_env.battle.pokemon_type import PokemonType
from poke_env.battle.status import Status

from env.states.opponent_pokemon_state import OpponentPokemonState
from env.states.pokemon_state import (
    PokemonState,
    ALL_STATUSES,
    TRACKED_EFFECTS,
    GEN1_STAT_KEYS,
)

FIXTURES_DIR = Path(__file__).parent.parent / "fixtures"

# ── Gen 1 formula constants (mirror production code) ────────────────────────
_DV      = 15
_EV      = 65535
_EV_TERM = int(math.sqrt(_EV) / 4)   # 63
_LEVEL   = 100

# Correct array length for Gen 1 opponent (see BUG-1 above)
_CORRECT_ARRAY_LEN = (
    1                               # hp
    + len(GEN1_STAT_KEYS)           # 5  estimated stats
    + len(PokemonState.BOOST_KEYS)  # 6  boosts
    + len(ALL_STATUSES)             # 7  status
    + len(TRACKED_EFFECTS)          # 2  confusion, encore
    + 1                             # preparing
    + 1                             # must_recharge
    + 1                             # stab
    + 1                             # protect (normalised)
)  # = 25

_BASE_STATS = {
    "starmie": {"hp": 60,  "atk": 75,  "def": 85, "spc": 100, "spe": 115},
    "tauros":  {"hp": 75,  "atk": 100, "def": 95, "spc": 70,  "spe": 110},
    "chansey": {"hp": 250, "atk": 5,   "def": 5,  "spc": 35,  "spe": 50},
}


def _expected_estimated_stats(base_stats: dict) -> np.ndarray:
    result = []
    for key in GEN1_STAT_KEYS:
        base = base_stats[key]
        if key == "hp":
            stat = ((base + _DV) * 2 + _EV_TERM) + _LEVEL + 10
        else:
            stat = ((base + _DV) * 2 + _EV_TERM) + 5
        result.append(stat)
    return np.array(result, dtype=np.float32)


# ---------------------------------------------------------------------------
# Fixture helper
# ---------------------------------------------------------------------------

def make_opponent_mock_from_fixture(filename: str, *, protect_counter: int = 0):
    """Load an opponent fixture and return (mock_pokemon, raw_data)."""
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
    """Minimal mock for unit tests that don't need a fixture."""
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
# Shared structural base mixin
# ---------------------------------------------------------------------------

class OpponentPokemonStateBaseTest:
    """Structural checks every fixture test class must pass."""
    ops: OpponentPokemonState

    # ── estimated stats ──────────────────────────────────────────────────

    def test_stats_shape(self):
        self.assertEqual(self.ops.stats.shape, (len(GEN1_STAT_KEYS),))

    def test_stats_dtype(self):
        self.assertEqual(self.ops.stats.dtype, np.float32)

    def test_stats_all_positive(self):
        self.assertTrue(np.all(self.ops.stats > 0))

    def test_normalize_stats_in_range(self):
        enc = self.ops.normalize_stats()
        self.assertTrue(np.all(enc >= 0.0) and np.all(enc <= 1.0))

    def test_normalize_stats_shape(self):
        self.assertEqual(self.ops.normalize_stats().shape, (len(GEN1_STAT_KEYS),))

    # ── boosts ───────────────────────────────────────────────────────────

    def test_boosts_shape(self):
        self.assertEqual(self.ops.boosts.shape, (len(PokemonState.BOOST_KEYS),))

    def test_boosts_dtype(self):
        self.assertEqual(self.ops.boosts.dtype, np.float32)

    def test_boosts_encoded_in_range(self):
        enc = self.ops.normalize_boosts()
        self.assertTrue(np.all(enc >= -1.0) and np.all(enc <= 1.0))

    # ── status ───────────────────────────────────────────────────────────

    def test_status_shape(self):
        self.assertEqual(self.ops.status.shape, (len(ALL_STATUSES),))

    def test_status_dtype(self):
        self.assertEqual(self.ops.status.dtype, np.float32)

    def test_status_at_most_one_hot(self):
        self.assertIn(self.ops.status.sum(), [0.0, 1.0])

    # ── effects ──────────────────────────────────────────────────────────

    def test_effects_shape(self):
        self.assertEqual(self.ops.effects.shape, (len(TRACKED_EFFECTS),))

    def test_effects_dtype(self):
        self.assertEqual(self.ops.effects.dtype, np.float32)

    def test_effects_has_two_slots(self):
        self.assertEqual(len(TRACKED_EFFECTS), 2)

    # ── opponent-specific scalars ─────────────────────────────────────────

    def test_preparing_is_float(self):
        self.assertIsInstance(self.ops.preparing, float)

    def test_preparing_is_binary(self):
        self.assertIn(self.ops.preparing, [0.0, 1.0])

    def test_must_recharge_is_float(self):
        self.assertIsInstance(self.ops.must_recharge, float)

    def test_must_recharge_is_binary(self):
        self.assertIn(self.ops.must_recharge, [0.0, 1.0])

    def test_protect_is_numeric(self):
        self.assertIsInstance(self.ops.protect, (int, float))

    def test_protect_non_negative(self):
        self.assertGreaterEqual(self.ops.protect, 0)

    # ── stab ─────────────────────────────────────────────────────────────

    def test_stab_raw_value(self):
        self.assertEqual(self.ops.stab, 0.75)

    # ── to_array: dtype / key positions ──────────────────────────────────

    def test_to_array_dtype(self):
        self.assertEqual(self.ops.to_array().dtype, np.float32)

    def test_to_array_hp_first(self):
        self.assertAlmostEqual(float(self.ops.to_array()[0]), self.ops.hp)

    def test_to_array_protect_last(self):
        arr = self.ops.to_array()
        self.assertAlmostEqual(float(arr[-1]), float(self.ops.normalize_protect()[0]))

    def test_to_array_stat_slice(self):
        n = len(GEN1_STAT_KEYS)
        arr = self.ops.to_array()
        np.testing.assert_array_almost_equal(arr[1: 1 + n], self.ops.normalize_stats())


# ---------------------------------------------------------------------------
# BUG EXPOSURE TESTS
# ---------------------------------------------------------------------------

class TestOpponentArrayLenBug(unittest.TestCase):
    """
    BUG-1: array_len() returns 20 (using _STAT_BELIEF_DIM=12, omitting
    must_recharge) but to_array() produces 25 elements, so the internal
    assert always fires.  These tests FAIL until the bug is fixed.
    """

    def setUp(self):
        self.ops = OpponentPokemonState(_make_bare_mock())

    def test_array_len_correct_value(self):
        self.assertEqual(self.ops.array_len(), _CORRECT_ARRAY_LEN)

    def test_to_array_does_not_raise(self):
        try:
            self.ops.to_array()
        except AssertionError as e:
            self.fail(f"to_array() raised AssertionError: {e}")

    def test_to_array_length_matches_array_len(self):
        self.assertEqual(len(self.ops.to_array()), self.ops.array_len())


class TestOpponentDescribeBug(unittest.TestCase):
    """
    BUG-2: describe() references self.stats_std and self.protect_belief
    which no longer exist.  These tests FAIL until the bug is fixed.
    """

    def setUp(self):
        self.ops = OpponentPokemonState(_make_bare_mock())

    def test_describe_does_not_raise(self):
        try:
            self.ops.describe()
        except AttributeError as e:
            self.fail(f"describe() raised AttributeError: {e}")

    def test_repr_does_not_raise(self):
        try:
            repr(self.ops)
        except AttributeError as e:
            self.fail(f"__repr__ raised AttributeError: {e}")


# ---------------------------------------------------------------------------
# Empty state
# ---------------------------------------------------------------------------

class TestOpponentPokemonStateEmpty(OpponentPokemonStateBaseTest, unittest.TestCase):

    def setUp(self):
        self.ops = OpponentPokemonState()

    # Zeros-specific overrides
    def test_stats_all_positive(self):
        np.testing.assert_array_equal(self.ops.stats, np.zeros(len(GEN1_STAT_KEYS), dtype=np.float32))

    def test_hp_is_zero(self):
        self.assertEqual(self.ops.hp, 0.0)

    def test_species_is_none(self):
        self.assertEqual(self.ops.species, "none")

    def test_boosts_all_zero(self):
        np.testing.assert_array_equal(
            self.ops.boosts, np.zeros(len(PokemonState.BOOST_KEYS), dtype=np.float32)
        )

    def test_status_all_zero(self):
        np.testing.assert_array_equal(
            self.ops.status, np.zeros(len(ALL_STATUSES), dtype=np.float32)
        )

    def test_effects_all_zero(self):
        np.testing.assert_array_equal(
            self.ops.effects, np.zeros(len(TRACKED_EFFECTS), dtype=np.float32)
        )

    def test_preparing_is_zero(self):
        self.assertEqual(self.ops.preparing, 0.0)

    def test_must_recharge_is_zero(self):
        self.assertEqual(self.ops.must_recharge, 0.0)

    def test_protect_is_zero(self):
        self.assertEqual(self.ops.protect, 0.0)

    def test_normalize_protect_is_one_when_counter_zero(self):
        """0.3 ** 0 == 1.0 — no prior protects means full probability."""
        np.testing.assert_array_almost_equal(self.ops.normalize_protect(), [1.0])


# ---------------------------------------------------------------------------
# Starmie  —  Water/Psychic, PAR, full HP, no boosts, not preparing
# ---------------------------------------------------------------------------

class TestOpponentPokemonStateStarmie(OpponentPokemonStateBaseTest, unittest.TestCase):

    def setUp(self):
        pokemon, self.data = make_opponent_mock_from_fixture("gen1_starmie_opponent.json")
        self.ops = OpponentPokemonState(pokemon)

    def test_hp(self):
        self.assertEqual(self.ops.hp, 1.0)

    def test_species(self):
        self.assertEqual(self.ops.species, "starmie")

    def test_status_par(self):
        self.assertEqual(self.ops.status[ALL_STATUSES.index(Status.PAR)], 1.0)

    def test_only_one_status_set(self):
        self.assertEqual(self.ops.status.sum(), 1.0)

    def test_boosts_all_zero(self):
        np.testing.assert_array_equal(
            self.ops.boosts, np.zeros(len(PokemonState.BOOST_KEYS), dtype=np.float32)
        )

    def test_preparing_false(self):
        self.assertEqual(self.ops.preparing, 0.0)

    def test_must_recharge_false(self):
        self.assertEqual(self.ops.must_recharge, 0.0)

    def test_protect_counter_zero(self):
        self.assertEqual(self.ops.protect, 0.0)

    def test_normalize_protect_is_one(self):
        """protect_counter=0 → 0.3**0 = 1.0."""
        np.testing.assert_array_almost_equal(self.ops.normalize_protect(), [1.0])

    def test_estimated_stats_match_formula(self):
        expected = _expected_estimated_stats(_BASE_STATS["starmie"])
        np.testing.assert_array_almost_equal(self.ops.stats, expected)

    def test_estimated_hp_formula(self):
        base_hp = _BASE_STATS["starmie"]["hp"]
        expected_hp = (base_hp + _DV) * 2 + _EV_TERM + _LEVEL + 10
        self.assertAlmostEqual(float(self.ops.stats[GEN1_STAT_KEYS.index("hp")]), expected_hp)


# ---------------------------------------------------------------------------
# Tauros  —  Normal, no status, full HP, boosted, preparing=True
# ---------------------------------------------------------------------------

class TestOpponentPokemonStateTauros(OpponentPokemonStateBaseTest, unittest.TestCase):

    def setUp(self):
        pokemon, self.data = make_opponent_mock_from_fixture("gen1_tauros_opponent.json")
        self.ops = OpponentPokemonState(pokemon)

    def test_hp(self):
        self.assertEqual(self.ops.hp, 1.0)

    def test_species(self):
        self.assertEqual(self.ops.species, "tauros")

    def test_no_status(self):
        self.assertEqual(self.ops.status.sum(), 0.0)

    def test_preparing_true(self):
        self.assertEqual(self.ops.preparing, 1.0)

    def test_must_recharge_false(self):
        self.assertEqual(self.ops.must_recharge, 0.0)

    def test_atk_boost(self):
        self.assertEqual(self.ops.boosts[PokemonState.BOOST_KEYS.index("atk")], 2.0)

    def test_spa_boost_negative(self):
        self.assertEqual(self.ops.boosts[PokemonState.BOOST_KEYS.index("spa")], -2.0)

    def test_spe_boost(self):
        self.assertEqual(self.ops.boosts[PokemonState.BOOST_KEYS.index("spe")], 1.0)

    def test_atk_boost_encoded(self):
        atk_idx = PokemonState.BOOST_KEYS.index("atk")
        self.assertAlmostEqual(float(self.ops.normalize_boosts()[atk_idx]), 2 / 6, places=5)

    def test_spa_boost_encoded(self):
        spa_idx = PokemonState.BOOST_KEYS.index("spa")
        self.assertAlmostEqual(float(self.ops.normalize_boosts()[spa_idx]), -2 / 6, places=5)

    def test_spa_spd_equal_in_fixture(self):
        self.assertEqual(self.data["boosts"]["spa"], self.data["boosts"]["spd"])

    def test_estimated_stats_match_formula(self):
        expected = _expected_estimated_stats(_BASE_STATS["tauros"])
        np.testing.assert_array_almost_equal(self.ops.stats, expected)

    def test_estimated_non_hp_formula(self):
        for key in GEN1_STAT_KEYS:
            if key == "hp":
                continue
            base = _BASE_STATS["tauros"][key]
            expected = (base + _DV) * 2 + _EV_TERM + 5
            idx = GEN1_STAT_KEYS.index(key)
            self.assertAlmostEqual(float(self.ops.stats[idx]), expected, msg=f"stat {key}")


# ---------------------------------------------------------------------------
# Chansey  —  Normal, BRN, half HP, +3 def
# ---------------------------------------------------------------------------

class TestOpponentPokemonStateChansey(OpponentPokemonStateBaseTest, unittest.TestCase):

    def setUp(self):
        pokemon, self.data = make_opponent_mock_from_fixture("gen1_chansey_opponent.json")
        self.ops = OpponentPokemonState(pokemon)

    def test_hp_fraction(self):
        self.assertAlmostEqual(self.ops.hp, 0.5)

    def test_species(self):
        self.assertEqual(self.ops.species, "chansey")

    def test_status_brn(self):
        self.assertEqual(self.ops.status[ALL_STATUSES.index(Status.BRN)], 1.0)

    def test_only_one_status_set(self):
        self.assertEqual(self.ops.status.sum(), 1.0)

    def test_def_boost(self):
        self.assertEqual(self.ops.boosts[PokemonState.BOOST_KEYS.index("def")], 3.0)

    def test_def_boost_encoded(self):
        def_idx = PokemonState.BOOST_KEYS.index("def")
        self.assertAlmostEqual(float(self.ops.normalize_boosts()[def_idx]), 0.5, places=5)

    def test_other_boosts_zero(self):
        def_idx = PokemonState.BOOST_KEYS.index("def")
        np.testing.assert_array_equal(
            np.delete(self.ops.boosts, def_idx),
            np.zeros(len(PokemonState.BOOST_KEYS) - 1)
        )

    def test_preparing_false(self):
        self.assertEqual(self.ops.preparing, 0.0)

    def test_must_recharge_false(self):
        self.assertEqual(self.ops.must_recharge, 0.0)

    def test_estimated_stats_match_formula(self):
        expected = _expected_estimated_stats(_BASE_STATS["chansey"])
        np.testing.assert_array_almost_equal(self.ops.stats, expected)

    def test_estimated_hp_large_base(self):
        base_hp = _BASE_STATS["chansey"]["hp"]
        expected_hp = (base_hp + _DV) * 2 + _EV_TERM + _LEVEL + 10
        self.assertAlmostEqual(float(self.ops.stats[GEN1_STAT_KEYS.index("hp")]), expected_hp)


# ---------------------------------------------------------------------------
# protect_counter static method
# ---------------------------------------------------------------------------

class TestProtectCounterStatic(unittest.TestCase):
    """Unit tests for the protect_counter() static method."""

    def test_counter_zero(self):
        p = _make_bare_mock(protect_counter=0)
        self.assertEqual(OpponentPokemonState.protect_counter(p), 0.0)

    def test_counter_one(self):
        p = _make_bare_mock(protect_counter=1)
        self.assertEqual(OpponentPokemonState.protect_counter(p), 1.0)

    def test_counter_three(self):
        p = _make_bare_mock(protect_counter=3)
        self.assertEqual(OpponentPokemonState.protect_counter(p), 3.0)

    def test_none_pokemon_returns_zero(self):
        self.assertEqual(OpponentPokemonState.protect_counter(None), 0.0)

    def test_missing_attribute_defaults_to_zero(self):
        p = MagicMock(spec=[])   # no attributes at all
        self.assertEqual(OpponentPokemonState.protect_counter(p), 0.0)

    def test_counter_stored_on_ops(self):
        p = _make_bare_mock(protect_counter=2)
        ops = OpponentPokemonState(p)
        self.assertEqual(ops.protect, 2.0)


# ---------------------------------------------------------------------------
# normalize_protect  —  geometric decay 0.3 ** protect
# ---------------------------------------------------------------------------

class TestNormalizeProtect(unittest.TestCase):
    """Unit tests for normalize_protect() = 0.3 ** self.protect."""

    def _make_ops(self, protect_counter: int) -> OpponentPokemonState:
        return OpponentPokemonState(_make_bare_mock(protect_counter=protect_counter))

    def test_counter_zero_gives_one(self):
        """0.3 ** 0 == 1.0 — no prior protects."""
        ops = self._make_ops(0)
        np.testing.assert_array_almost_equal(ops.normalize_protect(), [1.0])

    def test_counter_one_gives_point_three(self):
        """0.3 ** 1 == 0.3."""
        ops = self._make_ops(1)
        np.testing.assert_array_almost_equal(ops.normalize_protect(), [0.3], decimal=6)

    def test_counter_two_gives_point_zero_nine(self):
        """0.3 ** 2 == 0.09."""
        ops = self._make_ops(2)
        np.testing.assert_array_almost_equal(ops.normalize_protect(), [0.09], decimal=6)

    def test_counter_three(self):
        """0.3 ** 3 == 0.027."""
        ops = self._make_ops(3)
        np.testing.assert_array_almost_equal(ops.normalize_protect(), [0.027], decimal=6)

    def test_strictly_decreasing(self):
        """Each additional consecutive protect must lower the probability."""
        values = [float(self._make_ops(n).normalize_protect()[0]) for n in range(5)]
        for i in range(len(values) - 1):
            self.assertGreater(values[i], values[i + 1], msg=f"values[{i}] not > values[{i+1}]")

    def test_always_positive(self):
        """0.3 ** n is always > 0 for finite n."""
        for n in range(10):
            val = float(self._make_ops(n).normalize_protect()[0])
            self.assertGreater(val, 0.0)

    def test_always_at_most_one(self):
        """Value must never exceed 1.0."""
        for n in range(10):
            val = float(self._make_ops(n).normalize_protect()[0])
            self.assertLessEqual(val, 1.0)

    def test_returns_float32(self):
        self.assertEqual(self._make_ops(0).normalize_protect().dtype, np.float32)

    def test_returns_shape_one(self):
        self.assertEqual(self._make_ops(0).normalize_protect().shape, (1,))

    def test_in_to_array_last_position(self):
        """normalize_protect() value must appear as the last element of to_array()."""
        for counter in [0, 1, 2]:
            ops = self._make_ops(counter)
            arr = ops.to_array()
            expected = float(ops.normalize_protect()[0])
            self.assertAlmostEqual(float(arr[-1]), expected,
                                   msg=f"protect_counter={counter}")

    def test_fixture_with_nonzero_counter(self):
        """Fixture loaded with protect_counter=2 must use 0.3**2 = 0.09."""
        pokemon, _ = make_opponent_mock_from_fixture(
            "gen1_starmie_opponent.json", protect_counter=2
        )
        ops = OpponentPokemonState(pokemon)
        np.testing.assert_array_almost_equal(ops.normalize_protect(), [0.09], decimal=6)


# ---------------------------------------------------------------------------
# Static helpers  —  is_preparing / is_recharge
# ---------------------------------------------------------------------------

class TestOpponentStaticMethods(unittest.TestCase):

    def _mock(self, *, preparing=False, must_recharge=False):
        p = MagicMock()
        p.preparing     = preparing
        p.must_recharge = must_recharge
        return p

    def test_is_preparing_true(self):
        self.assertEqual(OpponentPokemonState.is_preparing(self._mock(preparing=True)), 1.0)

    def test_is_preparing_false(self):
        self.assertEqual(OpponentPokemonState.is_preparing(self._mock(preparing=False)), 0.0)

    def test_is_preparing_none_pokemon(self):
        self.assertEqual(OpponentPokemonState.is_preparing(None), False)

    def test_is_recharge_true(self):
        self.assertEqual(OpponentPokemonState.is_recharge(self._mock(must_recharge=True)), 1.0)

    def test_is_recharge_false(self):
        self.assertEqual(OpponentPokemonState.is_recharge(self._mock(must_recharge=False)), 0.0)

    def test_is_recharge_none_pokemon(self):
        self.assertEqual(OpponentPokemonState.is_recharge(None), False)

    def test_recharge_and_preparing_are_independent(self):
        p_prep     = self._mock(preparing=True,  must_recharge=False)
        p_recharge = self._mock(preparing=False, must_recharge=True)
        self.assertEqual(OpponentPokemonState.is_recharge(p_prep),     0.0)
        self.assertEqual(OpponentPokemonState.is_preparing(p_recharge), 0.0)


# ---------------------------------------------------------------------------
# must_recharge field
# ---------------------------------------------------------------------------

class TestMustRecharge(unittest.TestCase):

    def _make_ops(self, *, must_recharge: bool) -> OpponentPokemonState:
        return OpponentPokemonState(_make_bare_mock("tauros", must_recharge=must_recharge))

    def test_must_recharge_true(self):
        self.assertEqual(self._make_ops(must_recharge=True).must_recharge, 1.0)

    def test_must_recharge_false(self):
        self.assertEqual(self._make_ops(must_recharge=False).must_recharge, 0.0)

    def test_must_recharge_not_in_effects(self):
        """MUST_RECHARGE was removed from TRACKED_EFFECTS — effects vector is len 2."""
        self.assertEqual(len(self._make_ops(must_recharge=True).effects), 2)

    def test_must_recharge_independent_of_preparing(self):
        self.assertEqual(self._make_ops(must_recharge=True).preparing, 0.0)


# ---------------------------------------------------------------------------
# estimate_stats formula
# ---------------------------------------------------------------------------

class TestEstimateStats(unittest.TestCase):

    def _make_ops(self, base_stats: dict) -> OpponentPokemonState:
        p = _make_bare_mock()
        p.base_stats = base_stats
        return OpponentPokemonState(p)

    def test_hp_formula(self):
        ops = self._make_ops({"hp": 100, "atk": 50, "def": 50, "spc": 50, "spe": 50})
        self.assertAlmostEqual(float(ops.stats[0]), (100 + 15) * 2 + 63 + 100 + 10)

    def test_non_hp_formula(self):
        ops = self._make_ops({"hp": 50, "atk": 80, "def": 80, "spc": 80, "spe": 80})
        self.assertAlmostEqual(float(ops.stats[GEN1_STAT_KEYS.index("atk")]),
                               (80 + 15) * 2 + 63 + 5)

    def test_hp_greater_than_non_hp_for_equal_bases(self):
        ops = self._make_ops({"hp": 80, "atk": 80, "def": 80, "spc": 80, "spe": 80})
        hp  = float(ops.stats[GEN1_STAT_KEYS.index("hp")])
        atk = float(ops.stats[GEN1_STAT_KEYS.index("atk")])
        self.assertGreater(hp, atk)

    def test_higher_base_gives_higher_stat(self):
        low  = self._make_ops({"hp": 45, "atk": 49, "def": 49, "spc": 45, "spe": 45})
        high = self._make_ops({"hp": 75, "atk": 100,"def": 95, "spc": 70, "spe": 110})
        self.assertTrue(np.all(high.stats > low.stats))

    def test_ev_term_is_63(self):
        self.assertEqual(_EV_TERM, 63)

    def test_output_dtype(self):
        self.assertEqual(
            self._make_ops({"hp": 60, "atk": 75, "def": 85, "spc": 100, "spe": 115}).stats.dtype,
            np.float32,
        )

    def test_output_shape(self):
        self.assertEqual(
            self._make_ops({"hp": 60, "atk": 75, "def": 85, "spc": 100, "spe": 115}).stats.shape,
            (len(GEN1_STAT_KEYS),),
        )

    def test_all_stats_positive(self):
        ops = self._make_ops({"hp": 1, "atk": 1, "def": 1, "spc": 1, "spe": 1})
        self.assertTrue(np.all(ops.stats > 0))


if __name__ == "__main__":
    unittest.main()