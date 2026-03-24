import unittest
from typing import TYPE_CHECKING
from unittest.mock import MagicMock

import numpy as np

from poke_env.battle.move_category import MoveCategory
from poke_env.battle.pokemon_type import PokemonType
from poke_env.battle.status import Status

from env.states.move_state import MoveState, MOVE_CATEGORIES
from env.states.state_utils import ALL_STATUSES, GEN1_BOOST_KEYS, BOOST_NORM


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_EMPTY_BOOSTS = {k: 0 for k in GEN1_BOOST_KEYS}

# to_array() slice positions — mirrors move_state.py layout exactly
_BP_IDX        = 0
_ACC_IDX       = 1
_PRI_IDX       = 2
_HEAL_IDX      = 3
_CRIT_IDX      = 4
_CAT_S         = slice(5, 5 + len(MOVE_CATEGORIES))
_PROTECT_IDX   = 5 + len(MOVE_CATEGORIES)
_BREAKS_IDX    = _PROTECT_IDX + 1
_STAB_IDX      = _BREAKS_IDX + 1
_STATUS_S      = slice(_STAB_IDX + 1, _STAB_IDX + 1 + len(ALL_STATUSES))
_OPP_BOOST_S   = slice(_STATUS_S.stop, _STATUS_S.stop + len(GEN1_BOOST_KEYS))
_SELF_BOOST_S  = slice(_OPP_BOOST_S.stop, _OPP_BOOST_S.stop + len(GEN1_BOOST_KEYS))
_TYPE_MULT_IDX = _SELF_BOOST_S.stop
_RECOIL_IDX    = _TYPE_MULT_IDX + 1
_DRAIN_IDX     = _RECOIL_IDX + 1
_MIN_HITS_IDX  = _DRAIN_IDX + 1
_MAX_HITS_IDX  = _MIN_HITS_IDX + 1


def make_move_mock(
    base_power:      float         = 80.0,
    accuracy                       = 1.0,
    max_pp:          float         = 16.0,
    priority:        int           = 0,
    heal:            float         = 0.0,
    crit_ratio:      float         = 1.0,
    category:        MoveCategory  = MoveCategory.PHYSICAL,
    move_type:       PokemonType   = PokemonType.NORMAL,
    is_protect_move: bool          = False,
    breaks_protect:  bool          = False,
    status                         = None,
    boosts:          dict          = None,
    self_boost:      dict          = None,
    recoil:          float         = 0.0,
    drain:           float         = 0.0,
    n_hit                          = 1,
    move_id:         str           = "tackle",
) -> MagicMock:
    move = MagicMock()
    move.id              = move_id
    move.base_power      = base_power
    move.accuracy        = accuracy
    move.max_pp          = max_pp
    move.priority        = priority
    move.heal            = heal
    move.crit_ratio      = crit_ratio
    move.category        = category
    move.is_protect_move = is_protect_move
    move.breaks_protect  = breaks_protect
    move.status          = status
    move.recoil          = recoil
    move.drain           = drain
    move.n_hit           = n_hit
    move.boosts     = boosts     if boosts     is not None else dict(_EMPTY_BOOSTS)
    move.self_boost = self_boost if self_boost is not None else dict(_EMPTY_BOOSTS)
    move.type = MagicMock()
    move.type.__eq__ = lambda self, other: other == move_type
    move.type.__hash__ = lambda self: hash(move_type)
    move.type.damage_multiplier = MagicMock(return_value=1.0)
    return move


def make_move_state(
    opp_types=(PokemonType.NORMAL,),
    my_types=(PokemonType.NORMAL,),
    gen: int = 1,
    **kwargs,
) -> MoveState:
    return MoveState(make_move_mock(**kwargs), opp_types, my_types, gen)


def _expected_array_len() -> int:
    return 13 + len(MOVE_CATEGORIES) + len(ALL_STATUSES) + 2 * len(GEN1_BOOST_KEYS)


# ---------------------------------------------------------------------------
# Shared base mixin — structural guarantees every MoveState must satisfy
# ---------------------------------------------------------------------------

if TYPE_CHECKING:
    _Base = unittest.TestCase
else:
    _Base = object


class MoveStateBaseTest(_Base):
    ms: MoveState

    def test_to_array_dtype(self):
        self.assertEqual(self.ms.to_array().dtype, np.float32)

    def test_to_array_length_matches_array_len(self):
        self.assertEqual(len(self.ms.to_array()), self.ms.array_len())

    def test_array_len_correct_formula(self):
        self.assertEqual(self.ms.array_len(), _expected_array_len())

    def test_no_nan(self):
        self.assertFalse(np.any(np.isnan(self.ms.to_array())))

    def test_no_inf(self):
        self.assertFalse(np.any(np.isinf(self.ms.to_array())))


# ---------------------------------------------------------------------------
# None move — MoveState(None) must behave identically to MoveState.zero()
# ---------------------------------------------------------------------------

class TestMoveStateNone(unittest.TestCase):

    def setUp(self):
        self.ms = MoveState(None, (PokemonType.NORMAL,), (PokemonType.NORMAL,), 1)

    def test_to_array_all_zeros(self):
        np.testing.assert_array_equal(
            self.ms.to_array(), np.zeros(self.ms.array_len(), dtype=np.float32)
        )

    def test_array_len_correct(self):
        self.assertEqual(self.ms.array_len(), MoveState.zero().array_len())

    def test_dtype_float32(self):
        self.assertEqual(self.ms.to_array().dtype, np.float32)

    def test_no_nan(self):
        self.assertFalse(np.any(np.isnan(self.ms.to_array())))

    def test_no_inf(self):
        self.assertFalse(np.any(np.isinf(self.ms.to_array())))

    def test_id_is_none(self):
        self.assertIsNone(self.ms.id)

    def test_status_is_none(self):
        self.assertIsNone(self.ms.status)

    def test_category_is_none(self):
        self.assertIsNone(self.ms.category)

    def test_min_hits_is_zero(self):
        self.assertEqual(self.ms.min_hits, 0)

    def test_max_hits_is_zero(self):
        self.assertEqual(self.ms.max_hits, 0)


# ---------------------------------------------------------------------------
# Zero state
# ---------------------------------------------------------------------------


class TestMoveStateZero(unittest.TestCase):
    """MoveState.zero() must return an all-zeros array of the correct length."""

    def setUp(self):
        self.ms = MoveState.zero()

    def test_array_all_zeros(self):
        np.testing.assert_array_equal(
            self.ms.to_array(), np.zeros(_expected_array_len(), dtype=np.float32)
        )

    def test_array_len_correct(self):
        self.assertEqual(self.ms.array_len(), _expected_array_len())

    def test_dtype_float32(self):
        self.assertEqual(self.ms.to_array().dtype, np.float32)


# ---------------------------------------------------------------------------
# Class constant
# ---------------------------------------------------------------------------

class TestMoveStateClassConstants(unittest.TestCase):
    def test_boost_keys_are_gen1(self):
        self.assertIs(MoveState.BOOST_KEYS, GEN1_BOOST_KEYS)


# ---------------------------------------------------------------------------
# Field population — every field read from the move object
# ---------------------------------------------------------------------------

class TestMoveStateFields(MoveStateBaseTest, unittest.TestCase):
    """All scalar fields are read correctly from the mock."""

    def setUp(self):
        self.ms = make_move_state(
            base_power=90.0,
            accuracy=0.9,
            max_pp=10.0,
            priority=1,
            heal=0.5,
            crit_ratio=2.0,
            category=MoveCategory.SPECIAL,
            recoil=0.25,
            drain=0.5,
            status=Status.BRN,
            move_id="flamethrower",
        )

    def test_base_power(self):         self.assertAlmostEqual(self.ms.base_power, 90.0)
    def test_accuracy(self):           self.assertAlmostEqual(self.ms.accuracy, 0.9)
    def test_max_pp(self):             self.assertAlmostEqual(self.ms.max_pp, 10.0)
    def test_priority(self):           self.assertEqual(self.ms.priority, 1)
    def test_heal(self):               self.assertAlmostEqual(self.ms.heal, 0.5)
    def test_crit_ratio(self):         self.assertAlmostEqual(self.ms.crit_ratio, 2.0)
    def test_category(self):           self.assertEqual(self.ms.category, MoveCategory.SPECIAL)
    def test_recoil(self):             self.assertAlmostEqual(self.ms.recoil, 0.25)
    def test_drain(self):              self.assertAlmostEqual(self.ms.drain, 0.5)
    def test_status(self):             self.assertEqual(self.ms.status, Status.BRN)
    def test_id(self):                 self.assertEqual(self.ms.id, "flamethrower")


# ---------------------------------------------------------------------------
# accuracy edge cases
# ---------------------------------------------------------------------------

class TestMoveStateAccuracyTrue(MoveStateBaseTest, unittest.TestCase):
    """accuracy=True (never-miss) → stored as 1.0."""
    def setUp(self):
        self.ms = make_move_state(accuracy=True)

    def test_accuracy_true_maps_to_one(self):
        self.assertAlmostEqual(self.ms.accuracy, 1.0)


class TestMoveStateAccuracyNone(MoveStateBaseTest, unittest.TestCase):
    """accuracy=None → falls back to 0.0."""
    def setUp(self):
        self.ms = make_move_state(accuracy=None)

    def test_accuracy_none_maps_to_zero(self):
        self.assertAlmostEqual(self.ms.accuracy, 0.0)


# ---------------------------------------------------------------------------
# STAB
# ---------------------------------------------------------------------------

class TestMoveStateStab(MoveStateBaseTest, unittest.TestCase):
    def setUp(self):
        self.ms = make_move_state(
            move_type=PokemonType.FIRE,
            my_types=(PokemonType.FIRE, PokemonType.FLYING),
        )

    def test_is_stab_true(self):
        self.assertEqual(self.ms.is_stab, 1.0)


class TestMoveStateNoStab(MoveStateBaseTest, unittest.TestCase):
    def setUp(self):
        self.ms = make_move_state(
            move_type=PokemonType.FIRE,
            my_types=(PokemonType.WATER,),
        )

    def test_is_stab_false(self):
        self.assertEqual(self.ms.is_stab, 0.0)


# ---------------------------------------------------------------------------
# Protect flags
# ---------------------------------------------------------------------------

class TestMoveStateProtectFlags(MoveStateBaseTest, unittest.TestCase):
    def setUp(self):
        self.ms = make_move_state(is_protect_move=True, breaks_protect=False)

    def test_is_protect_move_one(self):
        self.assertEqual(self.ms.is_protect_move, 1.0)

    def test_breaks_protect_zero(self):
        self.assertEqual(self.ms.breaks_protect, 0.0)

    def test_breaks_protect_one(self):
        ms = make_move_state(is_protect_move=False, breaks_protect=True)
        self.assertEqual(ms.breaks_protect, 1.0)


# ---------------------------------------------------------------------------
# Multi-hit
# ---------------------------------------------------------------------------

class TestMoveStateMultiHitTuple(MoveStateBaseTest, unittest.TestCase):
    def setUp(self):
        self.ms = make_move_state(n_hit=(2, 5))

    def test_min_hits(self):   self.assertEqual(self.ms.min_hits, 2)
    def test_max_hits(self):   self.assertEqual(self.ms.max_hits, 5)


class TestMoveStateMultiHitInt(MoveStateBaseTest, unittest.TestCase):
    def setUp(self):
        self.ms = make_move_state(n_hit=2)

    def test_min_max_equal(self):
        self.assertEqual(self.ms.min_hits, self.ms.max_hits)
        self.assertEqual(self.ms.min_hits, 2)


class TestMoveStateNHitFallback(MoveStateBaseTest, unittest.TestCase):
    """n_hit=None → else branch → both default to 1."""
    def setUp(self):
        self.ms = make_move_state(n_hit=None)

    def test_min_hits_one(self):   self.assertEqual(self.ms.min_hits, 1)
    def test_max_hits_one(self):   self.assertEqual(self.ms.max_hits, 1)


# ---------------------------------------------------------------------------
# to_array() — slice positions
# Every slice is tested at its exact index/range
# ---------------------------------------------------------------------------

class TestMoveStateToArraySlices(MoveStateBaseTest, unittest.TestCase):
    """Verify every slice sits at the right position in to_array()."""

    def setUp(self):
        boosts     = {**_EMPTY_BOOSTS, "atk": -1}
        self_boost = {**_EMPTY_BOOSTS, "spa": 2}
        self.ms = make_move_state(
            base_power=100.0,
            accuracy=0.8,
            priority=1,
            heal=0.5,
            crit_ratio=2.0,
            category=MoveCategory.SPECIAL,
            is_protect_move=False,
            breaks_protect=True,
            move_type=PokemonType.WATER,
            my_types=(PokemonType.WATER,),   # STAB
            status=Status.PSN,
            boosts=boosts,
            self_boost=self_boost,
            recoil=0.25,
            drain=0.33,
            n_hit=(2, 3),
        )
        self.arr = self.ms.to_array()

    def test_base_power_position(self):
        self.assertAlmostEqual(float(self.arr[_BP_IDX]), 100.0 / 200.0, places=5)

    def test_accuracy_position(self):
        self.assertAlmostEqual(float(self.arr[_ACC_IDX]), 0.8, places=5)

    def test_priority_position(self):
        self.assertAlmostEqual(float(self.arr[_PRI_IDX]), 1.0 / 7.0, places=5)

    def test_heal_position(self):
        self.assertAlmostEqual(float(self.arr[_HEAL_IDX]), 0.5, places=5)

    def test_crit_ratio_position(self):
        self.assertAlmostEqual(float(self.arr[_CRIT_IDX]), 2.0 / 6.0, places=5)

    def test_category_one_hot_sum(self):
        self.assertEqual(float(self.arr[_CAT_S].sum()), 1.0)

    def test_category_correct_bit(self):
        idx = list(MOVE_CATEGORIES).index(MoveCategory.SPECIAL)
        self.assertEqual(float(self.arr[_CAT_S][idx]), 1.0)

    def test_protect_position(self):
        self.assertEqual(float(self.arr[_PROTECT_IDX]), 0.0)

    def test_breaks_protect_position(self):
        self.assertEqual(float(self.arr[_BREAKS_IDX]), 1.0)

    def test_stab_position(self):
        self.assertEqual(float(self.arr[_STAB_IDX]), 1.0)

    def test_status_one_hot_sum(self):
        self.assertEqual(float(self.arr[_STATUS_S].sum()), 1.0)

    def test_status_correct_bit(self):
        idx = list(ALL_STATUSES).index(Status.PSN)
        self.assertEqual(float(self.arr[_STATUS_S][idx]), 1.0)

    def test_opp_boost_atk_position(self):
        atk_idx = GEN1_BOOST_KEYS.index("atk")
        self.assertAlmostEqual(float(self.arr[_OPP_BOOST_S][atk_idx]), -1.0 / BOOST_NORM, places=5)

    def test_self_boost_spa_position(self):
        spa_idx = GEN1_BOOST_KEYS.index("spa")
        self.assertAlmostEqual(float(self.arr[_SELF_BOOST_S][spa_idx]), 2.0 / BOOST_NORM, places=5)

    def test_type_multiplier_position(self):
        # damage_multiplier mock returns 1.0 → log2(1)/2 = 0.0
        self.assertAlmostEqual(float(self.arr[_TYPE_MULT_IDX]), 0.0, places=5)

    def test_recoil_position(self):
        self.assertAlmostEqual(float(self.arr[_RECOIL_IDX]), 0.25, places=5)

    def test_drain_position(self):
        self.assertAlmostEqual(float(self.arr[_DRAIN_IDX]), 0.33, places=5)

    def test_min_hits_position(self):
        self.assertAlmostEqual(float(self.arr[_MIN_HITS_IDX]), 2.0 / 5.0, places=5)

    def test_max_hits_position(self):
        self.assertAlmostEqual(float(self.arr[_MAX_HITS_IDX]), 3.0 / 5.0, places=5)


# ---------------------------------------------------------------------------
# to_array() — zero slices when fields are default
# ---------------------------------------------------------------------------

class TestMoveStateToArrayDefaults(MoveStateBaseTest, unittest.TestCase):
    """When optional fields are zero/None the corresponding slices are zero."""

    def setUp(self):
        self.ms = make_move_state(
            base_power=80.0,
            accuracy=1.0,
            status=None,
            boosts=None,
            self_boost=None,
            recoil=0.0,
            drain=0.0,
        )
        self.arr = self.ms.to_array()

    def test_status_all_zero(self):
        np.testing.assert_array_equal(self.arr[_STATUS_S], np.zeros(len(ALL_STATUSES)))

    def test_opp_boosts_all_zero(self):
        np.testing.assert_array_equal(self.arr[_OPP_BOOST_S], np.zeros(len(GEN1_BOOST_KEYS)))

    def test_self_boosts_all_zero(self):
        np.testing.assert_array_equal(self.arr[_SELF_BOOST_S], np.zeros(len(GEN1_BOOST_KEYS)))

    def test_recoil_zero(self):
        self.assertEqual(float(self.arr[_RECOIL_IDX]), 0.0)

    def test_drain_zero(self):
        self.assertEqual(float(self.arr[_DRAIN_IDX]), 0.0)


# ---------------------------------------------------------------------------
# type_multiplier field
# ---------------------------------------------------------------------------

class TestMoveStateTypeMultiplier(MoveStateBaseTest, unittest.TestCase):
    """type_multiplier is stored from move.type.damage_multiplier()."""

    def setUp(self):
        self.ms = make_move_state()
        # default mock returns 1.0

    def test_type_multiplier_stored(self):
        self.assertAlmostEqual(self.ms.type_multiplier, 1.0)

    def test_type_multiplier_two(self):
        # Override damage_multiplier to return 2.0 (super-effective)
        mock = make_move_mock()
        mock.type.damage_multiplier = MagicMock(return_value=2.0)
        ms = MoveState(mock, (PokemonType.NORMAL,), (PokemonType.NORMAL,), 1)
        self.assertAlmostEqual(ms.type_multiplier, 2.0)

    def test_dual_type_opponent(self):
        # Two opp types are passed to damage_multiplier correctly
        mock = make_move_mock()
        mock.type.damage_multiplier = MagicMock(return_value=4.0)
        ms = MoveState(mock, (PokemonType.WATER, PokemonType.FLYING), (PokemonType.NORMAL,), 1)
        self.assertAlmostEqual(ms.type_multiplier, 4.0)


# ---------------------------------------------------------------------------
# opp_boosts / self_boost with None value in dict (pull_attribute fix)
# ---------------------------------------------------------------------------

class TestMoveStateNoneBoostValues(MoveStateBaseTest, unittest.TestCase):
    """None values in boost dicts must be treated as 0, not raise TypeError."""

    def setUp(self):
        self.ms = make_move_state(
            boosts={**_EMPTY_BOOSTS, "atk": None},
            self_boost={**_EMPTY_BOOSTS, "spa": None},
        )

    def test_opp_boost_none_treated_as_zero(self):
        self.assertEqual(self.ms.opp_boosts.get("atk"), None)  # stored as-is in dict
        # but the encoded array slot must be 0.0, not crash
        arr = self.ms.to_array()
        atk_idx = GEN1_BOOST_KEYS.index("atk")
        self.assertAlmostEqual(float(arr[_OPP_BOOST_S][atk_idx]), 0.0, places=5)

    def test_self_boost_none_treated_as_zero(self):
        arr = self.ms.to_array()
        spa_idx = GEN1_BOOST_KEYS.index("spa")
        self.assertAlmostEqual(float(arr[_SELF_BOOST_S][spa_idx]), 0.0, places=5)


# ---------------------------------------------------------------------------
# _encode_type_multiplier
# ---------------------------------------------------------------------------

class TestEncodeTypeMultiplier(unittest.TestCase):

    def _enc(self, mult):
        return float(MoveState._encode_type_multiplier(mult)[0])

    def test_immune(self):              self.assertAlmostEqual(self._enc(0.0),  -1.0)
    def test_quarter_effective(self):   self.assertAlmostEqual(self._enc(0.25), -1.0)
    def test_not_very_effective(self):  self.assertAlmostEqual(self._enc(0.5),  -0.5)
    def test_neutral(self):             self.assertAlmostEqual(self._enc(1.0),   0.0)
    def test_super_effective(self):     self.assertAlmostEqual(self._enc(2.0),   0.5)
    def test_double_super(self):        self.assertAlmostEqual(self._enc(4.0),   1.0)
    def test_returns_float32(self):     self.assertEqual(MoveState._encode_type_multiplier(1.0).dtype, np.float32)
    def test_returns_shape_one(self):   self.assertEqual(MoveState._encode_type_multiplier(1.0).shape, (1,))


# ---------------------------------------------------------------------------
# describe()
# ---------------------------------------------------------------------------

class TestMoveStateDescribe(unittest.TestCase):

    def setUp(self):
        self.ms = make_move_state(
            base_power=80.0,
            category=MoveCategory.PHYSICAL,
            move_id="body-slam",
            status=Status.PAR,
            is_protect_move=False,
            breaks_protect=False,
            boosts={**_EMPTY_BOOSTS, "atk": -1},
            self_boost={**_EMPTY_BOOSTS, "def": 1},
        )
        self.desc = self.ms.describe()

    def test_contains_move_id(self):        self.assertIn("body-slam", self.desc)
    def test_contains_base_power(self):     self.assertIn("80", self.desc)
    def test_contains_category(self):       self.assertIn("PHYSICAL", self.desc)
    def test_contains_status(self):         self.assertIn("PAR", self.desc)
    def test_contains_protect_line(self):   self.assertIn("Protect", self.desc)
    def test_contains_opp_boosts_line(self):self.assertIn("Opp boosts", self.desc)
    def test_contains_self_boosts_line(self):self.assertIn("Self boosts", self.desc)
    def test_repr_contains_base_power(self):self.assertIn("80", repr(self.ms))

    def test_describe_no_boosts_shows_none(self):
        ms = make_move_state(boosts=None, self_boost=None)
        desc = ms.describe()
        lines = {l.split(":")[0].strip(): l for l in desc.splitlines()}
        self.assertIn("none", lines.get("Opp boosts", ""))
        self.assertIn("none", lines.get("Self boosts", ""))

    def test_describe_no_status_shows_none(self):
        ms = make_move_state(status=None)
        lines = {l.split(":")[0].strip(): l for l in ms.describe().splitlines()}
        self.assertIn("none", lines.get("Status inflict", ""))

    def test_describe_no_category_does_not_raise(self):
        # category=None → guarded by `if self.category` in describe()
        mock = make_move_mock(category=None)  # type: ignore[arg-type]
        ms = MoveState(mock, (PokemonType.NORMAL,), (PokemonType.NORMAL,), 1)
        try:
            ms.describe()
        except Exception as e:
            self.fail(f"describe() with category=None raised: {e}")


if __name__ == "__main__":
    unittest.main()
