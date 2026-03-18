import unittest
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
    """Build a minimal mock Move for MoveState construction."""
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

    # boosts dicts must cover ALL boost keys so normalize_vector output
    # length matches array_len() (which uses len(BOOST_KEYS)).
    move.boosts     = boosts     if boosts     is not None else dict(_EMPTY_BOOSTS)
    move.self_boost = self_boost if self_boost is not None else dict(_EMPTY_BOOSTS)

    # type.damage_multiplier is called in __init__
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
    """Convenience wrapper around MoveState(make_move_mock(...))."""
    return MoveState(make_move_mock(**kwargs), opp_types, my_types, gen)


def _expected_array_len() -> int:
    return 13 + len(MOVE_CATEGORIES) + len(ALL_STATUSES) + 2 * len(GEN1_BOOST_KEYS)


# ---------------------------------------------------------------------------
# Shared base mixin — every concrete case must pass these
# ---------------------------------------------------------------------------

class MoveStateBaseTest:
    """
    Mixin — does NOT inherit TestCase so unittest won't run it directly.
    Subclasses must set self.ms in setUp().
    """
    ms: MoveState

    # ── to_array structure ───────────────────────────────────────────────

    def test_to_array_dtype(self):
        self.assertEqual(self.ms.to_array().dtype, np.float32)

    def test_to_array_length(self):
        self.assertEqual(len(self.ms.to_array()), self.ms.array_len())

    def test_array_len_formula(self):
        self.assertEqual(self.ms.array_len(), _expected_array_len())

    # ── scalar ranges ────────────────────────────────────────────────────

    def test_base_power_in_range(self):
        arr = self.ms.to_array()
        self.assertGreaterEqual(float(arr[0]), 0.0)
        self.assertLessEqual(float(arr[0]), 1.0)

    def test_accuracy_in_range(self):
        arr = self.ms.to_array()
        self.assertGreaterEqual(float(arr[1]), 0.0)
        self.assertLessEqual(float(arr[1]), 1.0)


# ---------------------------------------------------------------------------
# Zero / null move
# ---------------------------------------------------------------------------

class TestMoveStateZero(MoveStateBaseTest, unittest.TestCase):
    """MoveState.zero() must return an all-zeros vector without error."""

    def setUp(self):
        self.ms = MoveState.zero()

    def test_to_array_all_zeros(self):
        np.testing.assert_array_equal(self.ms.to_array(), np.zeros(_expected_array_len()))

    def test_to_array_dtype(self):
        self.assertEqual(self.ms.to_array().dtype, np.float32)

    def test_array_len_formula(self):
        self.assertEqual(self.ms.array_len(), _expected_array_len())

    # Override base shape tests that call to_array internally
    def test_to_array_length(self):
        self.assertEqual(len(self.ms.to_array()), _expected_array_len())

    def test_base_power_in_range(self):
        self.assertEqual(float(self.ms.to_array()[0]), 0.0)

    def test_accuracy_in_range(self):
        self.assertEqual(float(self.ms.to_array()[1]), 0.0)


# ---------------------------------------------------------------------------
# Standard physical move (Tackle-like)
# ---------------------------------------------------------------------------

class TestMoveStatePhysical(MoveStateBaseTest, unittest.TestCase):
    """A plain Normal-type physical move with no frills."""

    def setUp(self):
        self.ms = make_move_state(
            base_power=40.0,
            accuracy=1.0,
            category=MoveCategory.PHYSICAL,
            move_type=PokemonType.NORMAL,
            my_types=(PokemonType.NORMAL,),
        )

    def test_base_power_normalized(self):
        self.assertAlmostEqual(float(self.ms.to_array()[0]), 40.0 / 200.0)

    def test_accuracy_stored(self):
        self.assertAlmostEqual(self.ms.accuracy, 1.0)

    def test_is_stab_true(self):
        self.assertEqual(self.ms.is_stab, 1.0)

    def test_category_is_physical(self):
        self.assertEqual(self.ms.category, MoveCategory.PHYSICAL)

    def test_no_status(self):
        self.assertIsNone(self.ms.status)

    def test_recoil_is_zero(self):
        self.assertEqual(self.ms.recoil, 0.0)

    def test_drain_is_zero(self):
        self.assertEqual(self.ms.drain, 0.0)


# ---------------------------------------------------------------------------
# Special move
# ---------------------------------------------------------------------------

class TestMoveStateSpecial(MoveStateBaseTest, unittest.TestCase):
    """A special-category move; verifies category one-hot encoding."""

    def setUp(self):
        self.ms = make_move_state(
            base_power=90.0,
            category=MoveCategory.SPECIAL,
            move_type=PokemonType.FIRE,
            my_types=(PokemonType.WATER,),     # no STAB
        )

    def test_is_stab_false(self):
        self.assertEqual(self.ms.is_stab, 0.0)

    def test_category_is_special(self):
        self.assertEqual(self.ms.category, MoveCategory.SPECIAL)

    def test_category_one_hot_sum(self):
        arr = self.ms.to_array()
        category_start = 5  # after bp, acc, pri, heal, crit_ratio
        category_slice = arr[category_start: category_start + len(MOVE_CATEGORIES)]
        self.assertEqual(category_slice.sum(), 1.0)

    def test_category_correct_index_hot(self):
        arr = self.ms.to_array()
        category_start = 5
        expected_idx = list(MOVE_CATEGORIES).index(MoveCategory.SPECIAL)
        self.assertEqual(arr[category_start + expected_idx], 1.0)


# ---------------------------------------------------------------------------
# Status move
# ---------------------------------------------------------------------------

class TestMoveStateStatus(MoveStateBaseTest, unittest.TestCase):
    """A status-category move that inflicts PSN."""

    def setUp(self):
        self.ms = make_move_state(
            base_power=0.0,
            accuracy=0.85,
            category=MoveCategory.STATUS,
            status=Status.PSN,
        )

    def test_base_power_zero(self):
        self.assertAlmostEqual(float(self.ms.to_array()[0]), 0.0)

    def test_accuracy_stored(self):
        self.assertAlmostEqual(self.ms.accuracy, 0.85)

    def test_status_one_hot_sum(self):
        enc = self.ms.to_array()
        status_start = 5 + len(MOVE_CATEGORIES) + 3  # bp,acc,pri,heal,crit + cat + protect,breaks,stab
        status_slice = enc[status_start: status_start + len(ALL_STATUSES)]
        self.assertEqual(status_slice.sum(), 1.0)

    def test_status_correct_index_hot(self):
        enc = self.ms.to_array()
        status_start = 5 + len(MOVE_CATEGORIES) + 3
        expected_idx = list(ALL_STATUSES).index(Status.PSN)
        self.assertEqual(enc[status_start + expected_idx], 1.0)

    def test_category_is_status(self):
        self.assertEqual(self.ms.category, MoveCategory.STATUS)


# ---------------------------------------------------------------------------
# Accuracy edge cases
# ---------------------------------------------------------------------------

class TestMoveStateAccuracyTrue(MoveStateBaseTest, unittest.TestCase):
    """accuracy=True (never-miss) must be stored as 1.0."""

    def setUp(self):
        self.ms = make_move_state(accuracy=True)

    def test_accuracy_true_maps_to_one(self):
        self.assertAlmostEqual(self.ms.accuracy, 1.0)


class TestMoveStateAccuracyNone(MoveStateBaseTest, unittest.TestCase):
    """accuracy=None must fall back to 0.0."""

    def setUp(self):
        self.ms = make_move_state(accuracy=None)

    def test_accuracy_none_maps_to_zero(self):
        self.assertAlmostEqual(self.ms.accuracy, 0.0)


# ---------------------------------------------------------------------------
# Priority
# ---------------------------------------------------------------------------

class TestMoveStatePriority(MoveStateBaseTest, unittest.TestCase):
    """Quick-attack priority +1 must be normalised correctly."""

    def setUp(self):
        self.ms = make_move_state(priority=1)

    def test_priority_stored(self):
        self.assertEqual(self.ms.priority, 1)

    def test_priority_normalized_in_array(self):
        arr = self.ms.to_array()
        # index 2 = priority
        self.assertAlmostEqual(float(arr[2]), 1.0 / 7.0)


# ---------------------------------------------------------------------------
# Protect / Breaks protect flags
# ---------------------------------------------------------------------------

class TestMoveStateProtect(MoveStateBaseTest, unittest.TestCase):
    """is_protect_move flag must land as 1.0; breaks_protect as 0.0."""

    def setUp(self):
        self.ms = make_move_state(is_protect_move=True, breaks_protect=False)

    def test_is_protect_move_is_one(self):
        self.assertEqual(self.ms.is_protect_move, 1.0)

    def test_breaks_protect_is_zero(self):
        self.assertEqual(self.ms.breaks_protect, 0.0)


class TestMoveStateBreaksProtect(MoveStateBaseTest, unittest.TestCase):
    """breaks_protect flag must land as 1.0."""

    def setUp(self):
        self.ms = make_move_state(is_protect_move=False, breaks_protect=True)

    def test_breaks_protect_is_one(self):
        self.assertEqual(self.ms.breaks_protect, 1.0)

    def test_is_protect_move_is_zero(self):
        self.assertEqual(self.ms.is_protect_move, 0.0)


# ---------------------------------------------------------------------------
# Recoil & drain
# ---------------------------------------------------------------------------

class TestMoveStateRecoil(MoveStateBaseTest, unittest.TestCase):
    """Recoil move (e.g. Double-Edge)."""

    def setUp(self):
        self.ms = make_move_state(base_power=120.0, recoil=0.33)

    def test_recoil_stored(self):
        self.assertAlmostEqual(self.ms.recoil, 0.33)

    def test_drain_zero(self):
        self.assertAlmostEqual(self.ms.drain, 0.0)


class TestMoveStateDrain(MoveStateBaseTest, unittest.TestCase):
    """Drain move (e.g. Mega Drain)."""

    def setUp(self):
        self.ms = make_move_state(base_power=40.0, drain=0.5)

    def test_drain_stored(self):
        self.assertAlmostEqual(self.ms.drain, 0.5)

    def test_recoil_zero(self):
        self.assertAlmostEqual(self.ms.recoil, 0.0)


# ---------------------------------------------------------------------------
# Multi-hit
# ---------------------------------------------------------------------------

class TestMoveStateMultiHitTuple(MoveStateBaseTest, unittest.TestCase):
    """n_hit as tuple (2, 5) — min/max split correctly."""

    def setUp(self):
        self.ms = make_move_state(n_hit=(2, 5))

    def test_min_hits(self):
        self.assertEqual(self.ms.min_hits, 2)

    def test_max_hits(self):
        self.assertEqual(self.ms.max_hits, 5)

    def test_min_hits_normalized(self):
        arr = self.ms.to_array()
        self.assertAlmostEqual(float(arr[-2]), 2.0 / 5.0)

    def test_max_hits_normalized(self):
        arr = self.ms.to_array()
        self.assertAlmostEqual(float(arr[-1]), 5.0 / 5.0)


class TestMoveStateMultiHitInt(MoveStateBaseTest, unittest.TestCase):
    """n_hit as plain int — min and max must be equal."""

    def setUp(self):
        self.ms = make_move_state(n_hit=2)

    def test_min_hits_equals_n_hit(self):
        self.assertEqual(self.ms.min_hits, 2)

    def test_max_hits_equals_n_hit(self):
        self.assertEqual(self.ms.max_hits, 2)


class TestMoveStateSingleHit(MoveStateBaseTest, unittest.TestCase):
    """n_hit=1 (default) — both hit counts are 1."""

    def setUp(self):
        self.ms = make_move_state(n_hit=1)

    def test_min_hits_one(self):
        self.assertEqual(self.ms.min_hits, 1)

    def test_max_hits_one(self):
        self.assertEqual(self.ms.max_hits, 1)


# ---------------------------------------------------------------------------
# Opponent boosts encoding
# ---------------------------------------------------------------------------

class TestMoveStateOppBoosts(MoveStateBaseTest, unittest.TestCase):
    """Move that lowers opponent's Attack by 1 stage."""

    def setUp(self):
        boosts = {**_EMPTY_BOOSTS, "atk": -1}
        self.ms = make_move_state(boosts=boosts)

    def test_opp_boosts_atk_stored(self):
        self.assertEqual(self.ms.opp_boosts["atk"], -1)

    def test_opp_boost_in_array_symmetric(self):
        arr = self.ms.to_array()
        # opp boost slice starts after:
        # 5 scalars + MOVE_CATEGORIES + 3 flags + ALL_STATUSES
        start = 5 + len(MOVE_CATEGORIES) + 3 + len(ALL_STATUSES)
        atk_idx = GEN1_BOOST_KEYS.index("atk")
        self.assertAlmostEqual(float(arr[start + atk_idx]), -1.0 / BOOST_NORM)


# ---------------------------------------------------------------------------
# Self boost encoding
# ---------------------------------------------------------------------------

class TestMoveStateSelfBoost(MoveStateBaseTest, unittest.TestCase):
    """Move that raises own Special Attack by 2 stages (Nasty Plot-like)."""

    def setUp(self):
        self_boost = {**_EMPTY_BOOSTS, "spa": 2}
        self.ms = make_move_state(
            base_power=0.0,
            category=MoveCategory.STATUS,
            self_boost=self_boost,
        )

    def test_self_boost_spa_stored(self):
        self.assertEqual(self.ms.self_boost["spa"], 2)

    def test_self_boost_in_array(self):
        arr = self.ms.to_array()
        opp_start = 5 + len(MOVE_CATEGORIES) + 3 + len(ALL_STATUSES)
        self_start = opp_start + len(GEN1_BOOST_KEYS)
        spa_idx = GEN1_BOOST_KEYS.index("spa")
        self.assertAlmostEqual(float(arr[self_start + spa_idx]), 2.0 / BOOST_NORM)


# ---------------------------------------------------------------------------
# Type multiplier encoding
# ---------------------------------------------------------------------------

class TestEncodeTypeMultiplier(unittest.TestCase):
    """Unit tests for MoveState._encode_type_multiplier (static method)."""

    def test_immune_maps_to_minus_one(self):
        result = MoveState._encode_type_multiplier(0.0)[0]
        self.assertAlmostEqual(result, -1.0)

    def test_neutral_maps_to_zero(self):
        result = MoveState._encode_type_multiplier(1.0)[0]
        self.assertAlmostEqual(result, 0.0)

    def test_super_effective_maps_to_half(self):
        # 2x -> log2(2)/2 = 0.5
        result = MoveState._encode_type_multiplier(2.0)[0]
        self.assertAlmostEqual(result, 0.5)

    def test_double_super_effective_maps_to_one(self):
        # 4x -> log2(4)/2 = 1.0
        result = MoveState._encode_type_multiplier(4.0)[0]
        self.assertAlmostEqual(result, 1.0)

    def test_not_very_effective_maps_to_negative(self):
        # 0.5x -> log2(0.5)/2 = -0.5
        result = MoveState._encode_type_multiplier(0.5)[0]
        self.assertAlmostEqual(result, -0.5)

    def test_quarter_effective_maps_to_minus_one(self):
        # 0.25x -> log2(0.25)/2 = -1.0
        result = MoveState._encode_type_multiplier(0.25)[0]
        self.assertAlmostEqual(result, -1.0)


# ---------------------------------------------------------------------------
# describe() / __repr__
# ---------------------------------------------------------------------------

class TestMoveStateDescribe(unittest.TestCase):
    """describe() and __repr__ must return non-empty strings."""

    def setUp(self):
        self.ms = make_move_state(
            base_power=80.0,
            category=MoveCategory.PHYSICAL,
            move_id="body-slam",
            status=Status.PAR,
        )

    def test_describe_returns_string(self):
        self.assertIsInstance(self.ms.describe(), str)

    def test_describe_not_empty(self):
        self.assertTrue(len(self.ms.describe()) > 0)

    def test_describe_contains_move_id(self):
        self.assertIn("body-slam", self.ms.describe())

    def test_describe_contains_base_power(self):
        self.assertIn("80", self.ms.describe())

    def test_repr_returns_string(self):
        self.assertIsInstance(repr(self.ms), str)

    def test_repr_contains_base_power(self):
        self.assertIn("80", repr(self.ms))


# ---------------------------------------------------------------------------
# Heal
# ---------------------------------------------------------------------------

class TestMoveStateHeal(MoveStateBaseTest, unittest.TestCase):
    """Move that heals 50 % HP (Recover-like)."""

    def setUp(self):
        self.ms = make_move_state(heal=0.5, base_power=0.0, category=MoveCategory.STATUS)

    def test_heal_stored(self):
        self.assertAlmostEqual(self.ms.heal, 0.5)

    def test_heal_normalized_in_array(self):
        arr = self.ms.to_array()
        # index 3 = heal
        self.assertAlmostEqual(float(arr[3]), 0.5)


if __name__ == "__main__":
    unittest.main()
