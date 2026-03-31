import unittest
import numpy as np
from poke_env.battle.effect import Effect
from env.states.state_utils import (
    normalize,
    normalize_vector,
    encode_enum,
    encode_dicts,
    pull_attribute,
    ALL_STATUSES,
    GEN1_BOOST_KEYS,
    GEN1_STAT_KEYS,
    GEN1_TRACKED_EFFECTS,
    MODERN_STAT_KEYS,
    MODERN_BOOST_KEYS,
    BOOST_NORM,
    STAT_NORM,
    STAB_NORM,
)


class TestConstants(unittest.TestCase):
    def test_boost_norm(self):
        self.assertEqual(BOOST_NORM, 6.0)

    def test_stat_norm(self):
        self.assertEqual(STAT_NORM, 600.0)

    def test_stab_norm(self):
        self.assertEqual(STAB_NORM, 2.25)

    def test_all_statuses_is_list(self):
        self.assertIsInstance(ALL_STATUSES, list)
        self.assertGreater(len(ALL_STATUSES), 0)

    def test_gen1_tracked_effects(self):
        self.assertIn(Effect.CONFUSION, GEN1_TRACKED_EFFECTS)
        self.assertIn(Effect.ENCORE, GEN1_TRACKED_EFFECTS)

    def test_gen1_boost_keys(self):
        self.assertEqual(GEN1_BOOST_KEYS, ["atk", "def", "spa", "spe", "accuracy", "evasion"])

    def test_gen1_stat_keys(self):
        self.assertEqual(GEN1_STAT_KEYS, ["hp", "atk", "def", "spc", "spe"])

    def test_modern_stat_keys(self):
        self.assertEqual(MODERN_STAT_KEYS, ["hp", "atk", "def", "spa", "spd", "spe"])

    def test_modern_boost_keys(self):
        self.assertEqual(MODERN_BOOST_KEYS, ["atk", "def", "spa", "spd", "spe", "accuracy", "evasion"])


class TestNormalize(unittest.TestCase):
    def test_basic(self):
        self.assertAlmostEqual(normalize(50.0, 100.0), 0.5)

    def test_exact_max(self):
        self.assertAlmostEqual(normalize(100.0, 100.0), 1.0)

    def test_zero_value(self):
        self.assertAlmostEqual(normalize(0.0, 100.0), 0.0)

    def test_clamp_above_max(self):
        self.assertEqual(normalize(200.0, 100.0), 1.0)

    def test_clamp_below_zero(self):
        self.assertEqual(normalize(-10.0, 100.0), 0.0)

    def test_symmetric_positive(self):
        self.assertAlmostEqual(normalize(6.0, 6.0, symmetric=True), 1.0)

    def test_symmetric_negative(self):
        self.assertAlmostEqual(normalize(-6.0, 6.0, symmetric=True), -1.0)

    def test_symmetric_zero(self):
        self.assertAlmostEqual(normalize(0.0, 6.0, symmetric=True), 0.0)

    def test_symmetric_clamp_above(self):
        self.assertEqual(normalize(100.0, 6.0, symmetric=True), 1.0)

    def test_symmetric_clamp_below(self):
        self.assertEqual(normalize(-100.0, 6.0, symmetric=True), -1.0)

    def test_zero_max_returns_zero(self):
        self.assertEqual(normalize(10.0, 0.0), 0.0)

    def test_negative_max_returns_zero(self):
        self.assertEqual(normalize(10.0, -1.0), 0.0)

    def test_default_max_is_one(self):
        self.assertAlmostEqual(normalize(0.5), 0.5)


class TestNormalizeVector(unittest.TestCase):
    def test_basic(self):
        vec = np.array([50.0, 100.0, 150.0])
        result = normalize_vector(vec, 100.0)
        np.testing.assert_array_almost_equal(result, [0.5, 1.0, 1.0])

    def test_clamp_above(self):
        result = normalize_vector(np.array([200.0]), 100.0)
        self.assertEqual(result[0], 1.0)

    def test_clamp_below_zero(self):
        result = normalize_vector(np.array([-50.0]), 100.0)
        self.assertEqual(result[0], 0.0)

    def test_symmetric(self):
        vec = np.array([-6.0, 0.0, 6.0])
        result = normalize_vector(vec, 6.0, symmetric=True)
        np.testing.assert_array_almost_equal(result, [-1.0, 0.0, 1.0])

    def test_symmetric_clamp_above(self):
        result = normalize_vector(np.array([100.0]), 6.0, symmetric=True)
        self.assertEqual(result[0], 1.0)

    def test_symmetric_clamp_below(self):
        result = normalize_vector(np.array([-100.0]), 6.0, symmetric=True)
        self.assertEqual(result[0], -1.0)

    def test_all_zero_max_returns_zeros(self):
        result = normalize_vector(np.array([1.0, 2.0]), 0.0)
        np.testing.assert_array_equal(result, [0.0, 0.0])

    def test_partial_zero_max_returns_all_zeros(self):
        # any(vec_max <= 0) → True → all zeros
        result = normalize_vector(np.array([50.0, 100.0]), 0.0)
        np.testing.assert_array_equal(result, [0.0, 0.0])

    def test_negative_max_returns_zeros(self):
        result = normalize_vector(np.array([50.0]), -1.0)
        np.testing.assert_array_equal(result, [0.0])

    def test_output_dtype_float32(self):
        result = normalize_vector(np.array([1.0]), 1.0)
        self.assertEqual(result.dtype, np.float32)


class TestEncodeEnum(unittest.TestCase):
    def test_none_returns_zeros(self):
        result = encode_enum(None, ALL_STATUSES)
        np.testing.assert_array_equal(result, np.zeros(len(ALL_STATUSES)))

    def test_none_enums_list_raises(self):
        with self.assertRaises(ValueError):
            encode_enum(None, None)

    def test_all_statuses_one_hot(self):
        for i, status in enumerate(ALL_STATUSES):
            result = encode_enum(status, ALL_STATUSES)
            self.assertEqual(result[i], 1.0)
            self.assertEqual(result.sum(), 1.0)

    def test_output_length_matches_enums(self):
        self.assertEqual(len(encode_enum(None, ALL_STATUSES)), len(ALL_STATUSES))

    def test_output_dtype_float32(self):
        self.assertEqual(encode_enum(None, ALL_STATUSES).dtype, np.float32)

    def test_effect_confusion(self):
        result = encode_enum(Effect.CONFUSION, GEN1_TRACKED_EFFECTS)
        np.testing.assert_array_equal(result, [1.0, 0.0])

    def test_effect_encore(self):
        result = encode_enum(Effect.ENCORE, GEN1_TRACKED_EFFECTS)
        np.testing.assert_array_equal(result, [0.0, 1.0])

    def test_unknown_value_returns_zeros(self):
        result = encode_enum(Effect.AQUA_RING, GEN1_TRACKED_EFFECTS)
        np.testing.assert_array_equal(result, [0.0, 0.0])

    def test_last_element_one_hot(self):
        status = ALL_STATUSES[-1]
        result = encode_enum(status, ALL_STATUSES)
        self.assertEqual(result[-1], 1.0)
        self.assertEqual(result.sum(), 1.0)

    # -- collection (multi-hot) inputs ------------------------------------

    def test_dict_single_effect(self):
        # pokemon.effects is a dict keyed by Effect
        result = encode_enum({Effect.CONFUSION: object()}, GEN1_TRACKED_EFFECTS)
        np.testing.assert_array_equal(result, [1.0, 0.0])

    def test_dict_multiple_effects(self):
        result = encode_enum(
            {Effect.CONFUSION: object(), Effect.ENCORE: object()},
            GEN1_TRACKED_EFFECTS,
        )
        np.testing.assert_array_equal(result, [1.0, 1.0])

    def test_dict_empty_returns_zeros(self):
        result = encode_enum({}, GEN1_TRACKED_EFFECTS)
        np.testing.assert_array_equal(result, [0.0, 0.0])

    def test_dict_unknown_key_returns_zeros(self):
        result = encode_enum({Effect.AQUA_RING: object()}, GEN1_TRACKED_EFFECTS)
        np.testing.assert_array_equal(result, [0.0, 0.0])

    def test_set_single_effect(self):
        result = encode_enum({Effect.ENCORE}, GEN1_TRACKED_EFFECTS)
        np.testing.assert_array_equal(result, [0.0, 1.0])

    def test_set_both_effects(self):
        result = encode_enum({Effect.CONFUSION, Effect.ENCORE}, GEN1_TRACKED_EFFECTS)
        np.testing.assert_array_equal(result, [1.0, 1.0])

    def test_list_single_effect(self):
        result = encode_enum([Effect.CONFUSION], GEN1_TRACKED_EFFECTS)
        np.testing.assert_array_equal(result, [1.0, 0.0])

    def test_frozenset_single_effect(self):
        result = encode_enum(frozenset([Effect.ENCORE]), GEN1_TRACKED_EFFECTS)
        np.testing.assert_array_equal(result, [0.0, 1.0])

    def test_collection_sum_can_exceed_one(self):
        # multi-hot: both bits set → sum == 2, unlike one-hot status
        result = encode_enum({Effect.CONFUSION, Effect.ENCORE}, GEN1_TRACKED_EFFECTS)
        self.assertEqual(result.sum(), 2.0)

    def test_collection_output_dtype_float32(self):
        result = encode_enum({Effect.CONFUSION}, GEN1_TRACKED_EFFECTS)
        self.assertEqual(result.dtype, np.float32)

    def test_collection_output_length_matches_enums(self):
        result = encode_enum({Effect.CONFUSION}, GEN1_TRACKED_EFFECTS)
        self.assertEqual(len(result), len(GEN1_TRACKED_EFFECTS))

    def test_empty_list_returns_zeros(self):
        result = encode_enum([], GEN1_TRACKED_EFFECTS)
        np.testing.assert_array_equal(result, [0.0, 0.0])

    def test_tuple_is_not_a_collection_path(self):
        # tuple is NOT in the isinstance guard (dict/set/list/frozenset)
        # so it falls through to the single-value path: tuple == Effect → False → all zeros
        result = encode_enum((Effect.CONFUSION,), GEN1_TRACKED_EFFECTS)
        np.testing.assert_array_equal(result, [0.0, 0.0])


class TestEncodeDicts(unittest.TestCase):
    def test_all_keys_present(self):
        d = {k: i for i, k in enumerate(MODERN_BOOST_KEYS)}
        result = encode_dicts(d, MODERN_BOOST_KEYS)
        np.testing.assert_array_equal(result, list(range(len(MODERN_BOOST_KEYS))))

    def test_missing_keys_default_to_zero(self):
        np.testing.assert_array_equal(
            encode_dicts({}, MODERN_BOOST_KEYS),
            np.zeros(len(MODERN_BOOST_KEYS))
        )

    def test_partial_keys(self):
        result = encode_dicts({"atk": 3}, GEN1_BOOST_KEYS)
        self.assertEqual(result[0], 3.0)
        np.testing.assert_array_equal(result[1:], np.zeros(len(GEN1_BOOST_KEYS) - 1))

    def test_extra_keys_ignored(self):
        d = {k: 1 for k in MODERN_BOOST_KEYS}
        d["unknown_key"] = 999
        self.assertEqual(len(encode_dicts(d, MODERN_BOOST_KEYS)), len(MODERN_BOOST_KEYS))

    def test_negative_values(self):
        result = encode_dicts({"atk": -2, "def": -6}, ["atk", "def"])
        np.testing.assert_array_equal(result, [-2.0, -6.0])

    def test_gen1_stat_keys(self):
        stats = {"hp": 300, "atk": 100, "def": 80, "spc": 90, "spe": 110}
        result = encode_dicts(stats, GEN1_STAT_KEYS)
        self.assertEqual(len(result), len(GEN1_STAT_KEYS))
        self.assertAlmostEqual(float(result[0]), 300.0)

    def test_output_dtype_float32(self):
        self.assertEqual(encode_dicts({}, GEN1_BOOST_KEYS).dtype, np.float32)

    def test_output_length_matches_keys(self):
        self.assertEqual(len(encode_dicts({}, MODERN_STAT_KEYS)), len(MODERN_STAT_KEYS))

    def test_key_order_is_preserved(self):
        # result must follow _keys order, not dict insertion order
        result = encode_dicts({"spe": 9, "hp": 1}, ["hp", "spe"])
        np.testing.assert_array_equal(result, [1.0, 9.0])

    def test_float_values_pass_through(self):
        result = encode_dicts({"atk": 2.5}, ["atk"])
        self.assertAlmostEqual(float(result[0]), 2.5)

    def test_none_value_treated_as_zero(self):
        # dict.get(k, 0) returns None when the key exists with value None
        # the function must treat None as 0 rather than raising TypeError
        result = encode_dicts({"atk": None, "def": 3}, ["atk", "def"])
        self.assertEqual(float(result[0]), 0.0)
        self.assertEqual(float(result[1]), 3.0)


class TestPullAttribute(unittest.TestCase):
    def test_existing_attribute(self):
        class Obj:
            hp = 250
        self.assertEqual(pull_attribute(Obj(), "hp", 0, int), 250)

    def test_missing_attribute_returns_default(self):
        class Obj:
            pass
        self.assertEqual(pull_attribute(Obj(), "hp", 99, int), 99)

    def test_type_conversion_to_float(self):
        class Obj:
            hp = "300"
        self.assertAlmostEqual(pull_attribute(Obj(), "hp", 0, float), 300.0)

    def test_type_conversion_to_int(self):
        class Obj:
            hp = 3.9
        self.assertEqual(pull_attribute(Obj(), "hp", 0, int), 3)

    def test_bool_conversion(self):
        class Obj:
            active = 1
        self.assertTrue(pull_attribute(Obj(), "active", False, bool))

    def test_none_obj_returns_default_int(self):
        self.assertEqual(pull_attribute(None, "hp", 42, int), 42)

    def test_none_obj_returns_default_float(self):
        self.assertAlmostEqual(pull_attribute(None, "stab_multiplier", 1.5, float), 1.5)

    def test_none_key_returns_default(self):
        class Obj:
            hp = 250
        self.assertEqual(pull_attribute(Obj(), None, 99, int), 99)

    def test_none_obj_and_none_key_returns_default(self):
        self.assertEqual(pull_attribute(None, None, 0, int), 0)

    def test_attribute_value_is_none_returns_default(self):
        # getattr returns None when the attribute exists but its value is None
        # pull_attribute must fall back to default_value in this case
        class Obj:
            stab_multiplier = None
        self.assertAlmostEqual(pull_attribute(Obj(), "stab_multiplier", 1.5, float), 1.5)



if __name__ == '__main__':
    unittest.main()

