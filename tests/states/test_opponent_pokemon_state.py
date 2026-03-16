"""
Tests for OpponentPokemonState.

Fixture files used
------------------
gen1_starmie_opponent.json  — PAR, full HP, no boosts, no preparing, no protect
gen1_tauros_opponent.json   — no status, full HP, boosted, preparing=True
gen1_chansey_opponent.json  — BRN, half HP, +3 def, protect_belief=0.75
"""

import json
import unittest
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np

from poke_env.battle.status import Status

from env.states.opponent_pokemon_state import OpponentPokemonState, _STAT_BELIEF_DIM
from env.states.pokemon_state import (
    PokemonState,
    ALL_STATUSES,
    TRACKED_EFFECTS,
)

FIXTURES_DIR = Path(__file__).parent.parent / "fixtures"
_HALF = _STAT_BELIEF_DIM // 2   # 6


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------

def make_opponent_mock_from_fixture(filename: str):
    """Load an opponent fixture and return (mock_pokemon, mock_stat_belief, protect_belief, raw_data).

    The mock_stat_belief mimics ``StatBelief.to_array()`` by returning a
    12-element float32 array built from the fixture's ``stat_belief`` section.
    """
    with open(FIXTURES_DIR / filename) as f:
        data = json.load(f)

    # ── Pokémon mock ─────────────────────────────────────────────────────
    p = MagicMock()
    p.current_hp_fraction = data["current_hp_fraction"]
    p.species             = data["species"]
    p.boosts              = data["boosts"]
    p.status              = Status[data["status"]] if data["status"] else None
    p.effects             = {}
    p.stab_multiplier     = data["stab_multiplier"]
    p.preparing           = data.get("preparing", False)

    # ── StatBelief mock ──────────────────────────────────────────────────
    belief_section = data["stat_belief"]
    full_array = np.array(
        belief_section["mean"] + belief_section["std"],
        dtype=np.float32,
    )
    assert len(full_array) == _STAT_BELIEF_DIM, (
        f"Fixture {filename}: stat_belief must have {_STAT_BELIEF_DIM} total elements "
        f"(mean×{_HALF} + std×{_HALF}), got {len(full_array)}"
    )
    stat_belief = MagicMock()
    stat_belief.to_array.return_value = full_array

    protect_belief = float(data.get("protect_belief", 0.0))
    return p, stat_belief, protect_belief, data


# ---------------------------------------------------------------------------
# Shared structural base
# ---------------------------------------------------------------------------

class OpponentPokemonStateBaseTest:
    """
    Mixin — subclasses set self.ops in setUp().
    Does NOT inherit TestCase so it isn't run directly.
    """
    ops: OpponentPokemonState

    # ── stat belief arrays ───────────────────────────────────────────────

    def test_stats_shape(self):
        self.assertEqual(self.ops.stats.shape, (_HALF,))

    def test_stats_dtype(self):
        self.assertEqual(self.ops.stats.dtype, np.float32)

    def test_stats_in_range(self):
        """Belief means must be normalised to [0, 1]."""
        self.assertTrue(np.all(self.ops.stats >= 0.0))
        self.assertTrue(np.all(self.ops.stats <= 1.0))

    def test_stats_std_shape(self):
        self.assertEqual(self.ops.stats_std.shape, (_HALF,))

    def test_stats_std_dtype(self):
        self.assertEqual(self.ops.stats_std.dtype, np.float32)

    def test_stats_std_non_negative(self):
        self.assertTrue(np.all(self.ops.stats_std >= 0.0))

    # ── boosts ───────────────────────────────────────────────────────────

    def test_boosts_shape(self):
        self.assertEqual(self.ops.boosts.shape, (len(PokemonState.BOOST_KEYS),))

    def test_boosts_dtype(self):
        self.assertEqual(self.ops.boosts.dtype, np.float32)

    def test_boosts_encoded_in_range(self):
        enc = self.ops.boosts_encoded()
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

    # ── opponent-specific scalars ─────────────────────────────────────────

    def test_preparing_is_float(self):
        self.assertIsInstance(self.ops.preparing, float)

    def test_preparing_is_binary(self):
        self.assertIn(self.ops.preparing, [0.0, 1.0])

    def test_protect_belief_is_float(self):
        self.assertIsInstance(self.ops.protect_belief, float)

    def test_protect_belief_in_range(self):
        self.assertGreaterEqual(self.ops.protect_belief, 0.0)
        self.assertLessEqual(self.ops.protect_belief, 1.0)

    # ── stab: raw multiplier on object, normalised in to_array ───────────

    def test_stab_raw_value(self):
        self.assertEqual(self.ops.stab, 0.75)

    # ── to_array ─────────────────────────────────────────────────────────

    def test_to_array_length(self):
        self.assertEqual(len(self.ops.to_array()), self.ops.array_len())

    def test_to_array_dtype(self):
        self.assertEqual(self.ops.to_array().dtype, np.float32)

    def test_array_len_formula(self):
        """array_len must equal 1 + 12 + |BOOST_KEYS| + |STATUSES| + |EFFECTS| + 3."""
        expected = (
            1                       # hp
            + _STAT_BELIEF_DIM      # stat means + stds
            + len(PokemonState.BOOST_KEYS)
            + len(ALL_STATUSES)
            + len(TRACKED_EFFECTS)
            + 1                     # preparing
            + 1                     # stab
            + 1                     # protect_belief
        )
        self.assertEqual(self.ops.array_len(), expected)

    def test_to_array_hp_first(self):
        """First element of to_array() must be the HP fraction."""
        self.assertAlmostEqual(float(self.ops.to_array()[0]), self.ops.hp)

    def test_to_array_protect_belief_last(self):
        """Last element of to_array() must equal protect_belief."""
        self.assertAlmostEqual(float(self.ops.to_array()[-1]), self.ops.protect_belief)

    def test_to_array_stat_means_slice(self):
        """to_array()[1 : 1+HALF] must match self.stats."""
        arr = self.ops.to_array()
        np.testing.assert_array_almost_equal(arr[1: 1 + _HALF], self.ops.stats)

    def test_to_array_stat_stds_slice(self):
        """to_array()[1+HALF : 1+DIM] must match self.stats_std."""
        arr = self.ops.to_array()
        np.testing.assert_array_almost_equal(
            arr[1 + _HALF: 1 + _STAT_BELIEF_DIM],
            self.ops.stats_std,
        )


# ---------------------------------------------------------------------------
# Empty state
# ---------------------------------------------------------------------------

class TestOpponentPokemonStateEmpty(OpponentPokemonStateBaseTest, unittest.TestCase):
    """OpponentPokemonState with no pokemon or beliefs — almost all zeros."""

    def setUp(self):
        self.ops = OpponentPokemonState()

    def test_hp_is_zero(self):
        self.assertEqual(self.ops.hp, 0.0)

    def test_species_is_none(self):
        self.assertEqual(self.ops.species, "none")

    def test_stats_all_zero(self):
        np.testing.assert_array_equal(self.ops.stats, np.zeros(_HALF, dtype=np.float32))

    def test_stats_std_all_zero(self):
        np.testing.assert_array_equal(self.ops.stats_std, np.zeros(_HALF, dtype=np.float32))

    def test_boosts_all_zero(self):
        np.testing.assert_array_equal(
            self.ops.boosts, np.zeros(len(PokemonState.BOOST_KEYS), dtype=np.float32)
        )

    def test_status_all_zero(self):
        np.testing.assert_array_equal(self.ops.status, np.zeros(len(ALL_STATUSES), dtype=np.float32))

    def test_effects_all_zero(self):
        np.testing.assert_array_equal(
            self.ops.effects, np.zeros(len(TRACKED_EFFECTS), dtype=np.float32)
        )

    def test_preparing_is_zero(self):
        self.assertEqual(self.ops.preparing, 0.0)

    def test_protect_belief_is_zero(self):
        self.assertEqual(self.ops.protect_belief, 0.0)


# ---------------------------------------------------------------------------
# Starmie  –  Water/Psychic, PAR, full HP, no boosts, not preparing
# ---------------------------------------------------------------------------

class TestOpponentPokemonStateStarmie(OpponentPokemonStateBaseTest, unittest.TestCase):

    def setUp(self):
        pokemon, stat_belief, protect_belief, self.data = make_opponent_mock_from_fixture(
            "gen1_starmie_opponent.json"
        )
        self.ops = OpponentPokemonState(pokemon, stat_belief=stat_belief, protect_belief=protect_belief)

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

    def test_protect_belief_zero(self):
        self.assertEqual(self.ops.protect_belief, 0.0)

    def test_stat_means_match_fixture(self):
        expected = np.array(self.data["stat_belief"]["mean"], dtype=np.float32)
        np.testing.assert_array_almost_equal(self.ops.stats, expected)

    def test_stat_stds_match_fixture(self):
        expected = np.array(self.data["stat_belief"]["std"], dtype=np.float32)
        np.testing.assert_array_almost_equal(self.ops.stats_std, expected)


# ---------------------------------------------------------------------------
# Tauros  –  no status, full HP, stat boosts, preparing=True
# ---------------------------------------------------------------------------

class TestOpponentPokemonStateTauros(OpponentPokemonStateBaseTest, unittest.TestCase):

    def setUp(self):
        pokemon, stat_belief, protect_belief, self.data = make_opponent_mock_from_fixture(
            "gen1_tauros_opponent.json"
        )
        self.ops = OpponentPokemonState(pokemon, stat_belief=stat_belief, protect_belief=protect_belief)

    def test_hp(self):
        self.assertEqual(self.ops.hp, 1.0)

    def test_species(self):
        self.assertEqual(self.ops.species, "tauros")

    def test_no_status(self):
        self.assertEqual(self.ops.status.sum(), 0.0)

    def test_preparing_true(self):
        """Charging a two-turn move — preparing must be 1.0."""
        self.assertEqual(self.ops.preparing, 1.0)

    def test_protect_belief_zero(self):
        self.assertEqual(self.ops.protect_belief, 0.0)

    def test_atk_boost(self):
        atk_idx = PokemonState.BOOST_KEYS.index("atk")
        self.assertEqual(self.ops.boosts[atk_idx], 2.0)

    def test_spa_boost_negative(self):
        spa_idx = PokemonState.BOOST_KEYS.index("spa")
        self.assertEqual(self.ops.boosts[spa_idx], -2.0)

    def test_spe_boost(self):
        spe_idx = PokemonState.BOOST_KEYS.index("spe")
        self.assertEqual(self.ops.boosts[spe_idx], 1.0)

    def test_atk_boost_encoded(self):
        """+2 stage → encoded value of 2/6 ≈ 0.333."""
        atk_idx = PokemonState.BOOST_KEYS.index("atk")
        self.assertAlmostEqual(float(self.ops.boosts_encoded()[atk_idx]), 2 / 6, places=5)

    def test_spa_boost_encoded(self):
        """−2 stage → encoded value of −2/6 ≈ −0.333."""
        spa_idx = PokemonState.BOOST_KEYS.index("spa")
        self.assertAlmostEqual(float(self.ops.boosts_encoded()[spa_idx]), -2 / 6, places=5)

    def test_stat_means_match_fixture(self):
        expected = np.array(self.data["stat_belief"]["mean"], dtype=np.float32)
        np.testing.assert_array_almost_equal(self.ops.stats, expected)

    def test_preparing_in_to_array(self):
        """The preparing flag must appear correctly in the serialised array."""
        arr = self.ops.to_array()
        # preparing sits at offset: 1 + DIM + |BOOST_KEYS| + |STATUSES| + |EFFECTS|
        preparing_idx = (
            1
            + _STAT_BELIEF_DIM
            + len(PokemonState.BOOST_KEYS)
            + len(ALL_STATUSES)
            + len(TRACKED_EFFECTS)
        )
        self.assertAlmostEqual(float(arr[preparing_idx]), 1.0)


# ---------------------------------------------------------------------------
# Chansey  –  BRN, half HP, +3 def, protect_belief=0.75
# ---------------------------------------------------------------------------

class TestOpponentPokemonStateChansey(OpponentPokemonStateBaseTest, unittest.TestCase):

    def setUp(self):
        pokemon, stat_belief, protect_belief, self.data = make_opponent_mock_from_fixture(
            "gen1_chansey_opponent.json"
        )
        self.ops = OpponentPokemonState(pokemon, stat_belief=stat_belief, protect_belief=protect_belief)

    def test_hp_fraction(self):
        self.assertAlmostEqual(self.ops.hp, 0.5)

    def test_species(self):
        self.assertEqual(self.ops.species, "chansey")

    def test_status_brn(self):
        self.assertEqual(self.ops.status[ALL_STATUSES.index(Status.BRN)], 1.0)

    def test_only_one_status_set(self):
        self.assertEqual(self.ops.status.sum(), 1.0)

    def test_def_boost(self):
        def_idx = PokemonState.BOOST_KEYS.index("def")
        self.assertEqual(self.ops.boosts[def_idx], 3.0)

    def test_other_boosts_zero(self):
        def_idx = PokemonState.BOOST_KEYS.index("def")
        np.testing.assert_array_equal(
            np.delete(self.ops.boosts, def_idx),
            np.zeros(len(PokemonState.BOOST_KEYS) - 1)
        )

    def test_preparing_false(self):
        self.assertEqual(self.ops.preparing, 0.0)

    def test_protect_belief(self):
        self.assertAlmostEqual(self.ops.protect_belief, 0.75)

    def test_protect_belief_last_in_array(self):
        """protect_belief must appear as the final element of to_array()."""
        arr = self.ops.to_array()
        self.assertAlmostEqual(float(arr[-1]), 0.75)

    def test_stat_means_hp_capped(self):
        """Chansey's HP (703) exceeds STAT_NORM; belief mean must be 1.0."""
        self.assertAlmostEqual(float(self.ops.stats[0]), 1.0)

    def test_stat_stds_non_negative(self):
        self.assertTrue(np.all(self.ops.stats_std >= 0.0))


# ---------------------------------------------------------------------------
# Belief injection
# ---------------------------------------------------------------------------

class TestOpponentPokemonStateBelief(unittest.TestCase):
    """Unit tests for StatBelief injection independent of a specific fixture."""

    def _make_ops(self, mean, std, *, protect_belief=0.0, preparing=False):
        """Construct OpponentPokemonState from explicit mean/std arrays."""
        full = np.array(mean + std, dtype=np.float32)
        assert len(full) == _STAT_BELIEF_DIM

        pokemon = MagicMock()
        pokemon.current_hp_fraction = 1.0
        pokemon.species = "test"
        pokemon.boosts = {}
        pokemon.status = None
        pokemon.effects = {}
        pokemon.stab_multiplier = 1.5
        pokemon.preparing = preparing

        belief = MagicMock()
        belief.to_array.return_value = full

        return OpponentPokemonState(pokemon, stat_belief=belief, protect_belief=protect_belief)

    def test_stats_equal_mean(self):
        mean = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
        std  = [0.01] * 6
        ops = self._make_ops(mean, std)
        np.testing.assert_array_almost_equal(ops.stats, np.array(mean, dtype=np.float32))

    def test_stats_std_equal_std(self):
        mean = [0.5] * 6
        std  = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
        ops = self._make_ops(mean, std)
        np.testing.assert_array_almost_equal(ops.stats_std, np.array(std, dtype=np.float32))

    def test_zero_belief_gives_zero_stats(self):
        ops = self._make_ops([0.0] * 6, [0.0] * 6)
        np.testing.assert_array_equal(ops.stats,     np.zeros(_HALF, dtype=np.float32))
        np.testing.assert_array_equal(ops.stats_std, np.zeros(_HALF, dtype=np.float32))

    def test_none_belief_gives_zero_stats(self):
        pokemon = MagicMock()
        pokemon.current_hp_fraction = 1.0
        pokemon.species = "test"
        pokemon.boosts = {}
        pokemon.status = None
        pokemon.effects = {}
        pokemon.stab_multiplier = 1.5
        pokemon.preparing = False
        ops = OpponentPokemonState(pokemon, stat_belief=None)
        np.testing.assert_array_equal(ops.stats,     np.zeros(_HALF, dtype=np.float32))
        np.testing.assert_array_equal(ops.stats_std, np.zeros(_HALF, dtype=np.float32))

    def test_protect_belief_stored_correctly(self):
        ops = self._make_ops([0.5] * 6, [0.05] * 6, protect_belief=0.42)
        self.assertAlmostEqual(ops.protect_belief, 0.42)

    def test_preparing_true_stored_correctly(self):
        ops = self._make_ops([0.5] * 6, [0.05] * 6, preparing=True)
        self.assertEqual(ops.preparing, 1.0)

    def test_preparing_false_stored_correctly(self):
        ops = self._make_ops([0.5] * 6, [0.05] * 6, preparing=False)
        self.assertEqual(ops.preparing, 0.0)


if __name__ == "__main__":
    unittest.main()
