import json
import unittest
from unittest.mock import MagicMock
import numpy as np
from poke_env.battle.status import Status
from poke_env.battle.effect import Effect
from env.states.pokemon_state import PokemonState
from env.states.state_utils import (
    GEN1_BOOST_KEYS,
    GEN1_STAT_KEYS,
    GEN1_TRACKED_EFFECTS,
    ALL_STATUSES,
)
from pathlib import Path

FIXTURES_DIR = Path(__file__).parent.parent / "fixtures"


# ---------------------------------------------------------------------------
# Minimal concrete stub — implements the abstract interface with no extras
# so we test only PokemonState.__init__, not any subclass logic
# ---------------------------------------------------------------------------

class _StubState(PokemonState):
    def to_array(self): return np.array([])
    def array_len(self): return 0
    def describe(self): return ""
    def __repr__(self): return ""


def make_mock_from_fixture(filename: str) -> MagicMock:
    with open(FIXTURES_DIR / filename) as f:
        data = json.load(f)
    p = MagicMock()
    p.current_hp_fraction = data["current_hp_fraction"]
    p.species             = data["species"]
    p.stats               = data["stats"]
    p.boosts              = data["boosts"]
    p.status              = Status[data["status"]] if data["status"] else None
    p.effects             = {}
    p.types               = tuple(data["types"])
    p.stab_multiplier     = data["stab_multiplier"]
    return p


# ---------------------------------------------------------------------------
# Class constants — must REFERENCE state_utils objects, not copies
# ---------------------------------------------------------------------------

class TestPokemonStateClassConstants(unittest.TestCase):

    def test_stat_keys_are_gen1(self):
        self.assertIs(PokemonState.STAT_KEYS, GEN1_STAT_KEYS)

    def test_boost_keys_are_gen1(self):
        self.assertIs(PokemonState.BOOST_KEYS, GEN1_BOOST_KEYS)

    def test_tracked_effects_are_gen1(self):
        self.assertIs(PokemonState.TRACKED_EFFECTS, GEN1_TRACKED_EFFECTS)


# ---------------------------------------------------------------------------
# Empty state (pokemon=None) — every field must be a safe zero/default
# ---------------------------------------------------------------------------

class TestPokemonStateInitEmpty(unittest.TestCase):

    def setUp(self):
        self.ps = _StubState()

    def test_level_is_100(self):
        self.assertEqual(self.ps.level, 100)

    def test_hp_is_zero(self):
        self.assertEqual(self.ps.hp, 0.0)

    def test_species_is_none_string(self):
        self.assertEqual(self.ps.species, "none")

    def test_stab_is_default(self):
        self.assertAlmostEqual(self.ps.stab, 1.5)

    def test_types_is_none_list(self):
        self.assertEqual(self.ps.types, [None])

    def test_stats_all_zero(self):
        np.testing.assert_array_equal(self.ps.stats, np.zeros(len(GEN1_STAT_KEYS)))

    def test_boosts_all_zero(self):
        np.testing.assert_array_equal(self.ps.boosts, np.zeros(len(GEN1_BOOST_KEYS)))

    def test_status_all_zero(self):
        np.testing.assert_array_equal(self.ps.status, np.zeros(len(ALL_STATUSES)))

    def test_effects_all_zero(self):
        np.testing.assert_array_equal(self.ps.effects, np.zeros(len(GEN1_TRACKED_EFFECTS)))


# ---------------------------------------------------------------------------
# Populated state — every field read from pokemon correctly
# Chansey: BRN, hp=0.5, +3 def, Normal type
# ---------------------------------------------------------------------------

class TestPokemonStateInitPopulated(unittest.TestCase):

    def setUp(self):
        self.ps = _StubState(make_mock_from_fixture("gen1_chansey.json"))

    def test_level_always_100(self):
        self.assertEqual(self.ps.level, 100)

    def test_hp_read_from_pokemon(self):
        self.assertAlmostEqual(self.ps.hp, 0.5)

    def test_species_read_from_pokemon(self):
        self.assertEqual(self.ps.species, "chansey")

    def test_stab_read_from_pokemon(self):
        self.assertAlmostEqual(self.ps.stab, 1.5)

    def test_types_read_from_pokemon(self):
        self.assertIn("NORMAL", self.ps.types)

    def test_dual_types_both_stored(self):
        # Starmie: WATER + PSYCHIC — both must be present
        ps = _StubState(make_mock_from_fixture("gen1_starmie.json"))
        self.assertIn("WATER", ps.types)
        self.assertIn("PSYCHIC", ps.types)

    def test_status_brn_encoded_at_correct_index(self):
        brn_idx = ALL_STATUSES.index(Status.BRN)
        self.assertEqual(self.ps.status[brn_idx], 1.0)
        self.assertEqual(self.ps.status.sum(), 1.0)

    def test_boosts_def_encoded_correctly(self):
        # Chansey has +3 def
        def_idx = GEN1_BOOST_KEYS.index("def")
        self.assertEqual(self.ps.boosts[def_idx], 3.0)

    def test_boosts_zeroed_keys_are_zero(self):
        atk_idx = GEN1_BOOST_KEYS.index("atk")
        self.assertEqual(self.ps.boosts[atk_idx], 0.0)

    def test_negative_boosts_stored_correctly(self):
        # Tauros: spa=-2 — negative stages must be preserved as-is
        ps = _StubState(make_mock_from_fixture("gen1_tauros.json"))
        spa_idx = GEN1_BOOST_KEYS.index("spa")
        self.assertEqual(ps.boosts[spa_idx], -2.0)

    def test_stats_initialised_to_zero_by_base(self):
        # PokemonState.__init__ sets stats = encode_dicts({}, STAT_KEYS) → zeros
        # subclasses are responsible for populating stats from pokemon.stats
        np.testing.assert_array_equal(self.ps.stats, np.zeros(len(GEN1_STAT_KEYS)))


# ---------------------------------------------------------------------------
# Status field — one test per status variant present in fixtures
# Starmie=PAR, Alakazam=SLP, Tauros=no status
# ---------------------------------------------------------------------------

class TestPokemonStateStatusPAR(unittest.TestCase):
    def setUp(self):
        self.ps = _StubState(make_mock_from_fixture("gen1_starmie.json"))

    def test_par_at_correct_index(self):
        par_idx = ALL_STATUSES.index(Status.PAR)
        self.assertEqual(self.ps.status[par_idx], 1.0)
        self.assertEqual(self.ps.status.sum(), 1.0)


class TestPokemonStateStatusSLP(unittest.TestCase):
    def setUp(self):
        self.ps = _StubState(make_mock_from_fixture("gen1_alakazam.json"))

    def test_slp_at_correct_index(self):
        slp_idx = ALL_STATUSES.index(Status.SLP)
        self.assertEqual(self.ps.status[slp_idx], 1.0)
        self.assertEqual(self.ps.status.sum(), 1.0)


class TestPokemonStateStatusNone(unittest.TestCase):
    def setUp(self):
        self.ps = _StubState(make_mock_from_fixture("gen1_tauros.json"))

    def test_no_status_all_zero(self):
        self.assertEqual(self.ps.status.sum(), 0.0)


# ---------------------------------------------------------------------------
# Effects field — fixtures have no active effects, inject one via mock
# Base: Starmie (real fixture), effects overridden
# ---------------------------------------------------------------------------

class TestPokemonStateEffects(unittest.TestCase):

    def test_active_effect_bit_set(self):
        mock = make_mock_from_fixture("gen1_starmie.json")
        mock.effects = {Effect.CONFUSION: MagicMock()}
        ps = _StubState(mock)
        confusion_idx = GEN1_TRACKED_EFFECTS.index(Effect.CONFUSION)
        self.assertEqual(ps.effects[confusion_idx], 1.0)

    def test_inactive_effect_bit_zero(self):
        mock = make_mock_from_fixture("gen1_starmie.json")
        mock.effects = {Effect.CONFUSION: MagicMock()}
        ps = _StubState(mock)
        encore_idx = GEN1_TRACKED_EFFECTS.index(Effect.ENCORE)
        self.assertEqual(ps.effects[encore_idx], 0.0)

    def test_no_effects_all_zero(self):
        ps = _StubState(make_mock_from_fixture("gen1_starmie.json"))
        np.testing.assert_array_equal(ps.effects, np.zeros(len(GEN1_TRACKED_EFFECTS)))


if __name__ == "__main__":
    unittest.main()
