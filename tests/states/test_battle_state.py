"""
Tests for BattleState.

Covers:
  1. array_len() formula — correct sum of all sub-component lengths
  2. to_array() integrity — correct shape, dtype float32, no NaN/Inf
  3. Bench construction — active Pokémon excluded from both sides
  4. Move availability filtering — available moves have real data;
     unavailable moves produce zero base_power
  5. Sub-state types — each sub-state is the expected class
  6. Arena state wired correctly — turn and screens come from the battle
  7. describe() — doesn't crash, returns a non-empty string
"""
import unittest
from unittest.mock import MagicMock
from dataclasses import dataclass, field

import numpy as np
from poke_env.battle.move_category import MoveCategory
from poke_env.battle.pokemon_type import PokemonType

from env.states.gen1.battle_state_gen_1 import BattleStateGen1
from env.states.gen1.arena_state_gen1 import ArenaStateGen1
from env.states.gen1.my_pokemon_state_gen_1 import MyPokemonStateGen1
from env.states.gen1.opponent_pokemon_state_gen_1 import OpponentPokemonStateGen1
from env.states.move_state import MoveState
from env.states.team_state import TeamState
from env.states.state_utils import (
    MAX_MOVES, MAX_TEAM_SIZE,
    GEN1_STAT_KEYS, GEN1_BOOST_KEYS,
)


# ---------------------------------------------------------------------------
# Shared helpers  (move mock reused from test_move_state.py conventions)
# ---------------------------------------------------------------------------

_EMPTY_BOOSTS = {k: 0 for k in GEN1_BOOST_KEYS}


def make_move_mock(
    base_power: float = 80.0,
    move_id: str = "tackle",
    move_type: PokemonType = PokemonType.NORMAL,
) -> MagicMock:
    m = MagicMock()
    m.id              = move_id
    m.base_power      = base_power
    m.accuracy        = 1.0
    m.max_pp          = 35.0
    m.priority        = 0
    m.heal            = 0.0
    m.crit_ratio      = 1.0
    m.category        = MoveCategory.PHYSICAL
    m.is_protect_move = False
    m.breaks_protect  = False
    m.status          = None
    m.boosts          = dict(_EMPTY_BOOSTS)
    m.self_boost      = dict(_EMPTY_BOOSTS)
    m.recoil          = 0.0
    m.drain           = 0.0
    m.n_hit           = 1
    m.type            = MagicMock()
    m.type.__eq__     = lambda self, other: other == move_type
    m.type.__hash__   = lambda self: hash(move_type)
    m.type.damage_multiplier = MagicMock(return_value=1.0)
    return m


@dataclass
class FakePokemon:
    """
    Minimal stand-in for a poke-env Pokemon.
    Satisfies both MyPokemonStateGen1 and OpponentPokemonStateGen1 constructors
    as well as MoveState (needs .moves and .types).
    """
    species:              str
    current_hp_fraction:  float = 1.0
    active:               bool  = False
    fainted:              bool  = False
    types:                tuple = (PokemonType.NORMAL,)

    # populated in __post_init__
    stats:        dict = field(default_factory=dict)
    boosts:       dict = field(default_factory=dict)
    base_stats:   dict = field(default_factory=dict)
    effects:      dict = field(default_factory=dict)
    status:       None = None
    stab_multiplier: float = 1.5
    preparing:    bool  = False
    must_recharge: bool = False
    protect_counter: float = -1.0

    # move dict: {move_id: Move mock}
    moves: dict = field(default_factory=dict)

    def __post_init__(self):
        if not self.stats:
            self.stats      = {k: 100 for k in GEN1_STAT_KEYS}
        if not self.boosts:
            self.boosts     = {k: 0   for k in GEN1_BOOST_KEYS}
        if not self.base_stats:
            self.base_stats = {k: 50  for k in GEN1_STAT_KEYS}
        if not self.moves:
            self.moves = {
                f"move_{i}": make_move_mock(base_power=80.0, move_id=f"move_{i}")
                for i in range(MAX_MOVES)
            }

    def __lt__(self, other: "FakePokemon") -> bool:
        return self.species < other.species


def make_battle_mock(
    my_team_species: list[str] = None,
    opp_team_species: list[str] = None,
    all_moves_available: bool = True,
) -> MagicMock:
    """
    Build a minimal battle mock.

    my_team_species:  list of species names; first entry is the active Pokémon.
    opp_team_species: same for the opponent.
    all_moves_available: when False, available_moves is empty (all moves unavailable).
    """
    if my_team_species is None:
        my_team_species = ["charizard", "pikachu", "snorlax"]
    if opp_team_species is None:
        opp_team_species = ["blastoise"]

    my_pokemons  = [FakePokemon(s, active=(i == 0)) for i, s in enumerate(my_team_species)]
    opp_pokemons = [FakePokemon(s, active=(i == 0)) for i, s in enumerate(opp_team_species)]

    my_active  = my_pokemons[0]
    opp_active = opp_pokemons[0]

    battle = MagicMock()
    battle.turn                     = 7
    battle.side_conditions          = {}
    battle.opponent_side_conditions = {}

    battle.active_pokemon          = my_active
    battle.opponent_active_pokemon = opp_active
    battle.team          = {p.species: p for p in my_pokemons}
    battle.opponent_team = {p.species: p for p in opp_pokemons}

    battle.available_moves = (
        list(my_active.moves.values()) if all_moves_available else []
    )
    return battle


# ---------------------------------------------------------------------------
# 1. array_len() formula integrity — pure classmethod, no mock needed
# ---------------------------------------------------------------------------

class TestBattleStateArrayLenFormula(unittest.TestCase):

    def test_equals_exact_sum_of_parts(self):
        expected = (
                ArenaStateGen1.array_len()
                + OpponentPokemonStateGen1.array_len()                               # opp_active (no moves)
                + MoveState.array_len() * MAX_MOVES                                  # opp_moves
                + TeamState.compute_array_len(OpponentPokemonStateGen1, 5)  # opp_bench (5 slots, no moves)
                + MyPokemonStateGen1.array_len()                                     # my_active
                + TeamState.compute_array_len(MyPokemonStateGen1, 5)        # my_bench (5 slots, no moves)
        )
        self.assertEqual(BattleStateGen1.array_len(), expected)

    def test_both_move_sides_contribute_equally(self):
        """Opp and my move blocks are the same size."""
        opp_contribution = MoveState.array_len() * MAX_MOVES
        my_contribution  = MoveState.array_len() * MAX_MOVES
        self.assertEqual(opp_contribution, my_contribution)


# ---------------------------------------------------------------------------
# 2. to_array() integration — shape, dtype, numerical health
# ---------------------------------------------------------------------------

class TestBattleStateToArray(unittest.TestCase):

    def setUp(self):
        self.bs = BattleStateGen1(make_battle_mock())

    def test_length_matches_array_len(self):
        self.assertEqual(len(self.bs.to_array()), BattleStateGen1.array_len())

    def test_dtype_float32(self):
        self.assertEqual(self.bs.to_array().dtype, np.float32)

    def test_no_nan(self):
        self.assertFalse(np.any(np.isnan(self.bs.to_array())))

    def test_no_inf(self):
        self.assertFalse(np.any(np.isinf(self.bs.to_array())))

    def test_all_values_in_range(self):
        arr = self.bs.to_array()
        self.assertTrue(np.all(arr >= -1.0) and np.all(arr <= 1.0))


# ---------------------------------------------------------------------------
# 3. Bench construction — active excluded from both sides
# ---------------------------------------------------------------------------

class TestBattleStateBench(unittest.TestCase):

    def setUp(self):
        self.battle = make_battle_mock(
            my_team_species=["charizard", "pikachu", "snorlax"],
            opp_team_species=["blastoise", "venusaur"],
        )
        self.bs = BattleStateGen1(self.battle)

    def test_my_active_not_in_my_bench(self):
        active_species = self.battle.active_pokemon.species
        bench_species  = {p.species for p in self.bs.my_bench}
        self.assertNotIn(active_species, bench_species)

    def test_opp_active_not_in_opp_bench(self):
        opp_active_species = self.battle.opponent_active_pokemon.species
        bench_species      = {p.species for p in self.bs.opp_bench}
        self.assertNotIn(opp_active_species, bench_species)

    def test_my_bench_size(self):
        # 3-member team, 1 active → 2 on bench
        self.assertEqual(len(self.bs.my_bench), 2)

    def test_opp_bench_size(self):
        # 2-member opp team, 1 active → 1 on bench
        self.assertEqual(len(self.bs.opp_bench), 1)

    def test_opp_bench_empty_when_solo_pokemon(self):
        battle = make_battle_mock(opp_team_species=["blastoise"])
        bs = BattleStateGen1(battle)
        self.assertEqual(len(bs.opp_bench), 0)


# ---------------------------------------------------------------------------
# 4. Move availability filtering
# ---------------------------------------------------------------------------

class TestBattleStateMoveEncoding(unittest.TestCase):

    def test_available_move_has_nonzero_base_power(self):
        bs = BattleStateGen1(make_battle_mock(all_moves_available=True))
        for ms in bs.my_moves_state:
            self.assertGreater(ms.base_power, 0.0)

    def test_unavailable_move_has_zero_base_power(self):
        bs = BattleStateGen1(make_battle_mock(all_moves_available=False))
        for ms in bs.my_moves_state:
            self.assertEqual(ms.base_power, 0.0)

    def test_partially_unavailable_moves(self):
        """Only the first move is available; the rest must have base_power == 0."""
        battle = make_battle_mock(all_moves_available=True)
        all_move_objects = list(battle.active_pokemon.moves.values())
        battle.available_moves = [all_move_objects[0]]  # only first move
        bs = BattleStateGen1(battle)

        self.assertGreater(bs.my_moves_state[0].base_power, 0.0)
        for ms in bs.my_moves_state[1:]:
            self.assertEqual(ms.base_power, 0.0)

    def test_my_moves_state_count_matches_pokemon_move_count(self):
        bs = BattleStateGen1(make_battle_mock())
        self.assertEqual(len(bs.my_moves_state), len(bs.my_active.moves))

    def test_opp_moves_state_count_matches_pokemon_move_count(self):
        bs = BattleStateGen1(make_battle_mock())
        self.assertEqual(len(bs.opp_moves_state), len(bs.opp_active.moves))

    def test_all_moves_unavailable_to_array_still_correct_length(self):
        bs = BattleStateGen1(make_battle_mock(all_moves_available=False))
        self.assertEqual(len(bs.to_array()), BattleStateGen1.array_len())


# ---------------------------------------------------------------------------
# 5. Sub-state types
# ---------------------------------------------------------------------------

class TestBattleStateSubStateTypes(unittest.TestCase):

    def setUp(self):
        self.bs = BattleStateGen1(make_battle_mock())

    def test_arena_state_is_arena_state(self):
        self.assertIsInstance(self.bs.arena_state, ArenaStateGen1)

    def test_my_team_state_is_team_state(self):
        self.assertIsInstance(self.bs.my_bench_state, TeamState)

    def test_opp_team_state_is_team_state(self):
        self.assertIsInstance(self.bs.opp_bench_state, TeamState)

    def test_my_moves_state_elements_are_move_states(self):
        for ms in self.bs.my_moves_state:
            self.assertIsInstance(ms, MoveState)

    def test_opp_moves_state_elements_are_move_states(self):
        for ms in self.bs.opp_moves_state:
            self.assertIsInstance(ms, MoveState)


# ---------------------------------------------------------------------------
# 6. Arena state wired correctly
# ---------------------------------------------------------------------------

class TestBattleStateArenaWiring(unittest.TestCase):

    def test_turn_propagated_to_arena_state(self):
        battle = make_battle_mock()
        bs = BattleStateGen1(battle)
        self.assertEqual(bs.arena_state.turn, battle.turn)

    def test_arena_turn_encoded_in_to_array(self):
        """First element of to_array() is the normalised turn number."""
        from env.states.gen1.arena_state_gen1 import TURN_NORM
        battle = make_battle_mock()
        bs     = BattleStateGen1(battle)
        expected_turn = min(battle.turn / TURN_NORM, 1.0)
        self.assertAlmostEqual(float(bs.to_array()[0]), expected_turn, places=5)


# ---------------------------------------------------------------------------
# 7. describe() smoke test
# ---------------------------------------------------------------------------

class TestBattleStateDescribe(unittest.TestCase):

    def test_describe_returns_string(self):
        bs = BattleStateGen1(make_battle_mock())
        self.assertIsInstance(bs.describe(), str)

    def test_describe_non_empty(self):
        bs = BattleStateGen1(make_battle_mock())
        self.assertTrue(len(bs.describe()) > 0)

    def test_repr_equals_describe(self):
        bs = BattleStateGen1(make_battle_mock())
        self.assertEqual(repr(bs), bs.describe())


if __name__ == "__main__":
    unittest.main()
