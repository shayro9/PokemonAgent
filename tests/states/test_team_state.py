"""
Unit tests for Team (team_state.py).

Runs with:  python test_team_state.py

No poke-env import is required for the stub-based tests — all Pokémon objects
are faked with a lightweight dataclass so tests stay fast and dependency-free.

The integration section (TestMyPokemonStateGen1Integration,
TestOpponentPokemonStateGen1Integration, TestCrossClassConsistency) imports the
real state classes and is skipped gracefully when the env package is absent.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

try:
    from env.states.team_state import TeamState
    from env.states.state_utils import GEN1_STAT_KEYS, GEN1_BOOST_KEYS
    from env.states.gen1.my_pokemon_state_gen_1 import MyPokemonStateGen1
    from env.states.gen1.opponent_pokemon_state_gen_1 import OpponentPokemonStateGen1
    _REAL_CLASSES_AVAILABLE = True
except ImportError:
    _REAL_CLASSES_AVAILABLE = False


# ---------------------------------------------------------------------------
# Stubs (no poke-env required)
# ---------------------------------------------------------------------------

SLOT_LEN = 5  # fixed fake array length per Pokémon


@dataclass
class FakePokemon:
    """Stand-in for a poke-env Pokemon object.

    The base fields (species, hp, active, fainted) are enough for FakePokemonState.
    The extra fields satisfy the real MyPokemonStateGen1 / OpponentPokemonStateGen1
    constructors, so the same class can be reused for integration tests without
    a separate stub.
    """
    species: str
    hp: float = 1.0
    active: bool = False
    fainted: bool = False

    # poke-env attribute names used by PokemonState base
    current_hp_fraction: float = 1.0
    status: None = None
    effects: dict = field(default_factory=dict)
    types: list = field(default_factory=lambda: [None])
    stab_multiplier: float = 1.5
    level: int = 100

    # stats / boosts — populated from real key lists when available, else empty
    stats: dict = field(default_factory=dict)
    boosts: dict = field(default_factory=dict)

    # base_stats used by OpponentPokemonStateGen1.estimate_stats
    base_stats: dict = field(default_factory=dict)

    # opponent-only flags
    preparing: bool = False
    must_recharge: bool = False
    protect_counter: float = 0.0

    def __post_init__(self):
        # Keep current_hp_fraction in sync with hp so both attribute names work.
        self.current_hp_fraction = self.hp
        if _REAL_CLASSES_AVAILABLE:
            self.stats = {k: 100 for k in GEN1_STAT_KEYS}
            self.boosts = {k: 0 for k in GEN1_BOOST_KEYS}
            self.base_stats = {k: 50 for k in GEN1_STAT_KEYS}

    def __lt__(self, other: "FakePokemon") -> bool:
        return self.species < other.species


class FakePokemonState:
    """Concrete PokemonState stub — mirrors the interface TeamState relies on."""

    def __init__(self, pokemon: Optional[FakePokemon] = None) -> None:
        if pokemon is not None:
            self.species = pokemon.species
            self.hp = pokemon.hp
            self.active = pokemon.active
            self.fainted = pokemon.fainted
        else:
            self.species = "none"
            self.hp = 0.0
            self.active = False
            self.fainted = False

    def to_array(self) -> np.ndarray:
        return np.full(SLOT_LEN, self.hp, dtype=np.float32)

    def array_len(self) -> int:
        return SLOT_LEN


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_pokemons(
    n: int,
    *,
    active_idx: Optional[int] = None,
    fainted_idxs: Optional[list[int]] = None,
) -> list[FakePokemon]:
    fainted_idxs = fainted_idxs or []
    return [
        FakePokemon(
            species=f"pokemon_{i}",
            hp=0.0 if i in fainted_idxs else 1.0,
            active=(i == active_idx),
            fainted=(i in fainted_idxs),
        )
        for i in range(n)
    ]


def make_team(pokemons, state_cls=FakePokemonState, max_size=6):
    return TeamState(pokemons, state_cls, max_size)


def _skip_if_unavailable() -> bool:
    if not _REAL_CLASSES_AVAILABLE:
        warnings.warn(
            "Skipping real-class integration tests: env package not importable.",
            stacklevel=2,
        )
        return True
    return False


# ===========================================================================
# Stub-based unit tests
# ===========================================================================

# ---------------------------------------------------------------------------
# Construction & slot count
# ---------------------------------------------------------------------------

class TestConstruction:
    def test_full_team_no_padding(self):
        team = make_team(make_pokemons(6))
        non_empty = [m for m in team.members if m.species != "none"]
        assert len(non_empty) == 6

    def test_partial_team_is_padded(self):
        team = make_team(make_pokemons(3))
        empty = [m for m in team.members if m.species == "none"]
        assert len(empty) == 3

    def test_empty_team_all_padding(self):
        team = make_team([])
        assert all(m.species == "none" for m in team.members)

    def test_team_truncated_to_max_size(self):
        team = make_team(make_pokemons(10))
        assert len(team.members) == 6

    def test_custom_max_size(self):
        team = make_team(make_pokemons(2), max_size=3)
        assert len(team.members) == 3
        empty = [m for m in team.members if m.species == "none"]
        assert len(empty) == 1

    def test_max_size_one_pokemon(self):
        team = make_team(make_pokemons(1), max_size=1)
        assert len(team.members) == 1
        assert team.members[0].species == "pokemon_0"

    def test_pokemons_are_sorted(self):
        pokemons = [
            FakePokemon("zebra"),
            FakePokemon("arcanine"),
            FakePokemon("magikarp"),
        ]
        team = make_team(pokemons)
        filled_species = [m.species for m in team.members if m.species != "none"]
        assert filled_species == sorted(filled_species)


# ---------------------------------------------------------------------------
# array_len & to_array shape
# ---------------------------------------------------------------------------

class TestArrayDimensions:
    def test_array_len_formula(self):
        max_size = 6
        team = make_team(make_pokemons(4), max_size=max_size)
        assert team.array_len() == SLOT_LEN * max_size + max_size

    def test_to_array_length_matches_array_len(self):
        team = make_team(make_pokemons(3))
        assert len(team.to_array()) == team.array_len()

    def test_to_array_dtype(self):
        team = make_team(make_pokemons(3))
        assert team.to_array().dtype == np.float32

    def test_to_array_custom_max_size(self):
        max_size = 3
        team = make_team(make_pokemons(2), max_size=max_size)
        assert len(team.to_array()) == team.array_len()
        assert team.array_len() == SLOT_LEN * max_size + max_size

    def test_to_array_empty_team(self):
        team = make_team([], max_size=6)
        assert len(team.to_array()) == team.array_len()


# ---------------------------------------------------------------------------
# alive_vector values
# ---------------------------------------------------------------------------

class TestAliveVector:
    def test_active_member_encodes_as_1(self):
        pokemons = make_pokemons(3, active_idx=1)
        team = make_team(pokemons)
        active_pos = next(i for i, m in enumerate(team.members) if m.active)
        assert team.alive_vector[active_pos] == 1.0

    def test_fainted_member_encodes_as_minus1(self):
        pokemons = make_pokemons(3, fainted_idxs=[0])
        team = make_team(pokemons)
        fainted_pos = next(i for i, m in enumerate(team.members) if m.fainted)
        assert team.alive_vector[fainted_pos] == -1.0

    def test_bench_alive_member_encodes_as_0(self):
        pokemons = make_pokemons(3, active_idx=0)
        team = make_team(pokemons)
        bench_positions = [
            i for i, m in enumerate(team.members)
            if not m.active and not m.fainted and m.species != "none"
        ]
        for pos in bench_positions:
            assert team.alive_vector[pos] == 0.0

    def test_padding_slot_encodes_as_0(self):
        team = make_team(make_pokemons(2))
        empty_positions = [i for i, m in enumerate(team.members) if m.species == "none"]
        for pos in empty_positions:
            assert team.alive_vector[pos] == 0.0

    def test_alive_vector_length_equals_max_size(self):
        max_size = 4
        team = make_team(make_pokemons(2), max_size=max_size)
        assert len(team.alive_vector) == max_size

    def test_alive_vector_is_last_segment_of_to_array(self):
        max_size = 6
        team = make_team(make_pokemons(3, active_idx=1, fainted_idxs=[2]))
        arr = team.to_array()
        tail = arr[-max_size:]
        np.testing.assert_array_equal(tail, team.alive_vector)

    def test_all_fainted_alive_vector(self):
        pokemons = make_pokemons(3, fainted_idxs=[0, 1, 2])
        team = make_team(pokemons)
        filled_positions = [i for i, m in enumerate(team.members) if m.species != "none"]
        for pos in filled_positions:
            assert team.alive_vector[pos] == -1.0


# ---------------------------------------------------------------------------
# alive_count
# ---------------------------------------------------------------------------

class TestAliveCount:
    def test_all_alive(self):
        team = make_team(make_pokemons(4))
        assert team.alive_count() == 4

    def test_none_alive(self):
        team = make_team(make_pokemons(3, fainted_idxs=[0, 1, 2]))
        assert team.alive_count() == 0

    def test_mixed_alive_and_fainted(self):
        team = make_team(make_pokemons(4, fainted_idxs=[0, 2]))
        assert team.alive_count() == 2

    def test_padding_not_counted(self):
        team = make_team(make_pokemons(2))
        assert team.alive_count() == 2

    def test_empty_team_alive_count_is_zero(self):
        team = make_team([])
        assert team.alive_count() == 0


# ---------------------------------------------------------------------------
# to_array values
# ---------------------------------------------------------------------------

class TestToArrayValues:
    def test_member_arrays_are_embedded(self):
        max_size = 3
        pokemons = make_pokemons(2)
        team = make_team(pokemons, max_size=max_size)
        arr = team.to_array()
        member_block = arr[: SLOT_LEN * max_size]
        expected = np.concatenate([m.to_array() for m in team.members]).astype(np.float32)
        np.testing.assert_array_equal(member_block, expected)

    def test_padding_slots_are_zero(self):
        team = make_team(make_pokemons(2), max_size=4)
        arr = team.to_array()
        member_block = arr[: SLOT_LEN * 4]
        for slot_idx in [2, 3]:
            start = slot_idx * SLOT_LEN
            chunk = member_block[start : start + SLOT_LEN]
            np.testing.assert_array_equal(chunk, np.zeros(SLOT_LEN, dtype=np.float32))


# ---------------------------------------------------------------------------
# describe / __repr__
# ---------------------------------------------------------------------------

class TestDescribe:
    def test_repr_does_not_raise(self):
        team = make_team(make_pokemons(2))
        assert isinstance(repr(team), str)

    def test_describe_contains_alive_count(self):
        team = make_team(make_pokemons(4, fainted_idxs=[0]))
        desc = team.describe()
        assert "3" in desc

    def test_describe_contains_array_length(self):
        team = make_team(make_pokemons(2))
        desc = team.describe()
        assert str(team.array_len()) in desc


# ===========================================================================
# Integration tests — real MyPokemonStateGen1 / OpponentPokemonStateGen1
# ===========================================================================

# ---------------------------------------------------------------------------
# MyPokemonStateGen1
# ---------------------------------------------------------------------------

class TestMyPokemonStateGen1Integration:
    """TeamState behaviour when slots hold MyPokemonStateGen1 objects."""

    def test_to_array_length_matches_array_len(self):
        if _skip_if_unavailable():
            return
        team = make_team(make_pokemons(4), MyPokemonStateGen1)
        assert len(team.to_array()) == team.array_len()

    def test_to_array_dtype_is_float32(self):
        if _skip_if_unavailable():
            return
        team = make_team(make_pokemons(3), MyPokemonStateGen1)
        assert team.to_array().dtype == np.float32

    def test_slot_len_is_consistent(self):
        """Every member must report the same array_len as the first slot."""
        if _skip_if_unavailable():
            return
        team = make_team(make_pokemons(6), MyPokemonStateGen1)
        expected = team.members[0].array_len()
        for member in team.members:
            assert member.array_len() == expected

    def test_partial_team_padded_correctly(self):
        if _skip_if_unavailable():
            return
        team = make_team(make_pokemons(3), MyPokemonStateGen1, max_size=6)
        empty = [m for m in team.members if m.species == "none"]
        assert len(empty) == 3

    def test_empty_padding_slots_are_zero(self):
        if _skip_if_unavailable():
            return
        max_size = 6
        team = make_team(make_pokemons(2), MyPokemonStateGen1, max_size=max_size)
        slot_len = team._slot_len
        arr = team.to_array()
        member_block = arr[: slot_len * max_size]
        for slot_idx in range(2, max_size):
            chunk = member_block[slot_idx * slot_len : (slot_idx + 1) * slot_len]
            np.testing.assert_array_equal(chunk, np.zeros(slot_len, dtype=np.float32))

    def test_alive_vector_appended(self):
        if _skip_if_unavailable():
            return
        max_size = 6
        team = make_team(make_pokemons(3, active_idx=0), MyPokemonStateGen1, max_size=max_size)
        tail = team.to_array()[-max_size:]
        np.testing.assert_array_equal(tail, team.alive_vector)

    def test_active_pokemon_in_alive_vector(self):
        if _skip_if_unavailable():
            return
        team = make_team(make_pokemons(3, active_idx=1), MyPokemonStateGen1)
        active_pos = next(i for i, m in enumerate(team.members) if m.active)
        assert team.alive_vector[active_pos] == 1.0

    def test_fainted_pokemon_in_alive_vector(self):
        if _skip_if_unavailable():
            return
        team = make_team(make_pokemons(3, fainted_idxs=[2]), MyPokemonStateGen1)
        fainted_pos = next(i for i, m in enumerate(team.members) if m.fainted)
        assert team.alive_vector[fainted_pos] == -1.0

    def test_alive_count(self):
        if _skip_if_unavailable():
            return
        team = make_team(make_pokemons(4, fainted_idxs=[0, 1]), MyPokemonStateGen1)
        assert team.alive_count() == 2

    def test_describe_does_not_raise(self):
        if _skip_if_unavailable():
            return
        team = make_team(make_pokemons(3, active_idx=0), MyPokemonStateGen1)
        assert isinstance(team.describe(), str)

    def test_member_describe_does_not_raise(self):
        if _skip_if_unavailable():
            return
        team = make_team(make_pokemons(2), MyPokemonStateGen1)
        for member in team.members:
            assert isinstance(member.describe(), str)

    def test_hp_is_normalised_between_0_and_1(self):
        if _skip_if_unavailable():
            return
        team = make_team(make_pokemons(3), MyPokemonStateGen1)
        for member in team.members:
            assert 0.0 <= member.hp <= 1.0


# ---------------------------------------------------------------------------
# OpponentPokemonStateGen1
# ---------------------------------------------------------------------------

class TestOpponentPokemonStateGen1Integration:
    """TeamState behaviour when slots hold OpponentPokemonStateGen1 objects."""

    def test_to_array_length_matches_array_len(self):
        if _skip_if_unavailable():
            return
        team = make_team(make_pokemons(4), OpponentPokemonStateGen1)
        assert len(team.to_array()) == team.array_len()

    def test_to_array_dtype_is_float32(self):
        if _skip_if_unavailable():
            return
        team = make_team(make_pokemons(3), OpponentPokemonStateGen1)
        assert team.to_array().dtype == np.float32

    def test_slot_len_is_consistent(self):
        if _skip_if_unavailable():
            return
        team = make_team(make_pokemons(6), OpponentPokemonStateGen1)
        expected = team.members[0].array_len()
        for member in team.members:
            assert member.array_len() == expected

    def test_partial_team_padded_correctly(self):
        if _skip_if_unavailable():
            return
        team = make_team(make_pokemons(2), OpponentPokemonStateGen1, max_size=6)
        empty = [m for m in team.members if m.species == "none"]
        assert len(empty) == 4

    def test_empty_padding_slots_are_zero(self):
        if _skip_if_unavailable():
            return
        max_size = 6
        team = make_team(make_pokemons(2), OpponentPokemonStateGen1, max_size=max_size)
        slot_len = team._slot_len
        arr = team.to_array()
        member_block = arr[: slot_len * max_size]
        for slot_idx in range(2, max_size):
            chunk = member_block[slot_idx * slot_len : (slot_idx + 1) * slot_len]
            np.testing.assert_array_equal(chunk, np.zeros(slot_len, dtype=np.float32))

    def test_alive_vector_appended(self):
        if _skip_if_unavailable():
            return
        max_size = 6
        team = make_team(make_pokemons(3, fainted_idxs=[1]), OpponentPokemonStateGen1, max_size=max_size)
        tail = team.to_array()[-max_size:]
        np.testing.assert_array_equal(tail, team.alive_vector)

    def test_active_pokemon_in_alive_vector(self):
        if _skip_if_unavailable():
            return
        team = make_team(make_pokemons(3, active_idx=2), OpponentPokemonStateGen1)
        active_pos = next(i for i, m in enumerate(team.members) if m.active)
        assert team.alive_vector[active_pos] == 1.0

    def test_fainted_pokemon_in_alive_vector(self):
        if _skip_if_unavailable():
            return
        team = make_team(make_pokemons(3, fainted_idxs=[0]), OpponentPokemonStateGen1)
        fainted_pos = next(i for i, m in enumerate(team.members) if m.fainted)
        assert team.alive_vector[fainted_pos] == -1.0

    def test_alive_count(self):
        if _skip_if_unavailable():
            return
        team = make_team(make_pokemons(5, fainted_idxs=[1, 3]), OpponentPokemonStateGen1)
        assert team.alive_count() == 3

    def test_protect_counter_default_is_zero(self):
        if _skip_if_unavailable():
            return
        team = make_team(make_pokemons(1), OpponentPokemonStateGen1, max_size=1)
        assert team.members[0].protect == 0.0

    def test_preparing_and_recharge_default_false(self):
        if _skip_if_unavailable():
            return
        team = make_team(make_pokemons(1), OpponentPokemonStateGen1, max_size=1)
        assert team.members[0].preparing == 0.0
        assert team.members[0].must_recharge == 0.0

    def test_describe_does_not_raise(self):
        if _skip_if_unavailable():
            return
        team = make_team(make_pokemons(3, fainted_idxs=[0]), OpponentPokemonStateGen1)
        assert isinstance(team.describe(), str)

    def test_member_describe_does_not_raise(self):
        if _skip_if_unavailable():
            return
        team = make_team(make_pokemons(2), OpponentPokemonStateGen1)
        for member in team.members:
            assert isinstance(member.describe(), str)

    def test_hp_is_normalised_between_0_and_1(self):
        if _skip_if_unavailable():
            return
        team = make_team(make_pokemons(3), OpponentPokemonStateGen1)
        for member in team.members:
            assert 0.0 <= member.hp <= 1.0


# ---------------------------------------------------------------------------
# Cross-class consistency
# ---------------------------------------------------------------------------

class TestCrossClassConsistency:
    """Invariants that must hold for both state classes."""

    def test_array_len_formula_holds_for_both_classes(self):
        """array_len() == slot_len * max_size + max_size for both classes."""
        if _skip_if_unavailable():
            return
        for cls in (MyPokemonStateGen1, OpponentPokemonStateGen1):
            team = make_team(make_pokemons(4), cls, max_size=6)
            assert team.array_len() == team._slot_len * 6 + 6

    def test_none_pokemon_gives_zero_array_for_both_classes(self):
        """A placeholder (None) slot must produce an all-zero feature vector."""
        if _skip_if_unavailable():
            return
        for cls in (MyPokemonStateGen1, OpponentPokemonStateGen1):
            placeholder = cls(None)
            np.testing.assert_array_equal(
                placeholder.to_array(),
                np.zeros(placeholder.array_len(), dtype=np.float32),
            )

    def test_full_team_no_padding_for_both_classes(self):
        if _skip_if_unavailable():
            return
        for cls in (MyPokemonStateGen1, OpponentPokemonStateGen1):
            team = make_team(make_pokemons(6), cls)
            assert all(m.species != "none" for m in team.members)

    def test_alive_vector_length_equals_max_size_for_both_classes(self):
        if _skip_if_unavailable():
            return
        for cls in (MyPokemonStateGen1, OpponentPokemonStateGen1):
            max_size = 4
            team = make_team(make_pokemons(2), cls, max_size=max_size)
            assert len(team.alive_vector) == max_size