"""
Unit tests for Team (team_state.py).

Runs with:  pytest test_team_state.py -v

No poke-env import is required — all Pokémon and state objects are faked
with lightweight stubs so tests stay fast and dependency-free.
"""

from __future__ import annotations

import numpy as np
import pytest
from dataclasses import dataclass, field
from typing import Optional

# ---------------------------------------------------------------------------
# Minimal stubs (no poke-env required)
# ---------------------------------------------------------------------------

SLOT_LEN = 5  # fixed fake array length per Pokémon


@dataclass
class FakePokemon:
    """Stand-in for a poke-env Pokemon object."""
    species : str
    hp      : float  = 1.0
    active  : bool   = False
    fainted : bool   = False

    # Make sortable so pokemons.sort() inside Team works
    def __lt__(self, other: "FakePokemon") -> bool:
        return self.species < other.species


class FakePokemonState:
    """
    Concrete PokemonState stub.
    Mirrors the interface Team relies on:
        .species, .hp, .active, .fainted, .to_array(), .array_len()
    """

    def __init__(self, pokemon: Optional[FakePokemon] = None) -> None:
        if pokemon is not None:
            self.species = pokemon.species
            self.hp      = pokemon.hp
            self.active  = pokemon.active
            self.fainted = pokemon.fainted
        else:
            # zero / padding slot
            self.species = "none"
            self.hp      = 0.0
            self.active  = False
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
    """Return *n* FakePokemon with optional active / fainted flags."""
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


def make_team(pokemons, max_size=6):
    """Import-safe Team factory — deferred import so the stub is in place."""
    from env.states.team_state import Team
    return Team(pokemons, FakePokemonState, max_size)


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
        """Pokémon beyond max_size must be silently dropped."""
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
        """Team must sort the input list — verify lexicographic order."""
        pokemons = [FakePokemon("zebra"), FakePokemon("arcanine"), FakePokemon("magikarp")]
        team = make_team(pokemons)
        filled_species = [m.species for m in team.members if m.species != "none"]
        assert filled_species == sorted(filled_species)


# ---------------------------------------------------------------------------
# array_len & to_array shape
# ---------------------------------------------------------------------------

class TestArrayDimensions:
    def test_array_len_formula(self):
        """array_len == SLOT_LEN * max_size + max_size (alive_vector)."""
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
        # After sorting pokemon_0 < pokemon_1 < pokemon_2 → idx 1 still active
        active_pos = next(i for i, m in enumerate(team.members) if m.active)
        assert team.alive_vector[active_pos] == pytest.approx(1.0)

    def test_fainted_member_encodes_as_minus1(self):
        pokemons = make_pokemons(3, fainted_idxs=[0])
        team = make_team(pokemons)
        fainted_pos = next(i for i, m in enumerate(team.members) if m.fainted)
        assert team.alive_vector[fainted_pos] == pytest.approx(-1.0)

    def test_bench_alive_member_encodes_as_0(self):
        pokemons = make_pokemons(3, active_idx=0)  # only idx-0 active
        team = make_team(pokemons)
        bench_positions = [i for i, m in enumerate(team.members)
                           if not m.active and not m.fainted and m.species != "none"]
        for pos in bench_positions:
            assert team.alive_vector[pos] == pytest.approx(0.0)

    def test_padding_slot_encodes_as_0(self):
        team = make_team(make_pokemons(2))  # 4 padding slots
        empty_positions = [i for i, m in enumerate(team.members) if m.species == "none"]
        for pos in empty_positions:
            assert team.alive_vector[pos] == pytest.approx(0.0)

    def test_alive_vector_length_equals_max_size(self):
        max_size = 4
        team = make_team(make_pokemons(2), max_size=max_size)
        assert len(team.alive_vector) == max_size

    def test_alive_vector_is_last_segment_of_to_array(self):
        """The alive_vector must be the final max_size elements of to_array()."""
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
            assert team.alive_vector[pos] == pytest.approx(-1.0)


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
        team = make_team(make_pokemons(2))  # 4 empty slots
        assert team.alive_count() == 2

    def test_empty_team_alive_count_is_zero(self):
        team = make_team([])
        assert team.alive_count() == 0


# ---------------------------------------------------------------------------
# to_array values (content sanity)
# ---------------------------------------------------------------------------

class TestToArrayValues:
    def test_member_arrays_are_embedded(self):
        """The first SLOT_LEN * n elements must reflect the member arrays."""
        max_size = 3
        pokemons = make_pokemons(2)
        team = make_team(pokemons, max_size=max_size)
        arr = team.to_array()
        member_block = arr[: SLOT_LEN * max_size]
        expected = np.concatenate([m.to_array() for m in team.members]).astype(np.float32)
        np.testing.assert_array_equal(member_block, expected)

    def test_padding_slots_are_zero(self):
        """Zero-padded slots must emit all-zero member arrays."""
        team = make_team(make_pokemons(2), max_size=4)  # 2 padding slots
        arr = team.to_array()
        member_block = arr[: SLOT_LEN * 4]
        # Last two slots (indices 2, 3) should be zero vectors
        for slot_idx in [2, 3]:
            start = slot_idx * SLOT_LEN
            chunk = member_block[start : start + SLOT_LEN]
            np.testing.assert_array_equal(chunk, np.zeros(SLOT_LEN, dtype=np.float32))


# ---------------------------------------------------------------------------
# describe / __repr__ (smoke tests — must not raise)
# ---------------------------------------------------------------------------

class TestDescribe:
    def test_describe_does_not_raise(self):
        team = make_team(make_pokemons(3, active_idx=0, fainted_idxs=[2]))
        assert isinstance(team.describe(), str)

    def test_repr_does_not_raise(self):
        team = make_team(make_pokemons(2))
        assert isinstance(repr(team), str)

    def test_describe_contains_alive_count(self):
        team = make_team(make_pokemons(4, fainted_idxs=[0]))
        desc = team.describe()
        assert "3" in desc  # 3 alive

    def test_describe_contains_array_length(self):
        team = make_team(make_pokemons(2))
        desc = team.describe()
        assert str(team.array_len()) in desc