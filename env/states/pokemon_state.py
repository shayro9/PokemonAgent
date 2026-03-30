from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
from poke_env.battle import Move
from poke_env.battle.pokemon import Pokemon
from poke_env.battle.effect import Effect

from env.states.move_state import MoveState
from env.states.state_utils import GEN1_BOOST_KEYS, ALL_STATUSES, GEN1_TRACKED_EFFECTS, GEN1_STAT_KEYS, MAX_MOVES
from env.states.state_utils import normalize, normalize_vector, encode_enum, encode_dicts, pull_attribute

# Module-level cache: survives across BattleStateGen1 reconstructions each turn.
# Key: (move_id, defending_types_tuple, attacking_types_tuple, gen)
_MOVE_STATE_CACHE: dict[tuple, MoveState] = {}

def _get_cached_move_state(move, defending_types: tuple, attacking_types: tuple, gen: int) -> MoveState:
    if move is None:
        key = (None,)
    else:
        key = (move.id, defending_types, attacking_types, gen)
    cached = _MOVE_STATE_CACHE.get(key)
    if cached is None:
        cached = MoveState(move, defending_types, attacking_types, gen)
        _MOVE_STATE_CACHE[key] = cached
    return cached


class PokemonState(ABC):
    """
    Abstract base for agent and opponent Pokémon state.

    Provides:
      - Common field population from a poke-env ``Pokemon`` object.
      - All static encoding helpers.
      - ``boosts_encoded()`` convenience method.
      - Abstract ``to_array()`` and ``array_len()``.

    Class variables (override in subclasses)
    -----------------------------------------
    STAT_KEYS   controls which stat keys are read from ``pokemon.stats``.
    BOOST_KEYS  controls which boost keys are read from ``pokemon.boosts``.
    """

    STAT_KEYS: list[str]    = GEN1_STAT_KEYS
    BOOST_KEYS: list[str]   = GEN1_BOOST_KEYS
    TRACKED_EFFECTS: list[Effect] = GEN1_TRACKED_EFFECTS

    # ------------------------------------------------------------------
    # init
    # ------------------------------------------------------------------
    def __init__(self, pokemon: Optional[Pokemon] = None):
        self.level = 100
        self.moves_states: list[MoveState] = [MoveState.zero()] * MAX_MOVES
        if pokemon is not None:
            self.hp      = pokemon.current_hp_fraction
            self.species = pokemon.species
            self.stats = self.encode_dicts(pokemon.stats, self.STAT_KEYS)
            self.boosts  = self.encode_dicts(pokemon.boosts, self.BOOST_KEYS)
            self.types   = pokemon.types
            self.status  = self.encode_enum(pokemon.status, ALL_STATUSES)
            self.effects = self.encode_enum(pokemon.effects, self.TRACKED_EFFECTS)
            self.stab    = self.pull_attribute(pokemon, "stab_multiplier", default_value=0.0, type_value=float)
            self.active  = self.pull_attribute(pokemon, "active", default_value=0.0, type_value=float)
            self.fainted = self.pull_attribute(pokemon, "fainted", default_value=0.0, type_value=float)
            self.moves   = self.pull_attribute(pokemon, "moves", default_value={}, type_value=dict)
        else:
            self.hp      = 0.0
            self.species = "none"
            self.stats = self.encode_dicts({}, self.STAT_KEYS)
            self.boosts  = self.encode_dicts({}, self.BOOST_KEYS)
            self.types   = [None]
            self.status = self.encode_enum(None, ALL_STATUSES)
            self.effects = self.encode_enum(None, self.TRACKED_EFFECTS)
            self.stab    = self.pull_attribute(None, "stab_multiplier", default_value=0.0, type_value=float)
            self.active = self.pull_attribute(None, "active", default_value=0.0, type_value=float)
            self.fainted = self.pull_attribute(None, "fainted", default_value=0.0, type_value=float)
            self.moves = self.pull_attribute(None, "moves", default_value={}, type_value=dict)

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abstractmethod
    def to_array(self) -> np.ndarray:
        """Return the flat float32 feature vector for this Pokémon."""

    @classmethod
    @abstractmethod
    def array_len(cls) -> int:
        """Return the expected length of ``to_array()``.

        Must satisfy ``len(self.to_array()) == self.array_len()`` at all times.
        """

    def encode_moves(self, defending_pokemon: Pokemon, gen=1, available_moves=None):
        """Build up to MAX_MOVES MoveState objects, zero-padded.

        MoveState objects are cached globally by (move_id, defending_types,
        attacking_types, gen) so repeated reconstruction of PokemonState each
        turn hits the cache rather than rebuilding from scratch.
        """
        defending_types = tuple(str(t) for t in defending_pokemon.types)
        attacking_types = tuple(str(t) for t in self.types)
        all_moves = list(self.moves.values()) + [None] * MAX_MOVES
        moves_list = (
            [m if m in available_moves else None for m in all_moves[:MAX_MOVES]]
            if available_moves
            else all_moves[:MAX_MOVES]
        )
        self.moves_states = [
            _get_cached_move_state(m, defending_types, attacking_types, gen)
            for m in moves_list
        ]

    @abstractmethod
    def describe(self) -> str:
        """Human-readable breakdown of the pokemon state. Useful for debugging."""

    @abstractmethod
    def __repr__(self) -> str:
        pass

    # ------------------------------------------------------------------
    # Shared encoding helpers
    # ------------------------------------------------------------------

    @staticmethod
    def normalize(x: float, max_x: float = 1.0, symmetric: bool = False) -> float:
        return normalize(x, max_x=max_x, symmetric=symmetric)

    @staticmethod
    def normalize_vector(vec, vec_max, symmetric: bool = False) -> np.ndarray:
        return normalize_vector(vec, vec_max, symmetric=symmetric)

    @staticmethod
    def encode_enum(value, enums_list) -> np.ndarray:
        return encode_enum(value, enums_list)

    @staticmethod
    def encode_dicts(_dict: dict, _keys: list[str]) -> np.ndarray:
        return encode_dicts(_dict, _keys)

    @staticmethod
    def pull_attribute(obj, key, default_value, type_value):
        return pull_attribute(obj, key, default_value, type_value)

