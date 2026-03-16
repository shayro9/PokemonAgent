"""
Abstract base class for a single Pokémon's in-battle state.

Subclasses
----------
MyPokemonState
OpponentPokemonState

Shared responsibilities (this file)
-------------------------------------
* Encoding all fields that are identical on both sides:
    hp, boosts, status, effects, types, stab
* All static encoding helpers used by both subclasses.
* Class-level STAT_KEYS / BOOST_KEYS that subclasses override for gen or role.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
from poke_env.battle.effect import Effect
from poke_env.battle.pokemon import Pokemon
from poke_env.battle.pokemon_type import PokemonType
from poke_env.battle.status import Status

# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------

ALL_TYPES       = list(PokemonType)
ALL_STATUSES    = list(Status)
TRACKED_EFFECTS = [Effect.CONFUSION, Effect.MUST_RECHARGE, Effect.ENCORE]

# Per-gen stat schemas
GEN1_STAT_KEYS      = ["hp", "atk", "def", "spc", "spe"]  # 5  (no spa/spd split)
MODERN_STAT_KEYS    = ["hp", "atk", "def", "spa", "spd", "spe"]  # 6  (Gen 2+)

GEN1_BOOST_KEYS     = ["atk", "def", "spa", "spe", "accuracy", "evasion"]  # 6
MODERN_BOOST_KEYS   = ["atk", "def", "spa", "spd", "spe", "accuracy", "evasion"]  # 7

STAT_NORM   = 512.0  # divide raw stats → [0, 1]
BOOST_NORM  = 6.0    # boost stages −6…+6 → [−1, +1]


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

    # ------------------------------------------------------------------
    # init
    # ------------------------------------------------------------------
    def __init__(self, pokemon: Optional[Pokemon] = None):
        if pokemon is not None:
            self.hp      = pokemon.current_hp_fraction
            self.species = pokemon.species
            self.boosts  = self._encode_boosts(pokemon.boosts, self.BOOST_KEYS)
            self.types   = self._encode_types(pokemon.types)
            self.status  = self._encode_status(pokemon.status)
            self.effects = self._encode_effects(pokemon.effects)
            self.stab    = self._encode_stab(pokemon)
        else:
            self.hp      = 0.0
            self.species = "none"
            self.boosts  = np.zeros(len(self.BOOST_KEYS),  dtype=np.float32)
            self.types   = np.zeros(len(ALL_TYPES), dtype=np.int32)
            self.status  = self._encode_status(None)
            self.effects = self._encode_effects({})
            self.stab    = self._encode_stab(None)

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abstractmethod
    def to_array(self) -> np.ndarray:
        """Return the flat float32 feature vector for this Pokémon."""

    @abstractmethod
    def array_len(self) -> int:
        """Return the expected length of ``to_array()``.

        Must satisfy ``len(self.to_array()) == self.array_len()`` at all times.
        """

    @abstractmethod
    def describe(self) -> str:
        """Human-readable breakdown of the pokemon state. Useful for debugging."""

    @abstractmethod
    def __repr__(self) -> str:
        pass

    # ------------------------------------------------------------------
    # Shared encoding helpers
    # ------------------------------------------------------------------

    def boosts_encoded(self) -> np.ndarray:
        """Normalise raw boost stages → [−1, +1].

        :returns: Float32 array of length ``len(BOOST_KEYS)``.
        """
        return (self.boosts / BOOST_NORM).astype(np.float32)

    @staticmethod
    def _encode_status(status) -> np.ndarray:
        """One-hot encode a status condition over ``ALL_STATUSES``.

        :param status: poke-env ``Status`` enum value, or ``None``.
        :returns: Float32 one-hot vector of length ``len(ALL_STATUSES)``.
        """
        return np.array(
            [1.0 if status == s else 0.0 for s in ALL_STATUSES],
            dtype=np.float32,
        )

    @staticmethod
    def _encode_effects(effects) -> np.ndarray:
        """Binary-encode tracked in-battle effects.

        :param effects: Mapping or set of active ``Effect`` values.
        :returns: Float32 binary vector of length ``len(TRACKED_EFFECTS)``.
        """
        return np.array(
            [float(e in effects) for e in TRACKED_EFFECTS],
            dtype=np.float32,
        )

    @staticmethod
    def _encode_types(types) -> np.ndarray:
        """One-hot encode Pokémon typing over ``ALL_TYPES``.

        Dual-typed Pokémon produce two 1s.

        :param types: Iterable of ``PokemonType`` values.
        :returns: Float32 multi-hot vector of length ``len(ALL_TYPES)``.
        """
        return np.array(
            [1.0 if t in types else 0.0 for t in ALL_TYPES],
            dtype=np.float32,
        )

    @staticmethod
    def _encode_stats(stats: dict, stat_keys: list[str]) -> np.ndarray:
        """Extract raw stat values in the given key order.

        :param stats: Mapping of stat name → raw integer value.
        :param stat_keys: Ordered list of keys to extract.
        :returns: Float32 array of raw stat values.
        """
        return np.array(
            [stats.get(k, 0) for k in stat_keys],
            dtype=np.float32,
        )

    @staticmethod
    def _encode_boosts(boosts: dict, boost_keys: list[str]) -> np.ndarray:
        """Extract raw boost stage values in the given key order.

        :param boosts: Mapping of boost name → stage integer.
        :param boost_keys: Ordered list of keys to extract.
        :returns: Float32 array of raw boost stage values.
        """
        return np.array(
            [boosts.get(k, 0) for k in boost_keys],
            dtype=np.float32,
        )

    @staticmethod
    def _encode_stab(pokemon):
        return float(getattr(pokemon, "stab_multiplier", 1.5)) / 2.0
