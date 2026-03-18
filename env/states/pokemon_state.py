from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
from poke_env.battle.pokemon import Pokemon
from poke_env.battle.pokemon_type import PokemonType

from env.states.state_utils import encode_enum, GEN1_BOOST_KEYS, ALL_STATUSES, GEN1_TRACKED_EFFECTS, GEN1_STAT_KEYS
# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------

STAT_NORM   = 600.0
BOOST_NORM  = 6.0
STAB_NORM   = 2.25


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
    TRACKED_EFFECTS: list[str] = GEN1_TRACKED_EFFECTS

    # ------------------------------------------------------------------
    # init
    # ------------------------------------------------------------------
    def __init__(self, pokemon: Optional[Pokemon] = None):
        self.level = 100
        self.stats = np.zeros(len(self.STAT_KEYS), dtype=np.float32)
        if pokemon is not None:
            self.hp      = pokemon.current_hp_fraction
            self.species = pokemon.species
            self.boosts  = encode_enum(pokemon.boosts, self.BOOST_KEYS)
            self.types   = pokemon.types
            self.status  = encode_enum(pokemon.status, ALL_STATUSES)
            self.effects = encode_enum(pokemon.effects, self.TRACKED_EFFECTS)
            self.stab    = self._encode_stab(pokemon)
        else:
            self.hp      = 0.0
            self.species = "none"
            self.boosts  = np.zeros(len(self.BOOST_KEYS),  dtype=np.float32)
            self.types   = [None]
            self.status = encode_enum(None, ALL_STATUSES)
            self.effects = encode_enum(None, self.TRACKED_EFFECTS)
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

    def normalize_boosts(self) -> np.ndarray:
        """Normalise raw boost stages → [−1, +1].

        :returns: Float32 array of length ``len(BOOST_KEYS)``.
        """
        return (self.boosts / BOOST_NORM).astype(np.float32)

    def normalize_stats(self) -> np.ndarray:
        """Normalise raw stats → [0, 1] by dividing by ``STAT_NORM``.

        :returns: Float32 array of length ``len(STAT_KEYS)``.
        """
        return np.minimum(self.stats / STAT_NORM, 1.0).astype(np.float32)

    def normalize_stab(self) -> np.ndarray:
        return np.minimum(self.stab / STAB_NORM, 1.0).astype(np.float32)

    # ------------------------------------------------------------------
    # Encoders
    # ------------------------------------------------------------------

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
    def _encode_stab(pokemon):
        return float(getattr(pokemon, "stab_multiplier", 1.5))
