from __future__ import annotations

import numpy as np
from poke_env.battle.pokemon_type import PokemonType
from poke_env.battle.status import Status
from poke_env.battle.effect import Effect
from poke_env.battle.pokemon import Pokemon
from typing import Optional


ALL_TYPES    = list(PokemonType)
ALL_STATUSES = list(Status)
TRACKED_EFFECTS = [Effect.CONFUSION, Effect.MUST_RECHARGE, Effect.ENCORE]

# ── per-gen stat schemas ────────────────────────────────────────────────────
GEN1_STAT_KEYS   = ["hp", "atk", "def", "spc", "spe"]         # 5 stats – no spa/spd split
MODERN_STAT_KEYS = ["hp", "atk", "def", "spa", "spd", "spe"]  # 6 stats – Gen 2+  (spa/spd split)

GEN1_BOOST_KEYS   = ["atk", "def", "spa", "spe", "accuracy", "evasion"]  # poke-env uses spa even in gen1
MODERN_BOOST_KEYS = ["atk", "def", "spa", "spd", "spe", "accuracy", "evasion"]

STAT_NORM  = 512.0  # divide raw stats by this → [0, 1]
BOOST_NORM = 6.0    # boost stages range −6..+6 → [−1, +1]


class PokemonState:
    """
    Holds the HP, normalized stats, boost stages, status, effects and types
    for one Pokémon on one side of the field.

        Gen 1    →  PokemonState(gen1=True,  pokemon=my_pokemon)
        Gen 2–9+ →  PokemonState(gen1=False, pokemon=my_pokemon)
    """

    # ------------------------------------------------------------------
    # init
    # ------------------------------------------------------------------
    def __init__(self, gen1: bool = False, pokemon: Optional[Pokemon] = None):
        stat_keys = GEN1_STAT_KEYS if gen1 else MODERN_STAT_KEYS
        boost_keys = GEN1_BOOST_KEYS if gen1 else MODERN_BOOST_KEYS

        if pokemon is not None:
            self.hp = pokemon.current_hp_fraction
            self.stats = self._encode_stats(pokemon.stats, stat_keys)
            self.boosts = self._encode_boosts(pokemon.boosts, boost_keys)
            self.status = self._encode_status(pokemon.status)
            self.effects = self._encode_effects(pokemon.effects)
            self.types = self._encode_types(pokemon.types)
            self.stab = 1.5 if gen1 else pokemon.stab_multiplier
        else:
            self.hp = 0.0
            self.stats = np.zeros(len(stat_keys), dtype=np.float32)
            self.boosts = np.zeros(len(boost_keys), dtype=np.float32)
            self.status = self._encode_status(None)
            self.effects = self._encode_effects({})
            self.types = self._encode_types([])
            self.stab = 1.5

    # ------------------------------------------------------------------
    # static embedding helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _encode_status(status) -> np.ndarray:
        return np.array([1.0 if status == s else 0.0 for s in ALL_STATUSES], dtype=np.float32)

    @staticmethod
    def _encode_effects(effects) -> np.ndarray:
        return np.array([int(e in effects) for e in TRACKED_EFFECTS], dtype=np.float32)

    @staticmethod
    def _encode_types(types) -> np.ndarray:
        """One-hot over all PokemonType values. Dual-typed pokemon have two 1s."""
        return np.array([1.0 if t in types else 0.0 for t in ALL_TYPES], dtype=np.float32)

    @staticmethod
    def _encode_stats(stats: dict, stat_keys: list[str]) -> np.ndarray:
        """Extract raw stats by key, no normalization."""
        return np.array([stats.get(k, 0) for k in stat_keys], dtype=np.float32)

    @staticmethod
    def _encode_boosts(boosts: dict, boost_keys: list[str]) -> np.ndarray:
        """Extract raw boost stages by key."""
        return np.array([boosts.get(k, 0) for k in boost_keys], dtype=np.float32)

    # ------------------------------------------------------------------
    # encoding helpers
    # ------------------------------------------------------------------
    def boosts_encoded(self) -> np.ndarray:
        """Normalize boost stages → [−1, +1]."""
        return (self.boosts / BOOST_NORM).astype(np.float32)

    def stats_encoded(self) -> np.ndarray:
        """Normalize raw stats → [0, 1]."""
        return np.minimum(self.stats / STAT_NORM, 1.0).astype(np.float32)

    def to_array(self) -> np.ndarray:
        """
        Flat encoded array: [hp, *stats_encoded, *boosts_encoded, *status, *effects, *types, stab]
        """
        return np.concatenate([
            [self.hp],
            self.stats_encoded(),
            self.boosts_encoded(),
            self.status,
            self.effects,
            self.types,
            [self.stab],
        ]).astype(np.float32)

    def array_len(self) -> int:
        # +1 fo hp and +1 for stab, the rest are variable-length lists
        return 1 + len(self.stats) + len(self.boosts) + len(self.status) + len(self.effects) + len(self.types) + 1
