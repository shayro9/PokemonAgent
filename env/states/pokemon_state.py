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
    """

    STAT_KEYS  = GEN1_STAT_KEYS
    BOOST_KEYS = GEN1_BOOST_KEYS

    # ------------------------------------------------------------------
    # init
    # ------------------------------------------------------------------
    def __init__(self, pokemon: Optional[Pokemon] = None):
        if pokemon is not None:
            self.hp      = pokemon.current_hp_fraction
            self.stats   = self._encode_stats(pokemon.stats,   self.STAT_KEYS)
            self.boosts  = self._encode_boosts(pokemon.boosts, self.BOOST_KEYS)
            self.status  = self._encode_status(pokemon.status)
            self.effects = self._encode_effects(pokemon.effects)
            self.types   = self._encode_types(pokemon.types)
            self.stab    = self._encode_stab(pokemon)
        else:
            self.hp      = 0.0
            self.stats   = np.zeros(len(self.STAT_KEYS),   dtype=np.float32)
            self.boosts  = np.zeros(len(self.BOOST_KEYS),  dtype=np.float32)
            self.status  = self._encode_status(None)
            self.effects = self._encode_effects({})
            self.types   = self._encode_types([])
            self.stab    = 1.5

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

    @staticmethod
    def _encode_stab(pokemon: Pokemon) -> float:
        """Calculate STAB multiplier based on pokemon"""
        return 1.5

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

    def describe(self) -> str:
        """Human-readable breakdown of the pokemon state. Useful for debugging."""
        active_status  = [ALL_STATUSES[i].name for i, v in enumerate(self.status)  if v == 1.0]
        active_effects = [TRACKED_EFFECTS[i].name for i, v in enumerate(self.effects) if v == 1.0]
        active_types   = [ALL_TYPES[i].name for i, v in enumerate(self.types) if v == 1.0]

        stat_lines = " | ".join(
            f"{k}={int(v)}" for k, v in zip(self.STAT_KEYS, self.stats)
        )
        boost_lines = " | ".join(
            f"{k}={int(v):+d}" for k, v in zip(self.BOOST_KEYS, self.boosts) if v != 0
        )

        lines = [
            f"HP            : {self.hp:.2f}",
            f"Stats         : {stat_lines}",
            f"Boosts        : {boost_lines if boost_lines else 'none'}",
            f"Status        : {active_status  if active_status  else 'none'}",
            f"Effects       : {active_effects if active_effects else 'none'}",
            f"Types         : {active_types}",
            f"STAB          : {self.stab}",
            f"Array length  : {self.array_len()}",
        ]
        return "\n".join(lines)

    def __repr__(self) -> str:
        return self.describe()

