from __future__ import annotations
from typing import Optional

import numpy as np
from poke_env.battle.pokemon import Pokemon

from env.states.state_utils import STAT_NORM, STAB_NORM, BOOST_NORM
from env.states.pokemon_state import (
    PokemonState,
    ALL_STATUSES,
)


class OpponentPokemonState(PokemonState):
    """
    Opponent-side Pokémon state.
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(
        self,
        pokemon: Optional[Pokemon] = None,
    ) -> None:
        """
        :param pokemon: Opponent's active ``Pokemon`` object, or ``None`` for a
            zero placeholder.
        """
        super().__init__(pokemon)   # sets hp, boosts, status, effects, types, stab
        if pokemon is not None:
            self.stats          : np.ndarray = self.estimate_stats(pokemon)
            self.preparing      : float      = self.pull_attribute(pokemon, "preparing", False, bool)
            self.must_recharge  : float      = self.pull_attribute(pokemon, "must_recharge", False, bool)
            self.protect        : float      = self.pull_attribute(pokemon, "protect_counter", 0.0, float)
        else:
            self.stats          : np.ndarray = self.encode_enum(None, self.STAT_KEYS)
            self.preparing      : float      = 0.0
            self.must_recharge  : float      = 0.0
            self.protect        : float      = 0.0

    # ------------------------------------------------------------------
    # Initializations
    # ------------------------------------------------------------------
    def estimate_stats(self, pokemon: Pokemon) -> np.ndarray:
        base_stats = self.encode_dicts(pokemon.base_stats, self.STAT_KEYS)

        level = self.level
        dv      = 15
        ev_term = 64

        stats = []

        for i, base in enumerate(base_stats):
            if self.STAT_KEYS[i] == "hp":
                stat = ((base + dv) * 2 + ev_term) + level + 10
            else:
                stat = ((base + dv) * 2 + ev_term) + 5
            stats.append(stat)

        return np.array(stats, dtype=np.float32)

    # ------------------------------------------------------------------
    # Encoding helpers
    # ------------------------------------------------------------------
    def normalize_protect(self) -> np.ndarray:
        value = 0.3 ** self.protect
        return np.array([value], dtype=np.float32)

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_array(self) -> np.ndarray:
        """Return the flat float32 feature vector for the opponent.

        Layout matches the opponent portion of ``BattleState.to_array()``,
        extended with moves and protect belief so the entire opponent state
        is self-contained.
        """
        arr = np.concatenate([
            [self.hp],                  # (1)
            self.normalize_vector(self.stats, STAT_NORM),     # (6)
            self.normalize_vector(self.boosts, BOOST_NORM),    # (7)
            self.status,                # (7)
            self.effects,               # (3)
            [self.preparing],           # (1)
            [self.must_recharge],       # (1)
            [self.normalize(self.stab, STAB_NORM)],    # (1)
            self.normalize_protect(),   # (1)
        ]).astype(np.float32)

        assert len(arr) == self.array_len(), (
            f"OpponentPokemonState.to_array(): expected {self.array_len()}, "
            f"got {len(arr)}"
        )
        return arr

    def array_len(self) -> int:
        """Expected flat vector length."""
        return (
            1                       # hp
            + len(self.STAT_KEYS)   # stats
            + len(self.BOOST_KEYS)  # boosts
            + len(ALL_STATUSES)     # status
            + len(self.TRACKED_EFFECTS)  # effects
            + 1                     # preparing
            + 1                     # recharge
            + 1                     # stab
            + 1                     # protect
        )

    def describe(self) -> str:
        """Human-readable breakdown of the pokemon state. Useful for debugging."""
        active_status  = [ALL_STATUSES[i].name for i, v in enumerate(self.status)  if v == 1.0]
        active_effects = [self.TRACKED_EFFECTS[i].name for i, v in enumerate(self.effects) if v == 1.0]

        stat_lines = " | ".join(
            f"{k}={int(v)}" for k, v in zip(self.STAT_KEYS, self.stats)
        )
        boost_lines = " | ".join(
            f"{k}={int(v):+d}" for k, v in zip(self.BOOST_KEYS, self.boosts) if v != 0
        )

        lines = [
            f"Species       : {self.species}",
            f"HP            : {self.hp:.2f}",
            f"Stats         : {stat_lines}",
            f"Boosts        : {boost_lines if boost_lines else 'none'}",
            f"Status        : {active_status  if active_status  else 'none'}",
            f"Effects       : {active_effects if active_effects else 'none'}",
            f"STAB          : {self.stab}",
            f"Preparing     : {self.preparing}",
            f"MustRecharge  : {self.must_recharge}",
            f"protect       : {self.protect}",
            f"Array length  : {self.array_len()}",
        ]
        return "\n".join(lines)

    def __repr__(self) -> str:
        return self.describe()
