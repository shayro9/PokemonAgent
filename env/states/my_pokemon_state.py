"""
================================
Concrete PokemonState for the learning agent's own Pokémon.

Stats are known exactly (read directly from battle.active_pokemon.stats).

to_array() layout
-----------------
    hp              (1)
    stats_encoded   (6)   raw stats / STAT_NORM, clipped to [0, 1]
    boosts_encoded  (7)   boost stages / BOOST_NORM, clipped to [−1, +1]
    status          (7)
    effects         (3)
    stab            (1)   stab_multiplier
                   -----
    total           25
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from poke_env.battle.pokemon import Pokemon

from env.states.pokemon_state import (
    PokemonState,
    STAT_NORM, ALL_STATUSES, TRACKED_EFFECTS,
)


class MyPokemonState(PokemonState):
    """
    Agent-side Pokémon state.

    Extends ``PokemonState`` by adding the ``stats`` field
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(self, pokemon: Optional[Pokemon] = None) -> None:
        """Populate agent state from a poke-env Pokémon, or zero-init.

        :param pokemon: Agent's active ``Pokemon`` object, or ``None`` for an
            all-zero placeholder.
        """
        super().__init__(pokemon)

        if pokemon is not None:
            self.stats: np.ndarray = self._encode_stats(pokemon.stats, self.STAT_KEYS)
        else:
            self.stats = np.zeros(len(self.STAT_KEYS), dtype=np.float32)

    # ------------------------------------------------------------------
    # Encoding helpers
    # ------------------------------------------------------------------

    def stats_encoded(self) -> np.ndarray:
        """Normalise raw stats → [0, 1] by dividing by ``STAT_NORM``.

        :returns: Float32 array of length ``len(STAT_KEYS)``.
        """
        return np.minimum(self.stats / STAT_NORM, 1.0).astype(np.float32)

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_array(self) -> np.ndarray:
        """Return the flat float32 feature vector.

        Layout: ``[hp | stats_encoded | boosts_encoded | status | effects | stab]``
        """
        arr = np.concatenate([
            [self.hp],
            self.stats_encoded(),       # (len(STAT_KEYS),)
            self.boosts_encoded(),      # (len(BOOST_KEYS),)
            self.status,                # (len(ALL_STATUSES),)
            self.effects,               # (len(TRACKED_EFFECTS),)
            [self.stab],          # scalar, normalised to match BattleState
        ]).astype(np.float32)

        assert len(arr) == self.array_len(), (
            f"MyPokemonState.to_array(): expected {self.array_len()}, got {len(arr)}"
        )
        return arr

    def array_len(self) -> int:
        """Expected flat vector length.

        :returns: 1 + len(STAT_KEYS) + len(BOOST_KEYS) + len(ALL_STATUSES)
                    + len(TRACKED_EFFECTS) + 1
        """
        from env.states.pokemon_state import ALL_STATUSES, TRACKED_EFFECTS
        return (
            1                       # hp
            + len(self.STAT_KEYS)   # stats
            + len(self.BOOST_KEYS)  # boosts
            + len(ALL_STATUSES)     # status
            + len(TRACKED_EFFECTS)  # effects
            + 1                     # stab
        )

    def describe(self) -> str:
        """Human-readable breakdown of the pokemon state. Useful for debugging."""
        active_status  = [ALL_STATUSES[i].name for i, v in enumerate(self.status)  if v == 1.0]
        active_effects = [TRACKED_EFFECTS[i].name for i, v in enumerate(self.effects) if v == 1.0]

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
            f"Array length  : {self.array_len()}",
        ]
        return "\n".join(lines)

    def __repr__(self) -> str:
        return self.describe()
