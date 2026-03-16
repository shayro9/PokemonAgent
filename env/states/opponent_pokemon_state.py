"""
======================================
Concrete PokemonState for the opponent's Pokémon.

Stats are NOT known exactly — replaced by a Gaussian posterior belief
(``StatBelief``, 12 dims: mean × 6 + std × 6, all / STAT_NORM).

Extra fields compared to ``MyPokemonState``
--------------------------------------------
* stat_belief       (12) — Bayesian posterior mean + std for all 6 stats
* preparing         (1)  — opponent is charging a two-turn move
* protect_belief    (1)  — P(opponent uses Protect this turn)

to_array() layout
-----------------
    hp                  (1)
    stat_belief         (12)
    boosts_encoded      (7)
    status              (7)
    effects             (3)
    preparing           (1)
    stab                (1)   stab_multiplier / 2.0
    protect_belief      (1)
                       -----
    total               33
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from poke_env.battle import Battle
from poke_env.battle.pokemon import Pokemon

from combat.stats_belief import StatBelief
from env.embed import (
    MAX_MOVES,
    MOVE_EMBED_LEN,
    embed_move,
    calc_types_vector,
)
from env.states.pokemon_state import (
    PokemonState,
    ALL_STATUSES,
    TRACKED_EFFECTS,
)

_STAT_BELIEF_DIM = 12


class OpponentPokemonState(PokemonState):
    """
    Opponent-side Pokémon state.

    Differs from ``MyPokemonState`` in two fundamental ways:
      1. Stats are unknown — represented by a Bayesian ``StatBelief``
         (12-dim normalised array) instead of exact values.
      2. Several opponent-specific fields are added: preparing flag,
         protect belief.

    The ``stats`` field present on ``MyPokemonState`` is NOT populated
    use ``stat_belief`` for any stat-dependent logic.
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(
        self,
        pokemon: Optional[Pokemon] = None,
        *,
        stat_belief: Optional[StatBelief] = None,
        protect_belief: float = 0.0,
    ) -> None:
        """
        :param pokemon: Opponent's active ``Pokemon`` object, or ``None`` for a
            zero placeholder.
        :param stat_belief: Current ``StatBelief`` posterior over the opponent's
            six stats.  When ``None``, the belief vector is all zeros.
        :param protect_belief: Scalar in [0, 1] — probability the opponent will
            use Protect this turn, from ``ProtectBelief.expected_next_protect_belief()``.
        """
        super().__init__(pokemon)   # sets hp, boosts, status, effects, types, stab

        # ── Stat belief ──────────────────────────────────────────────────
        _belief: np.ndarray = (
            stat_belief.to_array()
            if stat_belief is not None
            else np.zeros(_STAT_BELIEF_DIM, dtype=np.float32)
        )
        self.stats: np.ndarray      = _belief[:_STAT_BELIEF_DIM // 2]
        self.stats_std: np.ndarray  = _belief[_STAT_BELIEF_DIM // 2:]

        # ── Opponent-specific flags ──────────────────────────────────────
        self.preparing: float = (
            float(getattr(pokemon, "preparing", False))
            if pokemon is not None else 0.0
        )

        # ── Protect belief ───────────────────────────────────────────────
        self.protect_belief: float = float(protect_belief)

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
            self.stats,                 # (6)
            self.stats_std,             # (6)
            self.boosts_encoded(),      # (7)
            self.status,                # (7)
            self.effects,               # (3)
            [self.preparing],           # (1)
            [self.stab],          # (1)
            [self.protect_belief],      # (1)
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
            + _STAT_BELIEF_DIM      # stat_belief
            + len(self.BOOST_KEYS)  # boosts
            + len(ALL_STATUSES)     # status
            + len(TRACKED_EFFECTS)  # effects
            + 1                     # preparing
            + 1                     # stab
            + 1                     # protect_belief
        )

    def describe(self) -> str:
        """Human-readable breakdown of the pokemon state. Useful for debugging."""
        active_status  = [ALL_STATUSES[i].name for i, v in enumerate(self.status)  if v == 1.0]
        active_effects = [TRACKED_EFFECTS[i].name for i, v in enumerate(self.effects) if v == 1.0]

        stat_lines = " | ".join(
            f"{k}={int(v)}" for k, v in zip(self.STAT_KEYS, self.stats)
        )
        stat_std_lines = " | ".join(
            f"{k}={int(v)}" for k, v in zip(self.STAT_KEYS, self.stats_std)
        )
        boost_lines = " | ".join(
            f"{k}={int(v):+d}" for k, v in zip(self.BOOST_KEYS, self.boosts) if v != 0
        )

        lines = [
            f"Species       : {self.species}",
            f"HP            : {self.hp:.2f}",
            f"Stats         : {stat_lines}",
            f"Stats STD     : {stat_std_lines}",
            f"Boosts        : {boost_lines if boost_lines else 'none'}",
            f"Status        : {active_status  if active_status  else 'none'}",
            f"Effects       : {active_effects if active_effects else 'none'}",
            f"STAB          : {self.stab}",
            f"Preparing     : {self.preparing}",
            f"protect       : {self.protect_belief}",
            f"Array length  : {self.array_len()}",
        ]
        return "\n".join(lines)

    def __repr__(self) -> str:
        return self.describe()
